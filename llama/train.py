import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    set_seed,
    AutoModelForSequenceClassification
)
from peft import (
    LoraConfig,
    TaskType
)

from utils import *


def main():
    # ---------------------------- Parse Arguments and constants ----------------------------
    args = parse_training_args()

    DEBUG = args.debug
    LLAMA = True
    trainer_callbacks = []
    SLURM_TMPDIR = "TMPDIR"
    set_seed(42)

    DTYPE, USE_FP16, USE_BF16 = set_dtype(args.mixed_precision)
    TASK = args.task_type
    print(f"▶️ Finetuning task type: {TASK}")

    if TASK == "seq_cls":
        FL_ALPHA = 2
        FL_GAMMA = 5
        RECALL_AT_TOP_K_PERCENTAGES = [0.05, 0.1, 0.3]

    TASK_TO_MODEL_CLASS = {
        "clm": AutoModelForCausalLM,
        "seq_cls": AutoModelForSequenceClassification,
    }

    TASK_TO_TRAINER_CLASS = {
        "clm": Trainer,
        "seq_cls": CustomTrainer,
    }

    ModelClass = TASK_TO_MODEL_CLASS[TASK]
    TrainerClass = TASK_TO_TRAINER_CLASS[TASK]

    # ---------------------------- distributed setup  ----------------------------
    local_rank = os.environ.get("LOCAL_RANK", 0)
    world_size = os.environ.get("WORLD_SIZE", 1)
    print(f"🚀 Local rank: {local_rank} | World size: {world_size}")

    # ---------------------------- handle directories  ----------------------------

    REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print(f"✅ Detected REPO_PATH: {REPO_PATH}")

    paths = setup_training_directories(REPO_PATH, SLURM_TMPDIR, args.continue_from_dir)

    output_dir = paths["output_dir"]
    run_timestamp = paths["run_timestamp"]
    metrics_dir = paths["metrics_dir"]
    tensorboard_dir = paths["tensorboard_dir"]
    config_path = paths["config_path"]
    live_metrics_path = paths["live_metrics_path"]
    finetuned_model_dir = paths["model_dir"]
    finetuned_tokenizer_dir = paths["tokenizer_dir"]
    offload_dir = paths["offload_dir"]
    slurm_tmpdir = paths["slurm_tmpdir"]

    # ------------------------- Local model path -------------------------
    MODEL_PATH = args.model_path
    print(f"✅ Using provided MODEL_PATH: {MODEL_PATH}")

    # ------------------------- HF login -------------------------
    # login_to_huggingface(REPO_PATH)

    # ------------------------- Register custom llama -------------------------
    if TASK == "seq_cls":
        register_custom_llama4_if_needed(MODEL_PATH)

    # ------------------------- define metrics -------------------------
    def custom_metrics_seq_cls(eval_pred):
        return compute_custom_metrics_seq_cls(eval_pred, threshold=args.threshold, percentages=RECALL_AT_TOP_K_PERCENTAGES)

    trainer_callbacks.extend(
        setup_live_metrics(args.live_metrics, live_metrics_path)
    )

    # ------------------------- Training arguments -------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing = True,
        num_train_epochs=3,
        max_steps=1 if DEBUG else -1,
        weight_decay=1e-4,
        logging_strategy="steps",
        logging_steps=1 if DEBUG else 25,
        report_to=["tensorboard"],
        logging_dir=tensorboard_dir,
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=1 if DEBUG else 50,
        save_steps=1 if DEBUG else 50,
        save_total_limit=2,
        load_best_model_at_end= True,
        metric_for_best_model="eval_loss" if TASK == "clm" else args.selection_metric,
        greater_is_better=False if TASK == "clm" else True,
        label_names=["labels"],
        max_grad_norm=1.0,
        bf16=USE_BF16,
        fp16=USE_FP16,
        log_level="info",
        log_level_replica="warning",
        remove_unused_columns=False,
        eval_accumulation_steps=16 if TASK == "clm" else None,
    )

    # ------------------------- Load model and quantize -------------------------
    optional_kwargs = {}
    bnb_4bit_quant_storage_dtype = DTYPE if DTYPE == torch.bfloat16 else torch.float32
    model_dtype = DTYPE if not args.quant else bnb_4bit_quant_storage_dtype

    if args.quant and LLAMA:
        print("🔢 Using 4-bit quantization...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=DTYPE,
            bnb_4bit_quant_storage=bnb_4bit_quant_storage_dtype,
        )
        optional_kwargs["quantization_config"] = quant_config

    if TASK == "seq_cls":
        optional_kwargs["id2label"] = {0: "NEGATIVE", 1: "POSITIVE"}
        optional_kwargs["label2id"] = {"NEGATIVE": 0, "POSITIVE": 1}
        optional_kwargs["num_labels"] = 2

    model = ModelClass.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=model_dtype,
        **optional_kwargs   
    )

    print(model)

    # ------------------------- Gradient Checkpointing -------------------------
    model.config.use_cache = not training_args.gradient_checkpointing
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    # ------------------------- Tokenizer -------------------------
    tokenizer_load_dir = resolve_tokenizer_dir(MODEL_PATH, args.continue_from_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_load_dir, 
        local_files_only=True, 
        trust_remote_code=True, 
        use_fast=True
    )
    
    config = model.config

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # ------------------------- Add special tokens-------------------------
    token_info = add_or_detect_special_tokens(
        tokenizer=tokenizer,
        model=model,
        task=TASK,
        use_lora=bool(args.lora),
    )

    # ------------------------- Load dataset -------------------------
    format_func = determine_format_fn(TASK)

    dataset = load_and_split_dataset(
        dataset_path=args.dataset_path,
        repo_path=REPO_PATH,
        slurm_tmpdir=slurm_tmpdir,
        debug=DEBUG,
        format_fn=format_func
    )

    if TASK == "seq_cls":
        dataset, class_weights, focal_loss_dict, original_class_distribution, class_distribution = apply_class_imbalance_strategy(
            dataset=dataset,
            strategy=args.class_imbalance_fix,
            seed=42,
            alpha=FL_ALPHA,
            gamma=FL_GAMMA
        )

        # Prepare loss function if needed
        focal_loss_fct = None
        if args.class_imbalance_fix == "focal_loss":
            focal_loss_fct = FocalLoss(**focal_loss_dict)

    # ------------------------- tokenize -------------------------
    should_truncate, tokenizer_max_len = determine_tokenizer_truncation(
        tokenizer=tokenizer,
        config=config,
        truncation_len=args.truncation_len,
        chunking_len=args.chunking_len if TASK == "clm" else None
    )

    def tokenize_data(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=should_truncate,
            max_length=tokenizer_max_len
        )
        return outputs

    tokenized_dataset = dataset.map(tokenize_data, batched=True, remove_columns=["text"])
    final_dataset = tokenized_dataset

    print("Tokenized dataset features:")
    print(tokenized_dataset['train'].features)

    # ------------------------------ Chunk commits ------------------------------
    if TASK == "clm" and args.chunking_len:
        chunked_dataset = chunk_long_samples(
            tokenized_dataset,
            max_seq_length=args.chunking_len,
            overlap_pct=0
        )
        final_dataset = chunked_dataset

        print("✅ Chunked dataset features:")
        print(chunked_dataset['train'].features)

    # ------------------------------ Data Collator ------------------------------
    data_collator = determine_data_collator(TASK, tokenizer)

    # ------------------------- LORA -------------------------
    if args.lora:
        print("✨ Applying LoRA...")

        modules_to_save = None
        if TASK == "clm" and token_info.get("modules_to_save_update"):
            modules_to_save = ['lm_head']

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules="all-linear" if LLAMA else ["query", "value"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_CLS if TASK == "seq_cls" else TaskType.CAUSAL_LM,
            modules_to_save=modules_to_save
        )

        # Make only the new embedding rows trainable (if PEFT supports it)
        if token_info.get("added_token_ids"):
            if hasattr(lora_config, "trainable_token_indices"):
                lora_config.trainable_token_indices = {"embed_tokens": token_info["added_token_ids"]}
                print(f"🔓 PEFT will train only new embed rows: {lora_config.trainable_token_indices['embed_tokens']}")
            else:
                print("⚠️ Your PEFT version may not support `trainable_token_indices`. "
                    "Consider upgrading PEFT. Falling back to adapter-only (new token rows may learn slowly).")
                lora_config.modules_to_save.append("embed_tokens")

        model = prepare_peft_model(model, lora_config, training_args)

        # Summarize trainable parameters
        _ = count_trainable_params(
            model=model,
            tokenizer=tokenizer,
            task=TASK,
            added_token_ids=token_info.get("added_token_ids", None),
            verbose=True,
        )

    # ------------------------- Save Config to File -------------------------
    save_training_config(
        config_path=config_path,
        run_timestamp=run_timestamp,
        args=args,
        training_args=training_args,
        class_distribution=class_distribution if TASK == "seq_cls" else None,
        original_class_distribution=original_class_distribution if TASK == "seq_cls" else None,
        truncation_len=tokenizer_max_len,
        chunking_len=args.chunking_len if TASK == "clm" else None,
        dtype=DTYPE,
        DEBUG=DEBUG,
        task=TASK,
        dataset=dataset,
        RECALL_AT_TOP_K_PERCENTAGES=RECALL_AT_TOP_K_PERCENTAGES if TASK == "seq_cls" else None,
        FL_GAMMA=FL_GAMMA if TASK == "seq_cls" else None,
        FL_ALPHA=FL_ALPHA if TASK == "seq_cls" else None,
        model_config=model.config.to_dict()
    )

    # ------------------------- Trainer -------------------------
    trainer_optional_kwargs = {}
    if TASK == "seq_cls":
        trainer_optional_kwargs["compute_metrics"] = custom_metrics_seq_cls
        # trainer_optional_kwargs["class_weights"] = class_weights if args.class_imbalance_fix == "weighted_loss" else None
        # trainer_optional_kwargs["focal_loss_fct"] = focal_loss_fct if args.class_imbalance_fix == "focal_loss" else None


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=final_dataset["train"],
        eval_dataset=final_dataset["test"],
        data_collator=data_collator,
        callbacks=trainer_callbacks,
        **trainer_optional_kwargs
    )

    trainer.accelerator.print(f"{trainer.model}")

    # ---------------------------- Train ----------------------------
    trainer.train(resume_from_checkpoint= True if args.continue_from_dir else False)

    # ---------------------------- Save ----------------------------
    # save_model_safely(trainer, finetuned_model_dir)

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(finetuned_model_dir)
    print(f"💾 Saved model to: {finetuned_model_dir}")

    # ---------------------------- Save Metrics ----------------------------
    save_training_metrics(trainer, metrics_dir, filename="metrics.json")

    # ---------------------------- Run inference on held-out test set ----------------------------
    # if TASK == "seq_cls":
    #     run_final_inference(
    #         trainer=trainer,
    #         test_dataset=final_dataset["final_test"],
    #         metrics_dir=metrics_dir,
    #         percentages=RECALL_AT_TOP_K_PERCENTAGES,
    #         threshold=args.threshold,
    #     )


if __name__ == "__main__":
    main()