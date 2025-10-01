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
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM
)
from peft import (
    LoraConfig,
    TaskType
)

from utils import *
from checks import *
from custom_llama import *


def main():
    # ---------------------------- Parse Arguments and constants ----------------------------
    args = parse_training_args()

    DEBUG = args.debug

    trainer_callbacks = []
    SLURM_TMPDIR = args.slurm_tmpdir_env
    set_seed(42)

    DTYPE, USE_FP16, USE_BF16 = set_dtype(args.mixed_precision)
    TASK = args.task_type
    print(f"▶️ Finetuning task type: {TASK}")

    FL_ALPHA = 2
    FL_GAMMA = 5
    RECALL_AT_TOP_K_PERCENTAGES = [0.05, 0.1, 0.3]

    TASK_TO_MODEL_CLASS = {
        "clm": AutoModelForCausalLM,
        "mlm": AutoModelForMaskedLM,
        "seq_cls": AutoModelForSequenceClassification,
    }

    TASK_TO_TRAINER_CLASS = {
        "clm": Trainer,
        "seq_cls": CustomTrainer,
    }

    ModelClass = TASK_TO_MODEL_CLASS[TASK]
    TrainerClass = TASK_TO_TRAINER_CLASS[TASK]

    if args.pooling:
        ModelClass = LlamaForSequenceClassificationMaxPoolMeanLast

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

    # ------------------------- Detect model -------------------------
    cfg = AutoConfig.from_pretrained(MODEL_PATH)
    LLAMA = ("llama" in cfg.model_type.lower())
    BIGCODE = any(x in cfg.model_type.lower() for x in ["gpt_bigcode", "starcoder", "starcoder2"])
    BERT = any(x in cfg.model_type.lower() for x in ["roberta", "bert"])

    # ------------------------- Register custom llama -------------------------
    if TASK == "seq_cls" and LLAMA:
        register_custom_llama4_if_needed(MODEL_PATH)

    # ------------------------- Training arguments -------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing = True,
        num_train_epochs=args.num_train_epochs,
        max_steps=1 if DEBUG else -1,
        weight_decay=args.weight_decay,
        logging_strategy="steps",
        logging_steps=1 if DEBUG else args.logging_steps,
        report_to=["tensorboard"],
        logging_dir=tensorboard_dir,
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=1 if DEBUG else args.eval_steps,
        save_steps=1 if DEBUG else args.save_steps,
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
        # lr_scheduler_type="cosine",
        warmup_ratio=args.lr_warmup_ratio,
        label_smoothing_factor=0.0 if TASK == "clm" else 0.05,
        # torch_compile=True,
        # lr_scheduler_type="reduce_lr_on_plateau"
    )

    # ------------------------- Load model and quantize -------------------------
    optional_kwargs = {}
    bnb_4bit_quant_storage_dtype = DTYPE if DTYPE == torch.bfloat16 else torch.float32
    model_dtype = DTYPE if not args.quant else bnb_4bit_quant_storage_dtype

    if args.quant:
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
        optional_kwargs["problem_type"] = "single_label_classification"
        # optional_kwargs["architectures"] = ["LlamaForSequenceClassification"]

    if args.flash_attn_2 and not BERT:
        optional_kwargs["attn_implementation"] = "flash_attention_2"

    model = ModelClass.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=model_dtype,
        **optional_kwargs   
    )

    if args.pooling:
        model.config.pooling = args.pooling
        print (f"✅ Using {args.pooling}-pooling for activation pooling.")

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

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id


    # ------------------------- Add special tokens-------------------------
    token_info = add_or_detect_special_tokens(
        tokenizer=tokenizer,
        model=model,
        task=TASK,
        new_tokens=args.new_tokens,
        use_lora=bool(args.lora),
    )

    # ------------------------- Load dataset -------------------------
    format_func = determine_format_fn(TASK, args.clm_for_seq_cls)

    dataset = load_and_split_dataset(
        dataset_path=args.dataset_path,
        repo_path=REPO_PATH,
        slurm_tmpdir=slurm_tmpdir,
        debug=DEBUG,
        format_fn=format_func
    )

    if args.class_imbalance_fix:
        dataset, class_weights, focal_loss_dict, original_class_distribution, class_distribution = apply_class_imbalance_strategy(
            dataset=dataset,
            strategy=args.class_imbalance_fix,
            seed=42,
            alpha=FL_ALPHA,
            gamma=FL_GAMMA,
            sampling_strategy=args.resampling_ratio
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

    # ------------------------------ clm for seq cls -----------------------------
    if TASK == "clm" and args.clm_for_seq_cls:
        final_dataset = final_dataset.map(append_drs_and_label_to_tokens, fn_kwargs=dict(tokenizer=tokenizer))

        report = check_drs_append(
            final_dataset,
            tokenizer,
            label_key="labels",            # where 0/1 is stored
            drs_token="[/drs]",
            zero_token="0",
            one_token="1",
            strict_single_token=True,     # ensure " 0"/" 1" are single pieces
            max_errors=10,                # show up to 10 bad examples
            tail_tokens_to_show=8,
            check_pad_consistency=True,   # ensure attention_mask==0 → pad_token_id
        )

        # Quick glance:
        print({k: report[k] for k in ["num_examples","num_ok","num_errors","error_rate","label_counts"]})
        if report["num_errors"]:
            from pprint import pprint
            pprint(report["errors"][0])   # inspect the first problem case


    # ------------------------------ Data Collator ------------------------------
    data_collator = determine_data_collator(TASK, tokenizer, args.clm_for_seq_cls)

    # ------------------------- LORA -------------------------
    if args.lora:
        print("✨ Applying LoRA...")

        modules_to_save = None
        if TASK == "clm" and token_info.get("modules_to_save_update"):
            modules_to_save = ['lm_head']

        target_modules = infer_lora_target_modules(model)
        print(f"🔧 LoRA target_modules = {target_modules}")

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
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
        class_distribution=class_distribution if TASK == "seq_cls" and args.class_imbalance_fix else None,
        original_class_distribution=original_class_distribution if TASK == "seq_cls" and args.class_imbalance_fix else None,
        truncation_len=tokenizer_max_len,
        chunking_len=args.chunking_len if TASK == "clm" else None,
        dtype=DTYPE,
        task=TASK,
        dataset=dataset,
        RECALL_AT_TOP_K_PERCENTAGES=RECALL_AT_TOP_K_PERCENTAGES if TASK == "seq_cls" else None,
        FL_GAMMA=FL_GAMMA if TASK == "seq_cls" else None,
        FL_ALPHA=FL_ALPHA if TASK == "seq_cls" else None,
        model_config=model.config.to_dict()
    )

    # ------------------------- define metrics -------------------------
    def custom_metrics_seq_cls(eval_pred):
        return compute_custom_metrics_seq_cls(eval_pred, REPO_PATH, threshold=args.threshold, percentages=RECALL_AT_TOP_K_PERCENTAGES)

    if TASK == "clm" and args.clm_for_seq_cls:
        clm_for_seq_cls_compute_metrics = make_compute_metrics_for_clm_seqcls_autoids(
            tokenizer=tokenizer,
            repo_root=REPO_ROOT,
            recall_at_top_k_fn=recall_at_top_k,   # your function
            percentages=[0.05, 0.1, 0.2],
            threshold=0.5,
            average="binary",
            zero_token=" 0",
            one_token=" 1",
        )

    trainer_callbacks.extend(
        setup_live_metrics(args.live_metrics, live_metrics_path)
    )

    # ------------------------- Trainer -------------------------
    trainer_optional_kwargs = {}
    if TASK == "seq_cls":
        trainer_optional_kwargs["compute_metrics"] = custom_metrics_seq_cls
        # if imbalance_strategy == "weighted_loss" and class_weights is not None:
        #     trainer_optional_kwargs["class_weights"] = class_weights
        # if imbalance_strategy == "focal_loss" and focal_loss_fct is not None:
        #     trainer_optional_kwargs["focal_loss_fct"] = focal_loss_fct

    if TASK == "clm" and args.clm_for_seq_cls:
        trainer_optional_kwargs["compute_metrics"] = clm_for_seq_cls_compute_metrics

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
    if TASK == "seq_cls":
        run_final_inference(
            trainer=trainer,
            test_dataset=final_dataset["final_test"],
            metrics_dir=metrics_dir,
            percentages=RECALL_AT_TOP_K_PERCENTAGES,
            repo_root=REPO_PATH,
            threshold=args.threshold,
        )


if __name__ == "__main__":
    main()