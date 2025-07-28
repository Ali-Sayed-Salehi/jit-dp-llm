import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# from accelerate import Accelerator
import builtins

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    TrainingArguments,
    DataCollatorWithPadding,
    BitsAndBytesConfig
)

from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)

from sequence_classification_utils import (
    parse_training_args,
    setup_training_directories,
    load_and_split_dataset,
    compute_class_distribution,
    apply_class_imbalance_strategy,
    FocalLoss,
    CustomTrainer,
    compute_custom_metrics,
    run_final_inference,
    save_training_metrics,
    save_training_config,
    setup_live_metrics,
    register_custom_llama4_if_needed
    )

from utils import (
    determine_tokenizer_truncation,
    login_to_huggingface,
    evaluate_and_save_best_model
)

def main():
    # ---------------------------- Parse Arguments ----------------------------
    args = parse_training_args()

    DEBUG = args.debug
    LLAMA = "llama" in args.model_path.lower()
    LONG_LLAMA = "long_llama" in args.model_path.lower()
    FL_ALPHA = 2
    FL_GAMMA = 5
    # what percentile of sequence lengths from the data we use as cut-off limit for tokenizer
    SEQ_LEN_PERCENTILE = 100
    RECALL_AT_TOP_K_PERCENTAGES = [0.05, 0.1, 0.3]
    trainer_callbacks = []
    SLURM_TMPDIR = "TMPDIR"

    # policy = get_mixed_precision_policy()
    # print(f"Recommended mixed precision policy: {policy}")

    # ---------------------------- distributed setup  ----------------------------
    local_rank = os.environ.get("LOCAL_RANK", 0)
    world_size = os.environ.get("WORLD_SIZE", 1)
    print(f"🚀 Local rank: {local_rank} | World size: {world_size}")

    # accelerator = Accelerator()

    # if not accelerator.is_main_process:
    #     builtins.print = lambda *args, **kwargs: None

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
    login_to_huggingface(REPO_PATH)

    # ------------------------- Register custom llama -------------------------
    register_custom_llama4_if_needed(MODEL_PATH)

    # ------------------------- Load dataset and fix class imbalance -------------------------

    dataset = load_and_split_dataset(
        dataset_path=args.dataset_path,
        repo_path=REPO_PATH,
        slurm_tmpdir=slurm_tmpdir,
        debug=DEBUG
    )

    imbalance_fix = args.class_imbalance_fix

    # Compute class distribution before imbalance fix
    original_class_distribution = compute_class_distribution(dataset)

    dataset, class_weights, focal_loss_dict = apply_class_imbalance_strategy(
        dataset=dataset,
        strategy=imbalance_fix,
        seed=42,
        alpha=FL_ALPHA,
        gamma=FL_GAMMA
    )

    # Prepare loss function if needed
    focal_loss_fct = None
    if imbalance_fix == "focal_loss":
        focal_loss_fct = FocalLoss(**focal_loss_dict)

    # Compute class distribution after imbalance fix
    class_distribution = compute_class_distribution(dataset)

    # ------------------------- tokenize -------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)
    config = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)

    if LLAMA:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    should_truncate, tokenizer_max_len = determine_tokenizer_truncation(
        tokenizer=tokenizer,
        config=config,
        truncation_len=args.truncation_len
    )

    def tokenize_data(examples):
        return tokenizer(
            examples["text"],
            truncation=should_truncate,
            max_length=tokenizer_max_len
        )

    tokenized_dataset = dataset.map(tokenize_data, batched=True, remove_columns=["text"])
    final_dataset = tokenized_dataset

    print("Tokenized dataset features:")
    print(tokenized_dataset['train'].features)

    # ------------------------------ final dataset ------------------------------
    final_dataset.set_format("torch")

    # ------------------------------ Data Collator ------------------------------
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ------------------------- define metrics -------------------------
    def custom_metrics(eval_pred):
        return compute_custom_metrics(eval_pred, threshold=args.threshold, percentages=RECALL_AT_TOP_K_PERCENTAGES)

    trainer_callbacks.extend(
        setup_live_metrics(args.live_metrics, live_metrics_path)
    )

    # ------------------------- Custom device map -------------------------
    # max_memory = {
    #     0: "20GB",
    #     "cpu": "200GB",
    #     "disk": "200GB"
    # }

    # device_map = calculate_custom_device_map(
    #     model_path=MODEL_PATH,
    #     max_memory=max_memory
    # )

    # ------------------------- Load model and quantization-------------------------
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    optional_kwargs = {}

    if args.quant and LLAMA:
        print("🔢 Using 4-bit quantization...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
            # llm_int8_enable_fp32_cpu_offload=True
        )
        optional_kwargs["quantization_config"] = quant_config
    else:
        print("🔢 Loading model without quantization...")

    if LONG_LLAMA:
        optional_kwargs["mem_attention_grouping"] = (1, 2048)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        local_files_only=True,
        trust_remote_code=True,
        # device_map="auto",
        torch_dtype=torch.bfloat16,
        # offload_folder=offload_dir,
        # offload_state_dict=True,
        **optional_kwargs
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if LLAMA:
        model.config.pretraining_tp = 1

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # ------------------------- LORA -------------------------
    if args.lora:
        print("✨ Applying LoRA...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=8,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'] if LLAMA else ["query", "value"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_CLS,
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # ------------------------- Training arguments -------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=1 if DEBUG else 1,
        per_device_eval_batch_size=1 if DEBUG else 1,
        gradient_accumulation_steps=16,
        num_train_epochs=1 if DEBUG else 3,
        max_steps=2 if DEBUG else -1,
        weight_decay=0.01,
        logging_strategy="steps",
        logging_steps=1 if DEBUG else 25,
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=1 if DEBUG else 25,
        save_steps=1 if DEBUG else 25,
        save_total_limit=2,
        load_best_model_at_end= True,
        metric_for_best_model=args.selection_metric,
        greater_is_better=True,
        label_names=["labels"],
        max_grad_norm=1.0,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        log_level="info",
        log_level_replica="warning",
        # disable_tqdm=not accelerator.is_main_process,
        remove_unused_columns=False,
        optim="paged_adamw_8bit",
    )

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {
            "use_reentrant": False
        }

    # ------------------------- Save Config to File -------------------------
    save_training_config(
        config_path=config_path,
        run_timestamp=run_timestamp,
        args=args,
        training_args=training_args,
        class_distribution=class_distribution,
        original_class_distribution=original_class_distribution,
        truncation_len=tokenizer_max_len,
        chunking_len=None,
        DEBUG=DEBUG,
        dataset=dataset,
        RECALL_AT_TOP_K_PERCENTAGES=RECALL_AT_TOP_K_PERCENTAGES,
        FL_GAMMA=FL_GAMMA,
        FL_ALPHA=FL_ALPHA,
        model_config=model.config.to_dict()
    )

    # ------------------------- Trainer -------------------------
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=final_dataset["train"],
        eval_dataset=final_dataset["test"],
        data_collator=data_collator,
        compute_metrics=custom_metrics,
        callbacks=trainer_callbacks,
        class_weights=class_weights if args.class_imbalance_fix == "weighted_loss" else None,
        focal_loss_fct=focal_loss_fct if args.class_imbalance_fix == "focal_loss" else None
    )

    #handle PEFT+FSDP case
    if getattr(trainer.accelerator.state, "fsdp_plugin", None) and args.lora:
        from peft.utils.other import fsdp_auto_wrap_policy

        fsdp_plugin = trainer.accelerator.state.fsdp_plugin
        fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

    # torch.cuda.empty_cache()

    trainer.train(resume_from_checkpoint= True if args.continue_from_dir else False)

    # ---------------------------- Evaluate Best Model and Save ----------------------------
    evaluate_and_save_best_model(
        trainer=trainer,
        training_args=training_args,
        metrics_dir=metrics_dir,
        adapter_dir=finetuned_model_dir,
        tokenizer_dir=finetuned_tokenizer_dir,
        tokenizer=tokenizer
    )

    # ---------------------------- Save Metrics ----------------------------
    save_training_metrics(trainer, metrics_dir, filename="metrics.json")

    # ---------------------------- Run inference on held-out test set ----------------------------
    run_final_inference(
        trainer=trainer,
        test_dataset=final_dataset["final_test"],
        metrics_dir=metrics_dir,
        percentages=RECALL_AT_TOP_K_PERCENTAGES,
        threshold=args.threshold,
    )

    pass


if __name__ == "__main__":
    main()