import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import builtins

import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    set_seed
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)

from causal_lm_utils import (
    compute_custom_metrics,
    run_final_inference,
    save_training_metrics,
    save_training_config,
    chunk_long_samples
)

from utils import (
    determine_tokenizer_truncation,
    login_to_huggingface,
    evaluate_and_save_best_model,
    handle_gradient_checkpointing,
    parse_training_args,
    setup_training_directories,
    setup_live_metrics,
    load_and_split_dataset,
    get_mixed_precision_dtype
)

from trl import SFTTrainer, SFTConfig

from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live

def main():
    # ---------------------------- Parse Arguments and constants ----------------------------
    args = parse_training_args()

    DEBUG = args.debug
    LLAMA = "llama" in args.model_path.lower()
    LONG_LLAMA = "long_llama" in args.model_path.lower()
    # what percentile of sequence lengths from the data we use as cut-off limit for tokenizer
    SEQ_LEN_PERCENTILE = 100
    trainer_callbacks = []
    SLURM_TMPDIR = "TMPDIR"
    set_seed(42)

    # DTYPE, USE_BF16, USE_FP16 = get_mixed_precision_dtype(args.mixed_precision)
    DTYPE = torch.bfloat16
    USE_BF16 = True
    USE_FP16 = False

    # ---------------------------- distributed setup  ----------------------------
    local_rank = os.environ.get("LOCAL_RANK", 0)
    world_size = os.environ.get("WORLD_SIZE", 1)
    print(f"🚀 Local rank: {local_rank} | World size: {world_size}")

    # local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # torch.cuda.set_device(local_rank)

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

    # ------------------------- Load dataset -------------------------
    def format_for_lm(example):
        if "text" in example: # imdb dataset
            return {"text": example["text"]}
        if "prompt" in example: # jit dataset
            return {"text": example["prompt"]}
        return {"text": ""}

    dataset = load_and_split_dataset(
        dataset_path=args.dataset_path,
        repo_path=REPO_PATH,
        slurm_tmpdir=slurm_tmpdir,
        debug=DEBUG,
        format_fn=format_for_lm
    )

    # ------------------------- tokenize -------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)
    # config = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    # should_truncate, tokenizer_max_len = determine_tokenizer_truncation(
    #     tokenizer=tokenizer,
    #     config=config,
    #     truncation_len=args.truncation_len,
    #     chunking_len=args.chunking_len
    # )

    packing_enabled = bool(args.chunking_len)
    max_seq_length = args.chunking_len if packing_enabled else tokenizer_max_len

    # def tokenize_data(examples):
    #     outputs = tokenizer(
    #         examples["text"],
    #         truncation=should_truncate,
    #         max_length=tokenizer_max_len
    #     )
    #     # outputs["labels"] = outputs["input_ids"].copy()
    #     return outputs

    # tokenized_dataset = dataset.map(tokenize_data, batched=True, remove_columns=["text"])
    # final_dataset = tokenized_dataset

    # print("Tokenized dataset features:")
    # print(tokenized_dataset['train'].features)

    # # ------------------------------ Chunk commits ------------------------------
    # if args.chunking_len:
    #     chunked_dataset = chunk_long_samples(
    #         tokenized_dataset,
    #         max_seq_length=args.chunking_len,
    #         overlap_pct=0
    #     )
    #     final_dataset = chunked_dataset

    #     print("✅ Chunked dataset features:")
    #     print(chunked_dataset['train'].features)

    # ------------------------------ final dataset ------------------------------
    # final_dataset.set_format("torch")

    # ------------------------------ Data Collator ------------------------------
    # data_collator = DataCollatorForLanguageModeling(
    #     tokenizer=tokenizer,
    #     mlm=False,
    # )

    # ------------------------- define metrics -------------------------
    def custom_metrics(eval_pred):
        return compute_custom_metrics(eval_pred)

    trainer_callbacks.extend(
        setup_live_metrics(args.live_metrics, live_metrics_path)
    )

    # ------------------------- Training arguments -------------------------
    training_args = SFTConfig(
        output_dir=output_dir,
        learning_rate=1e-4,
        per_device_train_batch_size=1 if DEBUG else 1,
        per_device_eval_batch_size=1 if DEBUG else 1,
        gradient_accumulation_steps=16,
        num_train_epochs=1 if DEBUG else 3,
        max_steps=1 if DEBUG else -1,
        weight_decay=0.01,
        logging_strategy="steps",
        logging_steps=1 if DEBUG else 25,
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=1 if DEBUG else 25,
        save_steps=1 if DEBUG else 25,
        save_total_limit=2,
        load_best_model_at_end= True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        label_names=["labels"],
        max_grad_norm=1.0,
        bf16=USE_BF16,
        fp16=USE_FP16,
        log_level="info",
        log_level_replica="warning",
        remove_unused_columns=False,
        eval_accumulation_steps=16,
        dataset_text_field="text",
        max_length=max_seq_length,
        packing=packing_enabled,
        dataset_kwargs={"append_concat_token": True, "add_special_tokens": False}
    )

    # ------------------------- Load model and quantization-------------------------
    optional_kwargs = {}

    if args.quant and LLAMA:
        print("🔢 Using 4-bit quantization...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=DTYPE,
            bnb_4bit_quant_storage=DTYPE,
            # llm_int8_enable_fp32_cpu_offload=True
        )
        optional_kwargs["quantization_config"] = quant_config
    else:
        print("🔢 Loading model without quantization...")

    if LONG_LLAMA:
        optional_kwargs["mem_attention_grouping"] = (1, 2048)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        # local_files_only=True,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=DTYPE,
        quantization_config = quant_config    
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    if LLAMA:
        model.config.pretraining_tp = 1

    # ------------------------- Gradient Checkpointing -------------------------
    handle_gradient_checkpointing(args, model, training_args)

    # ------------------------- LORA -------------------------
    if args.lora:
        print("✨ Applying LoRA...")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            # target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'] if LLAMA else ["query", "value"],
            target_modules="all-linear" if LLAMA else ["query", "value"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        # model = prepare_model_for_kbit_training(model)
        # model = get_peft_model(model, lora_config)

        # model.print_trainable_parameters()

    # sanity: ensure LoRA actually attached


    # ------------------------- Save Config to File -------------------------
    # save_training_config(
    #     config_path=config_path,
    #     run_timestamp=run_timestamp,
    #     args=args,
    #     training_args=training_args,
    #     truncation_len=tokenizer_max_len,
    #     chunking_len=args.chunking_len,
    #     dtype=DTYPE,
    #     DEBUG=DEBUG,
    #     FSDP = False,
    #     model_config=model.config.to_dict()
    # )

    # ------------------------- Trainer -------------------------
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=lora_config,
        # compute_metrics=custom_metrics,
        callbacks=trainer_callbacks,
    )

    print("🛑🛑🛑🛑")
    print(estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=4, num_nodes=1))

    trainer.accelerator.print(f"{trainer.model}")

    # ---------------------------- Train ----------------------------
    trainer.train(resume_from_checkpoint= True if args.continue_from_dir else False)

    # ---------------------------- Evaluate Best Model and Save ----------------------------
    evaluate_and_save_best_model(
        trainer=trainer,
        training_args=training_args,
        metrics_dir=metrics_dir,
        save_dir=finetuned_model_dir,
        tokenizer_dir=finetuned_tokenizer_dir,
        tokenizer=tokenizer
    )

    # ---------------------------- Save Metrics ----------------------------
    save_training_metrics(trainer, metrics_dir, filename="metrics.json")

    # ---------------------------- Run inference on held-out test set ----------------------------
    # run_final_inference(
    #     trainer=trainer,
    #     test_dataset=dataset["final_test"],
    #     metrics_dir=metrics_dir,
    # )

    pass


if __name__ == "__main__":
    main()