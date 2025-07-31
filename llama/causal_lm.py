#!/usr/bin/env python3

import os
import builtins

import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)

from causal_lm_utils import (
    parse_training_args,
    setup_training_directories,
    load_and_split_dataset,
    compute_custom_metrics,
    run_final_inference,
    save_training_metrics,
    save_training_config,
    setup_live_metrics,
    chunk_long_samples
)

from utils import (
    determine_tokenizer_truncation,
    login_to_huggingface,
    evaluate_and_save_best_model
)

# from accelerate import Accelerator

import logging
logging.basicConfig(level=logging.INFO)


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

    # ---------------------------- distributed setup  ----------------------------
    local_rank = os.environ.get("LOCAL_RANK", 0)
    world_size = os.environ.get("WORLD_SIZE", 1)
    print(f"🚀 Local rank: {local_rank} | World size: {world_size}")

    # accelerator = Accelerator()

    # if not accelerator.is_main_process:
    #     builtins.print = lambda *args, **kwargs: None

    # if accelerator.state.fsdp_plugin:
    #     FSDP = True
    #     print("🧩 Running with FSDP enabled!")
    # else:
    #     FSDP = False
    #     print("⚠️ FSDP Not enabled!")

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

    # ------------------------- Load dataset -------------------------
    dataset = load_and_split_dataset(
        dataset_path=args.dataset_path,
        repo_path=REPO_PATH,
        slurm_tmpdir=slurm_tmpdir,
        debug=DEBUG
    )

    # ------------------------- tokenize -------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)
    config = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    should_truncate, tokenizer_max_len = determine_tokenizer_truncation(
        tokenizer=tokenizer,
        config=config,
        truncation_len=args.truncation_len,
        chunking_len=args.chunking_len
    )

    def tokenize_data(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=should_truncate,
            max_length=tokenizer_max_len
        )
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    tokenized_dataset = dataset.map(tokenize_data, batched=True, remove_columns=["text"])
    final_dataset = tokenized_dataset

    print("Tokenized dataset features:")
    print(tokenized_dataset['train'].features)

    # ------------------------------ Chunk commits ------------------------------
    if args.chunking_len:
        chunked_dataset = chunk_long_samples(
            tokenized_dataset,
            max_seq_length=args.chunking_len,
            overlap_pct=0
        )
        final_dataset = chunked_dataset

        print("✅ Chunked dataset features:")
        print(chunked_dataset['train'].features)

    # ------------------------------ final dataset ------------------------------
    final_dataset.set_format("torch")

    # ------------------------------ Data Collator ------------------------------
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # ------------------------- define metrics -------------------------
    def custom_metrics(eval_pred):
        return compute_custom_metrics(eval_pred)

    trainer_callbacks.extend(
        setup_live_metrics(args.live_metrics, live_metrics_path)
    )

    # ------------------------- Load model and quantization-------------------------
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

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
        # device_map={"": torch.cuda.current_device()}, #if args.quant else None,
        # device_map= int(os.environ.get("LOCAL_RANK", -1)) if torch.distributed.is_available() and torch.distributed.is_initialized() else "auto",
        # attn_implementation="flash_attention",
        torch_dtype=torch.bfloat16,
        **optional_kwargs
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if LLAMA:
        model.config.pretraining_tp = 1

    # if args.gradient_checkpointing:
    #     model.gradient_checkpointing_enable()

    # ------------------------- LORA -------------------------

    # print([name for name, _ in model.named_modules() if "attn" in name])

    if args.lora:
        print("✨ Applying LoRA...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=8,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'] if LLAMA else ['c_attn', 'c_proj'],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)

        # model = model.to(torch.bfloat16)

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
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        label_names=["labels"],
        max_grad_norm=1.0,
        bf16=args.bf16,
        # gradient_checkpointing=args.gradient_checkpointing,
        log_level="info",
        log_level_replica="warning",
        # disable_tqdm=not accelerator.is_main_process,
        remove_unused_columns=False,
        optim="paged_adamw_8bit",
        # lr_scheduler_type="cosine",
        # warmup_steps=500,
        # fsdp=["full_shard", "auto_wrap"]
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
        truncation_len=tokenizer_max_len,
        chunking_len=args.chunking_len,
        DEBUG=DEBUG,
        FSDP = False,
        model_config=model.config.to_dict()
    )

    # ------------------------- Trainer -------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=final_dataset["train"],
        eval_dataset=final_dataset["test"],
        data_collator=data_collator,
        compute_metrics=custom_metrics,
        callbacks=trainer_callbacks
    )

    #handle PEFT+FSDP case
    if getattr(trainer.accelerator.state, "fsdp_plugin", None) and args.lora:
        from peft.utils.other import fsdp_auto_wrap_policy

        fsdp_plugin = trainer.accelerator.state.fsdp_plugin
        fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

    # ---------------------------- Model Setup Checks ----------------------------
    # print("🛑 Verifications:")
    # print(trainer.accelerator.state)

    # # The actual wrapped model
    # wrapped_model = trainer.model_wrapped
    # print(f"🔍 Model type: {type(wrapped_model)}")

    # trainer.accelerator.print(f"{trainer.model}")

    # # print(next(model.parameters()).dtype)

    # # torch.cuda.empty_cache()

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
    run_final_inference(
        trainer=trainer,
        test_dataset=final_dataset["final_test"],
        metrics_dir=metrics_dir,
    )

    pass


if __name__ == "__main__":
    main()