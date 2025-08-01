import os
import json
from dotenv import load_dotenv
from accelerate import Accelerator
from peft import PeftModel

from huggingface_hub import login as huggingface_hub_login
from accelerate.utils import DistributedType


def determine_tokenizer_truncation(
    tokenizer,
    config,
    truncation_len=None,
    chunking_len=None
):

    """
    Determines whether tokenizer truncation should be enabled, and what max_length
    should be passed to the tokenizer.

    This function considers the user-specified `truncation_len` and `chunking_len`, and
    compares them with the model's maximum supported input length.

    Args:
        tokenizer (PreTrainedTokenizer): 
            Hugging Face tokenizer object, used to fetch `model_max_length`.
        config (PretrainedConfig): 
            Model config with `max_position_embeddings`.
        truncation_len (int, optional): 
            If provided, this is the length to truncate each input sequence to.
        chunking_len (int, optional): 
            If provided, this is the length to chunk long inputs into fixed-size parts.

    Returns:
        tuple:
            - should_truncate (bool): Whether truncation should be applied in tokenizer().
            - tokenizer_max_len (int or None): Value to use for `max_length`. Can be None if no truncation needed.
    """

    should_truncate = False
    tokenizer_max_len = None

    model_max_len = min(tokenizer.model_max_length, config.max_position_embeddings)

    print(f"✂️ User_specified truncation len: {truncation_len}, chunking len: ({chunking_len}), model max len: {model_max_len}")

    if not truncation_len and not chunking_len:
        should_truncate = True
        tokenizer_max_len = model_max_len
        print(f"No truncation or chunking specified. Tokenizer will truncate to model_max_len ({tokenizer_max_len})")

    elif truncation_len and not chunking_len:
        should_truncate = True
        tokenizer_max_len = min(model_max_len, truncation_len)
        print(f"Tokenizer will truncate to min(model_max_len, truncation_len) ({tokenizer_max_len}). No chunking used")

    elif truncation_len and chunking_len:
        should_truncate = True
        user_specified_max_len = min(truncation_len, chunking_len)

        if user_specified_max_len > model_max_len:
            tokenizer_max_len = model_max_len
            print(f"user_specified_max_len ({user_specified_max_len}) > model_max_len ({chunking_len}). Tokenizer will truncate to model_max_len ({tokenizer_max_len})")
        else:
            tokenizer_max_len = truncation_len
            print(f"Tokenizer will truncate to truncation_len ({tokenizer_max_len})")

    elif not truncation_len and chunking_len:
        if chunking_len > model_max_len:
            should_truncate = True
            tokenizer_max_len = model_max_len
            print(f"chunking_len ({chunking_len}) > model_max_len ({model_max_len}). Tokenizer will truncate to model_max_len ({tokenizer_max_len})")
        else:
            should_truncate = False
            tokenizer_max_len = None
            print(f"chunking_len ({chunking_len}) used. No truncation.")
    
    return should_truncate, tokenizer_max_len


def login_to_huggingface(repo_path: str, env_path: str = "secrets/.env"):
    """
    Loads environment variables and logs into Hugging Face using the token in .env file.
    
    Args:
        repo_path (str): Base path to your repository.
        env_path (str): Relative path to the .env file from the repo path.
    """
    accelerator = Accelerator()

    dotenv_file = os.path.join(repo_path, env_path)
    load_dotenv(dotenv_path=dotenv_file)
    
    token = os.getenv("HUGGING_FACE_TOKEN")
    if not token:
        raise ValueError("🚫 HUGGING_FACE_TOKEN not found in environment variables")
    
    if accelerator.is_main_process:
        huggingface_hub_login(token)
        print("✅ Logged in to Hugging Face.")



def evaluate_and_save_best_model(
    trainer,
    training_args,
    metrics_dir,
    save_dir,
    tokenizer_dir,
    tokenizer=None
):
    """
    Evaluates the best model and saves:
    - LoRA adapter if using PEFT
    - Full model if not using LoRA
    - Handles FSDP automatically
    - Saves tokenizer
    """

    # ---------------- Evaluate ----------------
    if training_args.load_best_model_at_end:
        best_eval_metrics = trainer.evaluate()
        best_model_metrics_path = os.path.join(metrics_dir, "best_model_metrics.json")
        with open(best_model_metrics_path, "w") as f:
            json.dump(best_eval_metrics, f, indent=4)
        print(f"✅ Saved best model eval metrics to {best_model_metrics_path}")
    else:
        print("ℹ️ Skipping best model evaluation because load_best_model_at_end=False.")
        best_eval_metrics = None

    # ---------------- Save ----------------
    is_main_process = trainer.args.process_index == 0
    model = trainer.model

    if is_main_process:
        if trainer.is_fsdp_enabled and not isinstance(model, PeftModel):
            print(f"💾 Saving full model using FSDP to: {save_dir}")
            trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
            trainer.save_model(save_dir)
        elif isinstance(model, PeftModel):
            print(f"💾 Saving LoRA adapter weights to: {save_dir}")
            model.save_pretrained(save_dir, save_adapter=True)
        else:
            print(f"💾 Saving full model (non-FSDP) to: {save_dir}")
            trainer.save_model(save_dir)

        # Save tokenizer
        if tokenizer is not None:
            print(f"💾 Saving tokenizer to: {tokenizer_dir}")
            tokenizer.save_pretrained(tokenizer_dir)
        else:
            print("⚠️ No tokenizer provided; skipping tokenizer save.")
    else:
        print("🧵 Not the main process; skipping save.")

    return best_eval_metrics


def handle_gradient_checkpointing(args, model, training_args, trainer=None):
    """
    Handles gradient checkpointing configuration for DeepSpeed, FSDP, or vanilla HF Trainer.
    """

    if args.gradient_checkpointing:
        if trainer and getattr(trainer.accelerator.state, "deepspeed_plugin", None):
            # ⚡ Using DeepSpeed
            print("⚡ DeepSpeed activation checkpointing enabled via config.")
            model.config.use_cache = False
            training_args.gradient_checkpointing = False  # avoid double checkpointing

        elif trainer and getattr(trainer.accelerator.state, "fsdp_plugin", None):
            # ⚡ Using FSDP
            print("⚡ FSDP with Hugging Face gradient checkpointing.")
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
            training_args.gradient_checkpointing = True
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

        else:
            # ⚡ Using Hugging Face Trainer (no DS/FSDP)
            print("⚡ Hugging Face gradient checkpointing enabled in script.")
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
            training_args.gradient_checkpointing = True
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    else:
        print("ℹ️ Gradient checkpointing disabled.")
        training_args.gradient_checkpointing = False
