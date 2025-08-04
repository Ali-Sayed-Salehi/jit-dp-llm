import os
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv
import yaml
from accelerate import Accelerator
from peft import PeftModel

from huggingface_hub import login as huggingface_hub_login
from accelerate.utils import DistributedType

from transformers import TrainerCallback

from datasets import load_dataset, DatasetDict, concatenate_datasets
from subprocess import run, CalledProcessError
import numpy as np


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


def parse_training_args():
    # First, parse only the config file argument
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None, help="Path to YAML config file with default arguments")
    known_args, _ = pre_parser.parse_known_args()

    # Load config file values
    defaults = {}
    if known_args.config:
        if not os.path.isfile(known_args.config):
            raise FileNotFoundError(f"Config file not found: {known_args.config}")
        with open(known_args.config, "r") as f:
            defaults = yaml.safe_load(f) or {}

    # Full parser with defaults from config
    parser = argparse.ArgumentParser(parents=[pre_parser])
    parser.set_defaults(**defaults)

    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--live_metrics", action="store_true", help="Enable saving evaluation metrics after each eval step")
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Choose which dataset to use by specifying its absolute path. Default is imdb for sequence classification and eli5 for causal lm."
    )
    parser.add_argument("--model_path", type=str, required=not defaults.get("model_path"), help="Full path to the local pretrained model directory")
    parser.add_argument(
        "--class_imbalance_fix",
        type=str,
        choices=["oversampling", "weighted_loss", "focal_loss", "none"],
        help="Class imbalance handling method"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Optional decision threshold for classifying as class 1 (between 0 and 1). If not set, uses argmax."
    )
    parser.add_argument("--quant", action="store_true", help="Enable quantization with BitsAndBytesConfig")
    parser.add_argument("--lora", action="store_true", help="Enable LoRA fine-tuning using PEFT")
    parser.add_argument("--bf16", action="store_true", help="Enable bfloat16 training")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument(
        "--continue_from_dir", 
        type=str, 
        help="""Resume training from this checkpoint directory. 
        Example: '--continue_from_dir /speed-scratch/a_s87063/repos/perf-pilot/llama/training/run_2025-06-10_20-42-03/output'"""
    )
    parser.add_argument(
        "--selection_metric",
        type=str,
        help="Metric to select the best model: recall@top_5%, recall@top_10%, recall@top_30%, f1, precision, recall, accuracy. Default is recall@top_5%"
    )
    parser.add_argument(
        "--truncation_len",
        type=int,
        help="Optional. The length to which the sequences should be truncated using the tokenizer to reduce the size of the dataset."
    )

    parser.add_argument(
        "--chunking_len",
        type=int,
        help="Optional. The length to which the sequences should be chunked in order to reduce sequence length for the model."
    )

    args = parser.parse_args()

    # --- Path normalization ---
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    for key in ["dataset_path", "model_path", "continue_from_dir"]:
        path_val = getattr(args, key, None)
        if path_val:
            if os.path.isabs(path_val):
                # Leave absolute paths untouched
                abs_path = path_val
            else:
                # Join relative paths to repo_root without resolving symlinks
                abs_path = os.path.normpath(os.path.join(repo_root, path_val))
                # Strip "/nfs" if it's just a symlinked prefix
                if abs_path.startswith("/nfs/"):
                    abs_path = abs_path.replace("/nfs", "", 1)
            setattr(args, key, abs_path)

    # --- Validation Section ---
    if args.threshold is not None and not (0.0 <= args.threshold <= 1.0):
        raise ValueError("Threshold must be between 0 and 1 if specified")

    VALID_SELECTION_METRICS = {
        "recall@top_5%", "recall@top_10%", "recall@top_30%",
        "f1", "precision", "recall", "accuracy", None
    }
    if args.selection_metric not in VALID_SELECTION_METRICS:
        raise ValueError(f"Unsupported selection_metric '{args.selection_metric}'. Must be one of {VALID_SELECTION_METRICS}")

    if args.truncation_len and args.chunking_len and args.truncation_len < args.chunking_len:
        raise ValueError(f"truncation_len ({args.truncation_len}) cannot be less than chunking_len ({args.chunking_len})")

    return args


def setup_training_directories(repo_root, slurm_tmpdir, continue_from_dir=None):
    """
    Sets up output, metrics, tensorboard, and model/tokenizer directories.

    Returns:
        dict with keys: output_dir, run_timestamp, metrics_dir, tensorboard_dir,
        config_path, live_metrics_path, model_dir, tokenizer_dir, all_dirs (list of paths created)
    """
    if continue_from_dir:
        output_dir = continue_from_dir
        run_timestamp = os.path.basename(os.path.dirname(output_dir)).split("_", 1)[-1]
        print(f"🔁 Resuming from checkpoint in: {output_dir}")
    else:
        run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join(repo_root, "llama", "training", f"run_{run_timestamp}", "output")

    offload_dir = os.path.join(os.environ[slurm_tmpdir], "offload") 
    slurm_tmpdir = os.environ[slurm_tmpdir]

    base_run_dir = os.path.dirname(output_dir)
    tensorboard_dir = os.path.join(base_run_dir, "tensorboard")
    metrics_dir = os.path.join(base_run_dir, "metrics")
    model_dir = os.path.join(base_run_dir, "model")
    tokenizer_dir = os.path.join(base_run_dir, "tokenizer")
    config_path = os.path.join(metrics_dir, "config.json")
    live_metrics_path = os.path.join(metrics_dir, "live_metrics.jsonl")

    dirs_to_create = [output_dir, tensorboard_dir, metrics_dir, model_dir, tokenizer_dir, offload_dir, slurm_tmpdir]
    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)

    return {
        "output_dir": output_dir,
        "run_timestamp": run_timestamp,
        "tensorboard_dir": tensorboard_dir,
        "metrics_dir": metrics_dir,
        "config_path": config_path,
        "live_metrics_path": live_metrics_path,
        "model_dir": model_dir,
        "tokenizer_dir": tokenizer_dir,
        "all_dirs": dirs_to_create,
        "offload_dir": offload_dir,
        "slurm_tmpdir": slurm_tmpdir
    }


class SaveMetricsCallback(TrainerCallback):
    """
    A custom Hugging Face `TrainerCallback` for saving training and evaluation metrics to disk
    after each logging or evaluation step. Metrics are appended to a JSON Lines (JSONL) file,
    enabling later visualization or analysis (e.g., plotting loss/accuracy curves).

    Args:
        output_path (str): Path to the output `.jsonl` file where metrics will be saved.
                           The directory will be created if it does not exist.

    Example JSONL structure:
        {
            "step": 100,
            "type": "train",
            "metrics": {
                "loss": 0.543,
                "learning_rate": 5e-5
            }
        }

        {
            "step": 200,
            "type": "eval",
            "metrics": {
                "eval_accuracy": 0.88,
                "eval_loss": 0.42
            }
        }

    Methods:
        on_log(): Called after each log event (e.g., loss during training).
        on_evaluate(): Called after each evaluation step (e.g., validation metrics).
    """

    def __init__(self, output_path):
        self.output_path = output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def _write_metrics(self, state, metrics, metric_type):
        """
        Appends a single metric record to the JSONL file.

        Args:
            state (TrainerState): The current training state object.
            metrics (dict): Dictionary of metric values.
            metric_type (str): One of "train" or "eval" to identify the source.
        """
        if metrics is not None:
            with open(self.output_path, "a") as f:
                json.dump({
                    "step": state.global_step,
                    "type": metric_type,
                    "metrics": metrics
                }, f)
                f.write("\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Called after logging training metrics. Saves training metrics to file if present.

        Args:
            args (TrainingArguments): The training arguments.
            state (TrainerState): Current state of training.
            control (TrainerControl): Control flow handler.
            logs (dict, optional): Dictionary of logged training metrics.
        """
        if logs and not any(k.startswith("eval_") for k in logs):
            self._write_metrics(state, logs, "train")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        Called after evaluation. Saves evaluation metrics to file.

        Args:
            args (TrainingArguments): The training arguments.
            state (TrainerState): Current state of training.
            control (TrainerControl): Control flow handler.
            metrics (dict, optional): Evaluation metrics dictionary.
        """
        self._write_metrics(state, metrics, "eval")


def setup_live_metrics(live_metrics_enabled: bool, live_metrics_path: str):
    """
    Sets up Trainer callbacks for live metrics logging.

    Args:
        live_metrics_enabled (bool): Whether to enable live metrics logging.
        live_metrics_path (str): Path to save live metrics JSONL file if enabled.

    Returns:
        list: A list of TrainerCallback instances.
    """
    callbacks = []

    if live_metrics_enabled:
        callbacks.append(SaveMetricsCallback(live_metrics_path))
        print(f"📊 Live metrics will be saved to: {live_metrics_path}")
    else:
        print("📊 Live metrics logging disabled.")

    return callbacks


def load_and_split_dataset(
    dataset_path,
    repo_path,
    slurm_tmpdir,
    debug=False,
    seed=42,
    format_fn=None
):
    """
    Loads and splits a local JSONL dataset or the IMDb dataset (top 7000 longest reviews)
    for causal LM or classification, depending on the provided format_fn.
    Prints length statistics for the selected dataset.
    """

    if dataset_path is None or str(dataset_path).strip().lower() == "imdb":
        print("🎬 Loading IMDb dataset from Hugging Face...")

        dataset = load_dataset("imdb")
        full_dataset = concatenate_datasets([dataset["train"], dataset["test"]])


        # Add length field using raw text length
        full_dataset = full_dataset.map(lambda ex: {"length": len(ex["text"])})

        # Sort and take top 7000
        full_dataset = full_dataset.sort("length", reverse=True)
        top_dataset = full_dataset.select(range(min(7000, len(full_dataset))))
    else:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"❌ Dataset file does not exist: {dataset_path}")

        dataset_relpath = os.path.relpath(dataset_path, start=repo_path)
        dest_dir = os.path.join(slurm_tmpdir, os.path.dirname(dataset_relpath))
        dest_file = os.path.join(dest_dir, os.path.basename(dataset_path))
        dataset_copy_path = dataset_path

        try:
            os.makedirs(dest_dir, exist_ok=True)
            run(["rsync", "-a", "--delete", dataset_path, dest_file], check=True)
            print(f"✅ Copied {dataset_path} → {dest_file}")
            dataset_copy_path = dest_file
        except (OSError, CalledProcessError) as e:
            print(f"⚠️ Could not copy dataset to SLURM tmpdir: {e}")
            print(f"🔄 Falling back to using original dataset path: {dataset_path}")

        top_dataset = load_dataset("json", data_files=dataset_copy_path, split="train")
        # Compute length for JSON dataset as well
        top_dataset = top_dataset.map(lambda ex: {"length": len(ex.get("text", ex.get("prompt", "")))})

    # 🔍 Print length distribution
    lengths = [ex["length"] for ex in top_dataset]
    print(f"📊 Dataset length stats (chars):")
    print(f"   Min: {np.min(lengths)}")
    print(f"   Max: {np.max(lengths)}")
    print(f"   Avg: {np.mean(lengths):.2f}")
    print(f"   Median: {np.median(lengths)}")
    print(f"   90th percentile: {np.percentile(lengths, 90)}")

    # Split 64% train, 16% eval, 20% test
    n_total = len(top_dataset)
    n_train = int(n_total * 0.64)
    n_eval  = int(n_total * 0.16)

    train_dataset = top_dataset.select(range(0, n_train))
    eval_dataset  = top_dataset.select(range(n_train, n_train + n_eval))
    test_dataset  = top_dataset.select(range(n_train + n_eval, n_total))

    if debug:
        train_dataset = train_dataset.select(range(min(200, len(train_dataset))))
        eval_dataset  = eval_dataset.select(range(min(100, len(eval_dataset))))
        test_dataset  = test_dataset.select(range(min(100, len(test_dataset))))

    train_dataset = train_dataset.shuffle(seed=seed)

    # Apply formatter if provided
    if format_fn is not None:
        train_dataset = train_dataset.map(format_fn, remove_columns=train_dataset.column_names)
        eval_dataset  = eval_dataset.map(format_fn, remove_columns=eval_dataset.column_names)
        test_dataset  = test_dataset.map(format_fn, remove_columns=test_dataset.column_names)

    return DatasetDict({
        "train": train_dataset,
        "test": eval_dataset,
        "final_test": test_dataset
    })
