import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
import argparse
import json
from tqdm import tqdm
import os
import numpy as np

def load_dataset_local(dataset_path, field=None):
    """Load dataset from local text or JSON/JSONL file."""
    data = []
    if dataset_path.endswith(".json") or dataset_path.endswith(".jsonl"):
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if isinstance(entry, dict):
                        if field and field in entry and isinstance(entry[field], str):
                            data.append(entry[field])
                        else:
                            text = next((v for v in entry.values() if isinstance(v, str)), None)
                            if text:
                                data.append(text)
                    elif isinstance(entry, str):
                        data.append(entry)
                except json.JSONDecodeError:
                    continue
    else:  # plain text
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = [line.strip() for line in f if line.strip()]
    return data

def load_configs(config_path):
    """Load list of finetuning configs from a JSONL file."""
    configs = []
    with open(config_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                if isinstance(entry, dict):
                    configs.append(entry)
            except json.JSONDecodeError:
                continue
    return configs

def estimate_max_seq_len(config):
    """Heuristic fallback to estimate max sequence length if not provided."""
    base_len = 2048
    mem_factor = config.get("gpu_memory", 24) / 24
    seq_len = base_len * mem_factor
    seq_len *= max(1, config.get("num_gpus", 1) ** 0.5)

    if config.get("qlora", False):
        seq_len *= 1.5
    if config.get("deepspeed", False):
        seq_len *= 2

    if "model_size" in config and config["model_size"].endswith("b"):
        size_in_billion = float(config["model_size"].replace("b", ""))
        seq_len /= max(1, size_in_billion / 7)

    return int(seq_len)

def annotate_configs(ax, configs, lengths_arr, colors, orientation="violin"):
    """Draw config lines with assigned colors, no inline labels."""
    for cfg, color in zip(configs, colors):
        if "max_seq_len" in cfg:
            max_len = cfg["max_seq_len"]
        else:
            max_len = estimate_max_seq_len(cfg)

        coverage = np.mean(lengths_arr <= max_len) * 100
        label = (
            f"{cfg.get('llama_model', 'Unknown')} | "
            f"{cfg.get('num_gpus', '?')}x{cfg.get('gpu_memory', '?')}GB | "
            f"QLoRA={cfg.get('qlora', False)} | "
            f"DeepSpeed={cfg.get('deepspeed', False)} | "
            f"MaxLen={max_len} | Covers {coverage:.1f}%"
        )

        if orientation == "violin":
            ax.axhline(max_len, linestyle="--", color=color, label=label)
        elif orientation == "hist":
            ax.axvline(max_len, linestyle="--", color=color, label=label)

def main(dataset_path, model_path, field, output_path, logy, config_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    data = load_dataset_local(dataset_path, field)

    print(f"Loaded {len(data)} samples.")
    lengths = []
    for text in tqdm(data, desc="Tokenizing"):
        tokens = tokenizer.encode(text, truncation=False, add_special_tokens=True)
        lengths.append(len(tokens))
    lengths_arr = np.array(lengths)

    configs = load_configs(config_path) if config_path else []

    # Generate consistent colors across both plots
    color_map = plt.cm.tab20.colors  # 20 distinct colors
    if len(configs) > len(color_map):
        # Repeat colors if more configs than colors
        repeats = int(np.ceil(len(configs) / len(color_map)))
        colors = (color_map * repeats)[:len(configs)]
    else:
        colors = color_map[:len(configs)]

    # --- Violin Plot ---
    # --- Violin Plot ---
    plt.figure(figsize=(15, 6))
    ax = sns.violinplot(
        y=lengths,
        inner="box",
        cut=0,
        bw_adjust=0.5,
        width=0.9
    )

    plt.title(f"Violin Plot of Sequence Lengths [{field if field else 'auto-detected'}]")
    plt.ylabel("Token Count")
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)

    if configs:
        annotate_configs(ax, configs, lengths_arr, colors, orientation="violin")

    if logy:
        plt.yscale("log")

    violin_path = output_path.replace(".png", "_violin.png")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(violin_path, bbox_inches="tight")
    print(f"Saved violin plot with configs to {violin_path}")

    # --- Histogram Plot ---
    plt.figure(figsize=(15, 6))
    ax = sns.histplot(lengths, bins=500, kde=False)
    plt.title(f"Histogram of Sequence Lengths [{field if field else 'auto-detected'}]")
    plt.xlabel("Token Count")
    plt.ylabel("Number of Samples")
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)

    if configs:
        annotate_configs(ax, configs, lengths_arr, colors, orientation="hist")

    if logy:
        plt.xscale("log")

    hist_path = output_path.replace(".png", "_hist.png")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(hist_path, bbox_inches="tight")
    print(f"Saved histogram plot with configs to {hist_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw violin and histogram plots of sequence lengths in tokens.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset file (json/jsonl/txt).")
    parser.add_argument("--model", type=str, required=True, help="Path or Hugging Face model name.")
    parser.add_argument("--field", type=str, default=None, help="Field in JSON/JSONL file to use for text.")
    parser.add_argument("--configs", type=str, default=None, help="Path to JSONL configs file.")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sequence_lengths.png"),
        help="Base path to save the plots (default: script directory)."
    )
    parser.add_argument("--logy", action="store_true", help="Use log scale on Y-axis (violin) or X-axis (hist).")
    args = parser.parse_args()

    main(args.dataset, args.model, args.field, args.output, args.logy, args.configs)


# python /home/a_s87063/repos/perf-pilot/analysis/token_length_violin.py --dataset /home/a_s87063/repos/perf-pilot/datasets/jit_dp/apachejit_llm_small_struc.jsonl --model /home/a_s87063/repos/perf-pilot/LLMs/pretrained/sequence-classification/meta-llama/Llama-3.1-8B --configs /home/a_s87063/repos/perf-pilot/analysis/seq_len_confis.jsonl --field prompt --logy --output /home/a_s87063/repos/perf-pilot/analysis/output
