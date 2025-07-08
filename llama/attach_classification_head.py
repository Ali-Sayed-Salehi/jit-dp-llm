#!/usr/bin/env python3

import os
import argparse
import torch
from torch import nn
from functools import wraps

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    LlamaForCausalLM,
    LlamaForSequenceClassification
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_lm_path", type=str, required=True, help="Path to base LLaMA Causal LM")
    parser.add_argument("--save_path", type=str, required=True, help="Root path to save the new classification model")
    args = parser.parse_args()

    REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    path_parts = os.path.normpath(args.base_lm_path).split(os.sep)
    model_name = path_parts[-1]
    org_name = path_parts[-2]

    SAVE_PATH = os.path.join(args.save_path, org_name, model_name)
    pretrained_classification_model_path = os.path.join(REPO_PATH, "LLMs", "pretrained", "sequence-classification", org_name, model_name)

    causal_lm = AutoModelForCausalLM.from_pretrained(args.base_lm_path, local_files_only=True)
    classification_lm = AutoModelForSequenceClassification.from_pretrained(pretrained_classification_model_path, local_files_only=True)

    # Create new classification model with same config
    config = classification_lm.config
    config.num_labels = 2
    config.problem_type = "single_label_classification"  # For normal classification

    final_model = AutoModelForSequenceClassification.from_config(config)

    # Copy backbone weights
    final_model.model.load_state_dict(causal_lm.model.state_dict())

    # Save the final model + config
    os.makedirs(SAVE_PATH, exist_ok=True)
    final_model.save_pretrained(SAVE_PATH)
    config.save_pretrained(SAVE_PATH)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_lm_path)
    tokenizer.save_pretrained(SAVE_PATH)

    print(f"✅ New classification model + tokenizer saved to: {SAVE_PATH}")

if __name__ == "__main__":
    main()
