import os
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

# --------------------- argparse ---------------------
parser = argparse.ArgumentParser(description="Plot fine-tuning metrics for LLaMA JIT defect prediction.")
parser.add_argument("--metrics_dir", type=str, required=True, help="Path to metrics directory containing live_metrics.jsonl and best_model_metrics.json")
args = parser.parse_args()

metrics_dir = args.metrics_dir
live_metrics_path = os.path.join(metrics_dir, "live_metrics.jsonl")
best_model_path = os.path.join(metrics_dir, "best_model_metrics.json")

plots_dir = os.path.join(metrics_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# --------------------- Load Metrics ---------------------
live_metrics = []
if os.path.exists(live_metrics_path):
    with open(live_metrics_path) as f:
        for line in f:
            live_metrics.append(json.loads(line))

with open(best_model_path) as f:
    best_model_metrics = json.load(f)

# Try to load raw predictions if available
labels = best_model_metrics.get("true_labels")
probs = best_model_metrics.get("probabilities")
preds = best_model_metrics.get("predictions")

# --------------------- 1. Loss Curve ---------------------
train_steps = [m["step"] for m in live_metrics if m["type"] == "train" and "loss" in m["metrics"]]
train_loss = [m["metrics"]["loss"] for m in live_metrics if m["type"] == "train" and "loss" in m["metrics"]]
eval_steps = [m["step"] for m in live_metrics if m["type"] == "eval" and "eval_loss" in m["metrics"]]
eval_loss = [m["metrics"]["eval_loss"] for m in live_metrics if m["type"] == "eval" and "eval_loss" in m["metrics"]]

plt.figure(figsize=(10, 6))
plt.plot(train_steps, train_loss, label="Train Loss")
plt.plot(eval_steps, eval_loss, label="Eval Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training & Evaluation Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plots_dir, "loss_curve.png"))
plt.close()

# --------------------- 2. Evaluation Metric Trends ---------------------
plt.figure(figsize=(10, 6))
for metric in ["eval_f1", "eval_precision", "eval_recall", "eval_accuracy"]:
    steps = [m["step"] for m in live_metrics if m["type"] == "eval" and metric in m["metrics"]]
    values = [m["metrics"][metric] for m in live_metrics if m["type"] == "eval" and metric in m["metrics"]]
    if steps:
        plt.plot(steps, values, label=metric)
plt.xlabel("Steps")
plt.ylabel("Score")
plt.title("Evaluation Metrics over Steps")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plots_dir, "eval_metrics_curve.png"))
plt.close()

# If probabilities and labels are not available, skip further plots
if probs is None or labels is None:
    print("⚠️ Skipping ROC, PR, and Confusion Matrix plots due to missing predictions in metrics.")
else:
    probs = np.array(probs)
    labels = np.array(labels)
    preds = np.array(preds)

    # --------------------- 3. ROC Curve ---------------------
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, "roc_curve.png"))
    plt.close()

    # --------------------- 4. Precision-Recall Curve ---------------------
    precision, recall, _ = precision_recall_curve(labels, probs)
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, "pr_curve.png"))
    plt.close()

    # --------------------- 5. Confusion Matrix ---------------------
    conf_mat = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Buggy", "Buggy"], yticklabels=["Non-Buggy", "Buggy"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"))
    plt.close()

    # --------------------- 6. Recall@Top-K% Curve ---------------------
    sorted_indices = np.argsort(-probs)
    sorted_labels = labels[sorted_indices]
    percentages = np.linspace(0.01, 0.5, 30)
    recalls = []
    total_positives = np.sum(labels)
    for pct in percentages:
        k = int(len(probs) * pct)
        recall_at_k = np.sum(sorted_labels[:k]) / total_positives
        recalls.append(recall_at_k)

    plt.figure(figsize=(10, 6))
    plt.plot(percentages * 100, recalls)
    plt.xlabel("Top-K% of Commits Reviewed")
    plt.ylabel("Recall")
    plt.title("Recall@Top-K% Curve")
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, "recall_at_topk_curve.png"))
    plt.close()

    # --------------------- 7. Prediction Confidence Histogram ---------------------
    plt.figure(figsize=(8, 6))
    plt.hist(probs, bins=25, color='skyblue', edgecolor='black')
    plt.xlabel("Predicted Probability (Buggy)")
    plt.ylabel("Frequency")
    plt.title("Prediction Confidence Histogram")
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, "confidence_histogram.png"))
    plt.close()

print(f"✅ All plots saved to: {plots_dir}")
