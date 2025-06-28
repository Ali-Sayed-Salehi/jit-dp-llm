import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, f1_score, roc_auc_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import joblib
from imblearn.over_sampling import SMOTE
from itertools import product
from collections import Counter
import argparse
from sklearn.metrics import average_precision_score

# ---------------------------- Argparse ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--selection_metric", type=str, default="recall@top_10%",
                    help="Metric to select the best model: recall@top_5%, recall@top_10%, recall@top_30%, f1_score, precision, recall, auc, accuracy")
args = parser.parse_args()
selection_metric = args.selection_metric

valid_topk_metrics = {"recall@top_5%", "recall@top_10%", "recall@top_30%"}
print(f"🔧 Selecting best model based on: {selection_metric}")

# ---------------------------- constants ----------------------------

REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(f"✅ Detected REPO_PATH: {REPO_PATH}")

run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = f"{REPO_PATH}/llama/training/run_{run_timestamp}"

metrics_dir = os.path.join(output_dir, "metrics")
model_dir = os.path.join(output_dir, "model")

training_dirs = [output_dir, metrics_dir, model_dir]
for directory in training_dirs:
    os.makedirs(directory, exist_ok=True)

# ---------------------------- Load Dataset ----------------------------

dataset_path = os.path.join(REPO_PATH, "datasets", "jit_dp", "apachejit_logreg_small.csv")
df = pd.read_csv(dataset_path)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Chronologically split the dataset: 64% train, 16% eval, 20% test
n_total = len(X)
n_train = int(n_total * 0.64)
n_eval = int(n_total * 0.16)

X_train = X[:n_train]
y_train = y[:n_train]

X_eval = X[n_train:n_train + n_eval]
y_eval = y[n_train:n_train + n_eval]

X_test = X[n_train + n_eval:]
y_test = y[n_train + n_eval:]

print(f"🔢 Dataset split sizes — Train: {len(X_train)}, Eval: {len(X_eval)}, Test: {len(X_test)}")

original_class_distribution = {str(k): int(v) for k, v in Counter(y_train).items()}

# ---------------------------- Scaling ----------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_eval_scaled = scaler.transform(X_eval)
X_test_scaled = scaler.transform(X_test)

# ---------------------------- Imbalance Handling ----------------------------

def apply_class_imbalance_fix(strategy, X, y):
    if strategy == "class_weight":
        return X, y, 'balanced'
    elif strategy == "smote":
        sm = SMOTE()
        X_res, y_res = sm.fit_resample(X, y)
        return X_res, y_res, None
    elif strategy == "oversample":
        df_combined = pd.DataFrame(np.hstack((X, y.reshape(-1, 1))))
        majority = df_combined[df_combined.iloc[:, -1] == 0]
        minority = df_combined[df_combined.iloc[:, -1] == 1]
        minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
        upsampled = pd.concat([majority, minority_upsampled])
        X_resampled = upsampled.iloc[:, :-1].values
        y_resampled = upsampled.iloc[:, -1].values.astype(int)
        return X_resampled, y_resampled, None
    elif strategy == "downsample":
        df_combined = pd.DataFrame(np.hstack((X, y.reshape(-1, 1))))
        majority = df_combined[df_combined.iloc[:, -1] == 0]
        minority = df_combined[df_combined.iloc[:, -1] == 1]
        majority_downsampled = resample(majority, replace=False, n_samples=len(minority), random_state=42)
        downsampled = pd.concat([majority_downsampled, minority])
        X_resampled = downsampled.iloc[:, :-1].values
        y_resampled = downsampled.iloc[:, -1].values.astype(int)
        return X_resampled, y_resampled, None
    return X, y, None

# ---------------------------- Evaluation Metric ----------------------------

def recall_at_top_k(pred_probs, true_labels, percentages=[0.05, 0.1, 0.3]):
    results = {}
    total_positives = np.sum(true_labels)
    sorted_indices = np.argsort(-pred_probs)
    sorted_labels = true_labels[sorted_indices]
    for pct in percentages:
        k = int(len(pred_probs) * pct)
        top_k_labels = sorted_labels[:k]
        recall = np.sum(top_k_labels) / total_positives if total_positives > 0 else 0.0
        results[f"recall@top_{int(pct * 100)}%"] = recall
    return results

# ---------------------------- Grid Search Setup ----------------------------

imbalance_strategies = [None, "class_weight", "smote", "oversample", "downsample"]
penalties = ["l1", "l2", "elasticnet", None]
solvers = ["liblinear", "lbfgs", "newton-cg", "sag", "saga"]
C_values = [0.01, 0.1, 1, 10, 100]
valid_combinations = {
    'l1': ['liblinear', 'saga'],
    'l2': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'],
    'elasticnet': ['saga'],
    None: ['lbfgs', 'newton-cg', 'sag', 'saga']
}

best_metric_value = -1
best_model = None
best_scaler = None
best_metrics = None
all_results = []

# ---------------------------- Training Loop ----------------------------

for strategy, penalty, solver, C in product(imbalance_strategies, penalties, solvers, C_values):
    if solver not in valid_combinations.get(penalty, []):
        continue

    print(f"\n🔍 Trying: imbalance strategy={strategy}, penalty={penalty}, solver={solver}, C={C}")
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)
    X_fixed, y_fixed, class_weight = apply_class_imbalance_fix(strategy, X_train_scaled, y_train)
    new_class_distribution = {str(k): int(v) for k, v in Counter(y_fixed).items()}

    params = {
        "penalty": penalty,
        "solver": solver,
        "max_iter": 1000,
        "class_weight": class_weight
    }
    if penalty is not None:
        params["C"] = C
    if penalty == "elasticnet":
        params["l1_ratio"] = 0.5

    try:
        model = LogisticRegression(**params)
        model.fit(X_fixed, y_fixed)
    except Exception as e:
        print(f"❌ Skipping due to error: {e}")
        continue

    y_proba = model.predict_proba(X_eval_scaled)[:, 1]
    y_pred = model.predict(X_eval_scaled)

    recall_top_k = recall_at_top_k(y_proba, y_eval, percentages=[0.05, 0.1, 0.3])
    recall = recall_score(y_eval, y_pred, zero_division=0)
    accuracy = accuracy_score(y_eval, y_pred)
    precision = precision_score(y_eval, y_pred, zero_division=0)
    f1 = f1_score(y_eval, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_eval, y_proba)
    pr_auc = average_precision_score(y_eval, y_proba)

    # report = classification_report(y_eval, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_eval, y_pred).tolist()

    metrics = {
        "imbalance strategy": strategy,
        "penalty": str(penalty),
        "solver": solver,
        "C": C,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": conf_matrix,
        "recall_at_top_k": recall_top_k,
        "class_distribution_before": original_class_distribution,
        "class_distribution_after": new_class_distribution
    }

    all_results.append(metrics)

    current_metric_value = (
        recall_top_k.get(selection_metric, 0.0)
        if selection_metric in recall_top_k
        else metrics.get(selection_metric, 0.0)
    )

    if current_metric_value > best_metric_value:
        best_metric_value = current_metric_value
        best_model = model
        best_scaler = scaler
        best_metrics = metrics

# ---------------------------- Save Results ----------------------------

best_metrics["selection_metric"] = selection_metric

with open(os.path.join(metrics_dir, "all_metrics.json"), "w") as f:
    json.dump(all_results, f, indent=2)

with open(os.path.join(metrics_dir, "best_metrics.json"), "w") as f:
    json.dump(best_metrics, f, indent=2)

joblib.dump(best_model, os.path.join(model_dir, "logistic_regression_best_model.joblib"))
joblib.dump(best_scaler, os.path.join(model_dir, "scaler.joblib"))

print("\n🏆 Best config:")
print(json.dumps(best_metrics, indent=2))
print(f"📂 Results saved in {output_dir}")

# ---------------------------- Final Inference on Held-out Test Set ----------------------------

X_final_test_scaled = best_scaler.transform(X_test)
y_final_proba = best_model.predict_proba(X_final_test_scaled)[:, 1]
y_final_pred = best_model.predict(X_final_test_scaled)

final_recall_top_k = recall_at_top_k(y_final_proba, y_test, percentages=[0.05, 0.1, 0.3])

# Compute maximum possible recall at each top-k cutoff
max_possible_recall_top_k = {}
total_positives = np.sum(y_test)
n_samples = len(y_test)

for pct in [0.05, 0.1, 0.3]:
    k = int(n_samples * pct)
    positives_in_top_k = min(total_positives, k)
    max_possible_recall = positives_in_top_k / total_positives if total_positives > 0 else 0.0
    max_possible_recall_top_k[f"max_possible_recall@top_{int(pct * 100)}%"] = max_possible_recall

final_recall = recall_score(y_test, y_final_pred, zero_division=0)
final_accuracy = accuracy_score(y_test, y_final_pred)
final_precision = precision_score(y_test, y_final_pred, zero_division=0)
final_f1 = f1_score(y_test, y_final_pred, zero_division=0)
final_roc_auc = roc_auc_score(y_test, y_final_proba)
final_pr_auc = average_precision_score(y_test, y_final_proba)

final_conf_matrix = confusion_matrix(y_test, y_final_pred).tolist()

final_test_metrics = {
    "accuracy": final_accuracy,
    "precision": final_precision,
    "recall": final_recall,
    "f1_score": final_f1,
    "roc_auc": final_roc_auc,
    "pr_auc": final_pr_auc,
    "confusion_matrix": final_conf_matrix,
    "recall_at_top_k": final_recall_top_k,
    "max_possible_recall_at_top_k": max_possible_recall_top_k
}

final_test_output = {
    "metrics": final_test_metrics,
    "predictions": y_final_pred.tolist(),
    "probabilities": y_final_proba.tolist(),
    "true_labels": y_test.tolist()
}

with open(os.path.join(metrics_dir, "final_test_results.json"), "w") as f:
    json.dump(final_test_output, f, indent=2)

print("\n🧪 Final test set evaluation:")
print(json.dumps(final_test_metrics, indent=2))
