import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, f1_score, roc_auc_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import joblib
from imblearn.over_sampling import SMOTE
from itertools import product
from collections import Counter
import argparse

# ---------------------------- Argparse ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--selection_metric", type=str, default="recall@top_10%",
                    help="Metric to select the best model: recall@top_10%, f1_score, precision, recall, auc, accuracy")
args = parser.parse_args()
selection_metric = args.selection_metric

valid_topk_metrics = {"recall@top_5%", "recall@top_10%", "recall@top_50%"}
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

split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

original_class_distribution = {str(k): int(v) for k, v in Counter(y_train).items()}

# ---------------------------- Scaling ----------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
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
    return X, y, None

# ---------------------------- Evaluation Metric ----------------------------

def recall_at_top_k(pred_probs, true_labels, percentages=[0.1]):
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

imbalance_strategies = [None, "class_weight", "smote", "oversample"]
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
    X_test_scaled = scaler.transform(X_test)
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

    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)

    recall_top_k = recall_at_top_k(y_proba, y_test, percentages=[0.05, 0.1, 0.5])
    recall = recall_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    # report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "imbalance strategy": strategy,
        "penalty": str(penalty),
        "solver": solver,
        "C": C,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc,
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
