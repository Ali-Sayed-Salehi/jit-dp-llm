import os
import json
from datetime import datetime
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import joblib
from imblearn.over_sampling import SMOTE
import argparse
import numpy as np
from collections import Counter

# ----------- Argument Parsing -----------

parser = argparse.ArgumentParser()
parser.add_argument(
    "--class_imbalance_fix",
    type=str,
    choices=["class_weight", "smote", "oversample"],
    help="Apply method to handle class imbalance",
    default=None
)
args = parser.parse_args()

# ---------------------------- constants  ----------------------------

REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(f"✅ Detected REPO_PATH: {REPO_PATH}")

run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = f"{REPO_PATH}/llama/training/run_{run_timestamp}"

metrics_dir = os.path.join(output_dir, "metrics")
model_dir = os.path.join(output_dir, "model")

training_dirs = [output_dir, metrics_dir, model_dir]
for directory in training_dirs:
    os.makedirs(directory, exist_ok=True)

# ----------- Load Dataset -----------

dataset_path = os.path.join(REPO_PATH, "datasets", "jit_dp", "apachejit_logreg_small.csv")
df = pd.read_csv(dataset_path)

print("📊 Features:")
print(df.dtypes)

X = df.iloc[:, :-1].values  # features
y = df.iloc[:, -1].values   # labels

# ----------- Chronological Split -----------

split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

print("Class distribution before class imbalance fix:", Counter(y_train))

# ----------- Feature Scaling -----------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ---------------------------- Apply Class Imbalance Fix ----------------------------

class_weight = None

if args.class_imbalance_fix == "class_weight":
    print("⚖️ Applying class_weight='balanced'")
    class_weight = 'balanced'

elif args.class_imbalance_fix == "smote":
    print("🔁 Applying SMOTE...")
    sm = SMOTE()
    X_train_scaled, y_train = sm.fit_resample(X_train_scaled, y_train)

elif args.class_imbalance_fix == "oversample":
    print("🔁 Applying naive oversampling of minority class...")
    # Concatenate data for resampling
    X_train_combined = np.concatenate((X_train_scaled, y_train.reshape(-1, 1)), axis=1)
    df_train = pd.DataFrame(X_train_combined)
    
    majority = df_train[df_train.iloc[:, -1] == 0]
    minority = df_train[df_train.iloc[:, -1] == 1]
    
    minority_upsampled = resample(minority,
                                  replace=True,
                                  n_samples=len(majority),
                                  random_state=42)
    df_upsampled = pd.concat([majority, minority_upsampled])
    
    X_train_scaled = df_upsampled.iloc[:, :-1].values
    y_train = df_upsampled.iloc[:, -1].values.astype(int)

else:
    print("🚫 No class imbalance fix applied.")

print("Class distribution after class imbalance fix:", Counter(y_train))

# ----------- Train Logistic Regression -----------

model = LogisticRegression(
    verbose=1,
    solver='lbfgs',
    max_iter=1000,
    class_weight=class_weight
)

model.fit(X_train_scaled, y_train)

# ----------- Evaluate Model -----------

def recall_at_top_k(pred_probs, true_labels, percentages=[0.1, 0.25, 0.5]):
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

# Get predicted probabilities for the positive class (bug)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

recall_top_k = recall_at_top_k(y_proba, y_test, percentages=[0.05, 0.1, 0.5])

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred).tolist()

print("\n📊 Classification Report:")
print(json.dumps(report, indent=2))
print("\n🧩 Confusion Matrix:")
print(conf_matrix)

# ----------- Save Metrics to File -----------

metrics = {
    "accuracy": accuracy,
    "classification_report": report,
    "confusion_matrix": conf_matrix,
    "recall_at_top_k": recall_top_k
}

metrics_path = os.path.join(metrics_dir, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"💾 Saved metrics to {metrics_path}")

# ----------- Save Model to File -----------

model_path = os.path.join(model_dir, "logistic_regression_model.joblib")
scaler_path = os.path.join(model_dir, "scaler.joblib")

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print(f"💾 Saved model to {model_path}")
print(f"💾 Saved scaler to {scaler_path}")
