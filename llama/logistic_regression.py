import os
import json
from datetime import datetime

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib  # For saving the model

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

X = df.iloc[:, :-1].values  # features
y = df.iloc[:, -1].values   # labels

# ----------- Chronological Split -----------

split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# ----------- Feature Scaling -----------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------- Train Logistic Regression -----------

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# ----------- Evaluate Model -----------

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred).tolist()

print("✅ Accuracy:", accuracy)
print("\n📊 Classification Report:")
print(json.dumps(report, indent=2))
print("\n🧩 Confusion Matrix:")
print(conf_matrix)

# ----------- Save Metrics to File -----------

metrics = {
    "accuracy": accuracy,
    "classification_report": report,
    "confusion_matrix": conf_matrix
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
