import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json, csv

# ---------------- SETTINGS ----------------
OUT_DIR = "./lr_model"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- DATA LOADING ----------------
df = pd.read_csv("datasets/cleaned_dataset.csv")
df = df.dropna(subset=["tweets", "class"]).reset_index(drop=True)
df["label"] = df["class"].apply(lambda x: 1 if str(x).strip().lower() == "figurative" else 0)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    df["tweets"], df["label"], test_size=0.1, random_state=42, stratify=df["label"]
)

# ---------------- TF-IDF + MODEL ----------------
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# ---------------- PREDICTIONS ----------------
y_pred = model.predict(X_val_vec)

# ---------------- METRICS ----------------
acc = accuracy_score(y_val, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="binary", zero_division=0)

print("\n=== Logistic Regression Model Performance ===")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")

# ---------------- SAVE METRICS ----------------
metrics = {
    "model": os.path.basename(__file__).replace(".py", ""),
    "accuracy": round(float(acc), 4),
    "precision": round(float(prec), 4),
    "recall": round(float(rec), 4),
    "f1": round(float(f1), 4)
}

# Save per-model metrics JSON
json_path = os.path.join(OUT_DIR, "metrics_summary.json")
with open(json_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nSaved metrics summary to {json_path}")

# Save also as CSV
csv_path = os.path.join(OUT_DIR, "metrics_summary.csv")
pd.DataFrame([metrics]).to_csv(csv_path, index=False)
print(f"Saved metrics CSV to {csv_path}")

# Append to global metrics file for comparison
global_csv = "./all_model_metrics.csv"
fieldnames = ["model", "accuracy", "precision", "recall", "f1"]
file_exists = os.path.exists(global_csv)

with open(global_csv, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
    writer.writerow(metrics)

print(f"✅ Appended results to {global_csv}\n")
