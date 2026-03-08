# train_distilbert_baseline.py
import os
import json
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --------- CONFIG ----------
DATA_PATH = "datasets/cleared_dataset.csv"
OUT_DIR = "./distilbert_baseline"
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
BATCH = 16
EPOCHS = 7
LR = 1e-5
MAX_LEN = 96
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --------- DATA ----------
df = pd.read_csv(DATA_PATH).dropna(subset=["tweets","class"]).reset_index(drop=True)
df["label"] = df["class"].apply(lambda x: 1 if str(x).strip().lower() == "figurative" else 0)
texts = df["tweets"].tolist()
labels = df["label"].tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.1, random_state=SEED, stratify=labels
)

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def encode(texts):
    return tokenizer(texts, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt')

train_enc = encode(train_texts)
val_enc = encode(val_texts)

class EncDataset(Dataset):
    def __init__(self, encodings, labels):
        self.enc = encodings
        self.labels = torch.tensor(labels, dtype=torch.float)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {
            "input_ids": self.enc["input_ids"][idx],
            "attention_mask": self.enc["attention_mask"][idx],
            "labels": self.labels[idx]
        }

train_ds = EncDataset(train_enc, train_labels)
val_ds = EncDataset(val_enc, val_labels)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False)

# --------- MODEL ----------
class DistilBinary(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(self.encoder.config.dim, 1)
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]
        pooled = self.drop(pooled)
        logits = self.fc(pooled).squeeze(-1)
        return logits

model = DistilBinary().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05*total_steps), num_training_steps=total_steps)

# --------- TRAIN LOOP ----------
best_f1 = 0.0
for epoch in range(EPOCHS):
    model.train()
    losses = []
    for batch in tqdm(train_loader, desc=f"Train {epoch+1}/{EPOCHS}"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    # validation
    model.eval()
    preds, probs, golds = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            p = torch.sigmoid(logits).cpu().numpy()
            preds.extend((p>=0.5).astype(int).tolist())
            probs.extend(p.tolist())
            golds.extend(labels.cpu().numpy().astype(int).tolist())
    acc = accuracy_score(golds, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(golds, preds, average='binary', zero_division=0)
    print(f"\nEpoch {epoch+1} val — acc: {acc:.4f} prec: {prec:.4f} rec: {rec:.4f} f1: {f1:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), os.path.join(OUT_DIR, "distilbert_baseline_best.pt"))
        tokenizer.save_pretrained(OUT_DIR)
        print("Saved best DistilBERT baseline model.")

# save eval csv
pd.DataFrame({"text": val_texts, "gold": golds, "pred": preds, "prob": probs}).to_csv(os.path.join(OUT_DIR, "distilbert_baseline_eval.csv"), index=False)
print("Training finished. Best F1:", best_f1)


# ---- Metrics Logging Section (universal) ----
import json, csv
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
acc = accuracy_score(golds, preds)
prec, rec, f1, _ = precision_recall_fscore_support(golds, preds, average='binary', zero_division=0)


# Ensure you have acc, prec, rec, f1 defined
metrics = {
    "model": os.path.basename(__file__).replace(".py", ""),
    "accuracy": round(float(acc), 4),
    "precision": round(float(prec), 4),
    "recall": round(float(rec), 4),
    "f1": round(float(f1), 4)
}

# Save individual metrics JSON
json_path = os.path.join(OUT_DIR, "metrics_summary.json")
with open(json_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Saved metrics summary to {json_path}")

# Save also as CSV (for quick view)
csv_path = os.path.join(OUT_DIR, "metrics_summary.csv")
pd.DataFrame([metrics]).to_csv(csv_path, index=False)
print(f"Saved metrics CSV to {csv_path}")

# Append to global metrics file
global_csv = "./all_model_metrics.csv"
fieldnames = ["model", "accuracy", "precision", "recall", "f1"]

# Append or create file
file_exists = os.path.exists(global_csv)
with open(global_csv, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
    writer.writerow(metrics)
print(f"✅ Appended results to {global_csv}\n")
