# train_bilstm.py (patched)
import os
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast
import json, csv

# --------- CONFIG ----------
DATA_PATH = "Datasets/cleaned_dataset.csv"
OUT_DIR = "./bilstm_model"
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
BATCH = 32
EPOCHS = 7
LR = 1e-3               # changed for embedding+LSTM training
MAX_LEN = 96
EMBED_DIM = 100
HIDDEN = 128
NUM_LAYERS = 1
DROPOUT = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0         # safe on Windows

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --------- DATA ----------
df = pd.read_csv(DATA_PATH).reset_index(drop=True)
df = df.dropna(subset=["tweets", "class"]).reset_index(drop=True)

# If cleaned_dataset already provided label column, use it. Otherwise map classes.
if "label" in df.columns:
    df["label"] = df["label"].astype(int)
else:
    df["label"] = df["class"].astype(str).str.lower().apply(
        lambda x: 1 if x in {"irony", "sarcasm", "figurative"} else 0
    )

print("Label distribution (global):")
print(df["label"].value_counts())

texts = df["tweets"].tolist()
labels = df["label"].tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.1, random_state=SEED, stratify=labels
)

print(f"Train size: {len(train_texts)}, Val size: {len(val_texts)}")
print("Train positives:", int(sum(train_labels)), "Train negatives:", len(train_labels) - int(sum(train_labels)))

# tokenizer for vocab size (embedding)
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
VOCAB_SIZE = tokenizer.vocab_size

def encode_texts(texts):
    enc = tokenizer(texts, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt')
    return enc['input_ids']

train_input_ids = encode_texts(train_texts)
val_input_ids = encode_texts(val_texts)

class SeqDataset(Dataset):
    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx]
        }

train_ds = SeqDataset(train_input_ids, train_labels)
val_ds = SeqDataset(val_input_ids, val_labels)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)

# --------- MODEL ----------
class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden, n_layers=1, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=n_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if n_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, input_ids):
        x = self.embed(input_ids)                 # (B, L, E)
        out, _ = self.lstm(x)                     # (B, L, 2H)
        pooled = out.mean(dim=1)                  # mean pooling across tokens -> (B, 2H)
        pooled = self.drop(pooled)
        logits = self.fc(pooled).squeeze(-1)
        return logits

model = BiLSTMModel(VOCAB_SIZE, EMBED_DIM, HIDDEN, NUM_LAYERS, DROPOUT).to(DEVICE)

# --------- LOSS: pos_weight (capped) ----------
n_pos = int(sum(train_labels))
n_neg = len(train_labels) - n_pos
raw_pw = float(n_neg) / max(1.0, float(n_pos))
pw_cap = 3.0
pw = min(raw_pw, pw_cap)
pos_weight = torch.tensor([pw], dtype=torch.float).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
print("Using pos_weight (raw, capped):", raw_pw, pw)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# --------- TRAIN LOOP ----------
best_f1 = 0.0
for epoch in range(EPOCHS):
    model.train()
    losses = []
    loop = tqdm(train_loader, desc=f"Train {epoch+1}/{EPOCHS}")
    for batch in loop:
        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        loop.set_postfix(loss=np.mean(losses))

    # VALIDATION
    model.eval()
    preds, probs, golds = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            logits = model(input_ids)
            p = torch.sigmoid(logits).cpu().numpy()
            preds.extend((p >= 0.5).astype(int).tolist())
            probs.extend(p.tolist())
            golds.extend(labels.cpu().numpy().astype(int).tolist())

    acc = accuracy_score(golds, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(golds, preds, average='binary', zero_division=0)
    print(f"\nEpoch {epoch+1} val — acc: {acc:.4f} prec: {prec:.4f} rec: {rec:.4f} f1: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), os.path.join(OUT_DIR, "bilstm_best.pt"))
        print("Saved best BiLSTM model.")

# Save validation predictions
pd.DataFrame({"text": val_texts, "gold": golds, "pred": preds, "prob": probs}).to_csv(
    os.path.join(OUT_DIR, "bilstm_eval.csv"), index=False
)
print("BiLSTM training finished. Best F1:", best_f1)

# --------- METRICS LOGGING ----------
metrics = {
    "model": os.path.basename(__file__).replace(".py", ""),
    "accuracy": round(float(acc), 4),
    "precision": round(float(prec), 4),
    "recall": round(float(rec), 4),
    "f1": round(float(f1), 4)
}

# Save metrics JSON
json_path = os.path.join(OUT_DIR, "metrics_summary.json")
with open(json_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Saved metrics summary to {json_path}")

# Save metrics CSV
csv_path = os.path.join(OUT_DIR, "metrics_summary.csv")
pd.DataFrame([metrics]).to_csv(csv_path, index=False)
print(f"Saved metrics CSV to {csv_path}")

# Append to global metrics file
global_csv = "./all_model_metrics.csv"
fieldnames = ["model", "accuracy", "precision", "recall", "f1"]
file_exists = os.path.exists(global_csv)
with open(global_csv, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
    writer.writerow(metrics)
print(f"✅ Appended results to {global_csv}\n")
