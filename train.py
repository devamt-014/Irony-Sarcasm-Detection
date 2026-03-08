# train.py — cleaned, Windows-safe, GPU-enabled
import os
import random
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DistilBertTokenizerFast, get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

# ----------------- SETTINGS -----------------
DATA_PATH = "E:/SENTIMENT ANALYSIS/Datasets/cleaned_dataset.csv"           # point to your CSV (relative to project folder)
MODEL_NAME = "distilbert-base-uncased"
OUT_DIR = "./irony_model"
os.makedirs(OUT_DIR, exist_ok=True)

# runtime / resource settings
USE_CUDA = torch.cuda.is_available()
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "0"))  # safe default 0 on Windows
BATCH_SIZE = 32 if USE_CUDA else 8
MAX_LEN = int(os.getenv("MAX_LEN", "96"))
EPOCHS = int(os.getenv("EPOCHS", "7"))
LR = float(os.getenv("LR", "1e-5"))
SEED = int(os.getenv("SEED", "42"))
PIN_MEM = True if USE_CUDA else False

# tune PyTorch threads for CPU
torch.set_num_threads(int(os.getenv("TORCH_THREADS", "4")))

# ----------------- HELPERS / CLASSES -----------------
class DistilIronyModel(nn.Module):
    def __init__(self, model_name=MODEL_NAME):
        super().__init__()
        from transformers import DistilBertModel
        self.encoder = DistilBertModel.from_pretrained(model_name)
        hidden = self.encoder.config.dim
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled).squeeze(-1)
        return logits

class PreTokenizedDataset(Dataset):
    def __init__(self, encodings, labels):
        self.enc = encodings
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.enc.items()}
        item['labels'] = self.labels[idx]
        return item

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ----------------- MAIN -----------------
def main():
    import importlib
    global torch
    torch = importlib.import_module("torch")
    set_seed()
    print("Device:", "cuda" if USE_CUDA else "cpu")
    print("NUM_WORKERS:", NUM_WORKERS, "BATCH_SIZE:", BATCH_SIZE, "MAX_LEN:", MAX_LEN, "EPOCHS:", EPOCHS)

    # Load dataset
    print("Loading dataset from:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["tweets", "class"]).reset_index(drop=True)

    # Map textual classes to binary label: 'figurative' -> 1 else 0
    df["label"] = df["class"].apply(lambda x: 1 if str(x).strip().lower() == "figurative" else 0)
    print("Label counts (entire dataset):", df["label"].value_counts().to_dict())

    # Train/val split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["tweets"].tolist(), df["label"].tolist(), test_size=0.1, random_state=SEED,
        stratify=df["label"].tolist()
    )
    print(f"Train size: {len(train_texts)}, Val size: {len(val_texts)}")
    print("Train positives:", int(sum(train_labels)), "Train negatives:", len(train_labels) - int(sum(train_labels)))

    # Tokenizer & pre-tokenize (speeds up CPU)
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    print("Pre-tokenizing datasets...")
    train_enc = tokenizer(train_texts, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt')
    val_enc = tokenizer(val_texts, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt')

    train_ds = PreTokenizedDataset(train_enc, train_labels)
    val_ds = PreTokenizedDataset(val_enc, val_labels)

    # Model, optimizer, scheduler
    device = torch.device("cuda" if USE_CUDA else "cpu")
    model = DistilIronyModel(MODEL_NAME).to(device)

    # Try torch.compile only if Triton/inductor appear supported; otherwise skip
    compiled = False
    try:
        import torch._inductor as _ind  # quick presence check
        try:
            import triton  # may raise
            model = torch.compile(model)
            compiled = True
            print("torch.compile applied.")
        except Exception:
            print("Skipping torch.compile: Triton/compilation support missing on this system.")
    except Exception:
        print("Skipping torch.compile: inductor not available or unsupported.")

    # Safety: let Dynamo fall back to eager if compile fails at runtime
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
    except Exception:
        pass

    # Pos weight for imbalanced BCE
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    if n_pos == 0:
        raise ValueError("No positive examples found in training labels.")
    pos_weight = torch.tensor([(n_neg / (n_pos + 1e-8))], dtype=torch.float).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print("pos_weight:", float(pos_weight.item()))

    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = max(1, int(len(train_ds) / BATCH_SIZE) * EPOCHS)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05 * total_steps),
                                                num_training_steps=total_steps)

    # WeightedRandomSampler to oversample positives
    label_list = np.array(train_labels, dtype=int)
    counts = np.bincount(label_list)
    counts = np.where(counts == 0, 1, counts)
    weights = 1.0 / np.array([counts[label] for label in label_list])
    weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEM)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEM)

    print(f"Train loader batches: {len(train_loader)}, Val loader batches: {len(val_loader)}")

    # AMP (mixed precision) if CUDA available
    scaler = None
    use_amp = USE_CUDA
    if use_amp:
        try:
            from torch.cuda.amp import GradScaler, autocast
            scaler = GradScaler()
            print("AMP enabled (mixed precision).")
        except Exception:
            use_amp = False
            print("AMP not available; falling back to full precision.")

    best_f1 = 0.0
    best_path = None

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        losses = []
        loop = tqdm(train_loader, desc=f"Train Epoch {epoch+1}")
        for batch in loop:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)

            optimizer.zero_grad()
            if use_amp and scaler is not None:
                with autocast():
                    logits = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            scheduler.step()
            losses.append(loss.item())
            loop.set_postfix(loss=np.mean(losses))

        # Validation
        model.eval()
        preds = []
        golds = []
        probs = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                prob = torch.sigmoid(logits).cpu().numpy()
                batch_preds = (prob >= 0.5).astype(int).tolist()
                preds.extend(batch_preds)
                probs.extend(prob.tolist())
                golds.extend(labels.cpu().numpy().astype(int).tolist())

        preds = np.array(preds)
        golds = np.array(golds)
        probs = np.array(probs).reshape(-1)

        acc = accuracy_score(golds, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(golds, preds, average='binary', zero_division=0)
        print(f"\nEpoch {epoch+1} validation — acc: {acc:.4f}, prec: {precision:.4f}, rec: {recall:.4f}, f1: {f1:.4f}\n")

        # multi-threshold quick check
        for thr in [0.3, 0.4, 0.5]:
            preds_t = (probs >= thr).astype(int)
            p, r, f, _ = precision_recall_fscore_support(golds, preds_t, average='binary', zero_division=0)
            print(f" thr={thr:.2f} -> prec={p:.4f} rec={r:.4f} f1={f:.4f}")

        # diagnostics
        print("VAL size:", len(golds))
        print("Gold positives:", int(golds.sum()), "Gold negatives:", len(golds) - int(golds.sum()))
        print("Predicted positives (0.5):", int(preds.sum()), "Predicted negatives:", len(preds) - int(preds.sum()))
        if len(probs) > 0:
            print("Probs: mean {:.4f}, min {:.4f}, max {:.4f}".format(probs.mean(), probs.min(), probs.max()))
            print("Prob quartiles:", np.percentile(probs, [0, 25, 50, 75, 100]))

        # save val outputs
        val_out = [{"text": t, "gold": int(g), "prob": float(p)} for t, g, p in zip(val_texts, golds.tolist(), probs.tolist())]
        with open(os.path.join(OUT_DIR, f"val_epoch{epoch+1}.json"), "w", encoding="utf8") as f:
            json.dump(val_out, f, ensure_ascii=False, indent=2)

        # checkpoint
        ckpt_path = os.path.join(OUT_DIR, f"checkpoint_epoch{epoch+1}.pt")
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, ckpt_path)
        print("Saved checkpoint:", ckpt_path)

        if f1 > best_f1:
            best_f1 = f1
            best_path = os.path.join(OUT_DIR, "best_model.pt")
            torch.save(model.state_dict(), best_path)
            tokenizer.save_pretrained(OUT_DIR)
            print(f"Saved best model (f1={best_f1:.4f}) to {OUT_DIR}")

    print("Training finished. Best F1:", best_f1)
    print("Best model path:", best_path)

    # ---- Metrics Logging Section (safe + auto) ----
    print("\nSaving evaluation metrics...")

    # Compute metrics safely (final validation results are still in scope)
    acc = accuracy_score(golds, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(golds, preds, average='binary', zero_division=0)

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

    # Append to global metrics file (for cross-model comparison)
    global_csv = "./all_model_metrics.csv"
    fieldnames = ["model", "accuracy", "precision", "recall", "f1"]
    file_exists = os.path.exists(global_csv)
    with open(global_csv, "a", newline="") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)
    print(f"✅ Appended results to {global_csv}\n")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()

