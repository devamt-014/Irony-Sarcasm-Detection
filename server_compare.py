# server_compare.py
# Run: uvicorn server_compare:app --host 0.0.0.0 --port 8001

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DistilBertTokenizerFast
import torch
import os
from fastapi.middleware.cors import CORSMiddleware

# ----------------- App + CORS -----------------
app = FastAPI()
origins = ["*"]  # safe for local demo

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextIn(BaseModel):
    text: str

# ----------------- Load baseline sentiment model -----------------
BASELINE = "distilbert-base-uncased-finetuned-sst-2-english"
print("Loading baseline model...")
baseline_tokenizer = AutoTokenizer.from_pretrained(BASELINE)
baseline_model = AutoModelForSequenceClassification.from_pretrained(BASELINE).to(DEVICE)
baseline_model.eval()
print("Baseline ready!")

# ----------------- Load your irony model -----------------
# ----------------- Load your irony model (BiLSTM) -----------------
# We're switching to the BiLSTM model you trained (saved under ./bilstm_model)
IRONY_DIR = "./bilstm_model"
BILSTM_CHECKPOINT = os.path.join(IRONY_DIR, "bilstm_best.pt")
print("Loading BiLSTM model...")

# Use the same tokenizer as training (you used DistilBertTokenizerFast)
# Note: keep the same tokenizer name used during training
irony_tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

import torch.nn as nn

class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden=128, n_layers=1, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=n_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if n_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, input_ids, attention_mask=None):
        # attention_mask ignored for BiLSTM (kept for API compatibility)
        x = self.embed(input_ids)
        out, _ = self.lstm(x)
        pooled = out.mean(dim=1)
        pooled = self.drop(pooled)
        logits = self.fc(pooled).squeeze(-1)
        return logits

# instantiate with the tokenizer's vocab size (must match training)
VOCAB_SIZE = irony_tokenizer.vocab_size
# If you trained with different embed_dim/hidden, change the arguments below to match
irony_model = BiLSTMModel(VOCAB_SIZE, embed_dim=100, hidden=128, n_layers=1, dropout=0.3).to(DEVICE)

if not os.path.exists(BILSTM_CHECKPOINT):
    raise FileNotFoundError(f"BiLSTM checkpoint not found at {BILSTM_CHECKPOINT}. Train the BiLSTM or update path.")

state = torch.load(BILSTM_CHECKPOINT, map_location=DEVICE)
# support both plain state_dict and full checkpoint dict
if isinstance(state, dict) and "model_state_dict" in state:
    irony_model.load_state_dict(state["model_state_dict"])
else:
    irony_model.load_state_dict(state)
irony_model.eval()
print("BiLSTM (irony) model ready!")


# ----------------- Helper route -----------------
@app.post("/compare")
def compare(input: TextIn):
    text = input.text or ""

    # Baseline sentiment
    enc1 = baseline_tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out1 = baseline_model(**enc1)
        probs1 = torch.softmax(out1.logits, dim=-1).cpu().numpy()[0]
    # baseline_label and baseline_conf
    baseline_label = "Positive" if probs1[1] > probs1[0] else "Negative"
    baseline_conf = float(max(probs1))

        # ---- Irony model (BiLSTM) inference ----
    enc2 = irony_tokenizer(text, truncation=True, padding="max_length", max_length=96, return_tensors="pt")
    input_ids = enc2["input_ids"].to(DEVICE)
    with torch.no_grad():
        logits = irony_model(input_ids=input_ids)
    prob = float(torch.sigmoid(logits).cpu().item())


    # ----- Final Hybrid Heuristic -----
    # normalize / safety
    baseline_conf = float(baseline_conf)
    baseline_label = str(baseline_label)
    prob = float(prob)

    # thresholds (tweakable)
    IRONY_HIGH = 0.35      # sarcasm trigger threshold
    IRONY_STRONG = 0.60    # strong irony
    BASE_POS = 0.85        # strong positive sentiment
    BASE_NEG = 0.75        # strong negative sentiment (not used heavily here)

    # logic: combine sentiment + irony score
    if baseline_label == "Positive" and baseline_conf > BASE_POS and prob > IRONY_HIGH:
        # classic sarcastic pattern: positive sentiment but non-trivial irony score
        irony_label = "Ironic"
        irony_strength = min(prob * 1.3, 1.0)
    elif baseline_label == "Negative" and prob < IRONY_HIGH:
        # clearly negative and low irony -> not ironic
        irony_label = "Not Ironic"
        irony_strength = prob * 0.6
    elif prob > IRONY_STRONG:
        # high irony score regardless
        irony_label = "Ironic"
        irony_strength = prob
    else:
        # default: not ironic (scale down to reflect uncertainty)
        irony_label = "Not Ironic"
        irony_strength = prob * 0.7

    # round for nice JSON
    irony_strength = float(round(irony_strength, 4))

    return {
        "input": text,
        "baseline_sentiment": baseline_label,
        "baseline_confidence": baseline_conf,
        "irony_label": irony_label,
        "irony_prob": irony_strength
    }

# small health check
@app.get("/ping")
def ping():
    return {"status": "ok"}
