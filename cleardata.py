# clean_dataset.py
# Run this before training to clean and prepare your dataset.

import pandas as pd
import re
import os

# === CONFIG ===
INPUT_PATH = "E:/SENTIMENT ANALYSIS/Datasets/test.csv"
OUTPUT_PATH = "E:/SENTIMENT ANALYSIS/Datasets/cleaned_dataset.csv"

# === LOAD DATA ===
df = pd.read_csv(INPUT_PATH)
print(f"Loaded {len(df)} rows from dataset.")

# Drop rows with missing tweets or class
df = df.dropna(subset=["tweets", "class"]).reset_index(drop=True)
print(f"After dropping nulls: {len(df)} rows remain.")

# === TEXT CLEANING FUNCTION ===
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)         # remove URLs
    text = re.sub(r"@\w+", "", text)            # remove @mentions
    text = re.sub(r"#\w+", "", text)            # remove hashtags (prevents leakage)
    text = re.sub(r"&amp;", "&", text)          # decode '&amp;'
    text = re.sub(r"\s+", " ", text).strip()    # remove extra spaces
    return text

# Apply cleaning
df["tweets"] = df["tweets"].apply(clean_text)

# === FIX LABELS ===
# Combine figurative, irony, sarcasm into one class (1 = ironic / figurative)
figurative_classes = {"irony", "sarcasm", "figurative"}
df["label"] = df["class"].astype(str).str.lower().apply(lambda x: 1 if x in figurative_classes else 0)

# === REMOVE DUPLICATES ===
before = len(df)
df = df.drop_duplicates(subset=["tweets"]).reset_index(drop=True)
after = len(df)
print(f"Removed {before - after} duplicate tweets.")

# === REMOVE SHORT TWEETS ===
before = len(df)
df = df[df["tweets"].str.split().str.len() >= 2].reset_index(drop=True)
after = len(df)
print(f"Removed {before - after} very short tweets (1 word or less).")

# === SAVE CLEANED DATA ===
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Cleaned dataset saved to: {OUTPUT_PATH}")

# === SUMMARY ===
print("\n--- Dataset Summary ---")
print(df["label"].value_counts(normalize=True).rename("proportion"))
print("Total samples:", len(df))
print("Nulls per column:\n", df.isnull().sum())
print("Average tweet length:", df["tweets"].str.split().str.len().mean())
print("\nExample tweets:")
print(df.sample(5)["tweets"].tolist())
