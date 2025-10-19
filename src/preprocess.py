import pandas as pd
import re
from sklearn.model_selection import train_test_split
import os
from collections import Counter
import json


# -------------------------------
# 1️⃣  Urdu Text Normalization
# -------------------------------
def normalize_urdu(text):
    """
    Normalize Urdu text:
    - Remove diacritics
    - Standardize Alef, Yeh, and Kaaf forms
    - Remove punctuation
    """
    text = re.sub(r"[ًٌٍَُِّْٰ]", "", str(text))       # remove diacritics
    text = text.replace("ي", "ی").replace("ك", "ک")   # normalize forms
    text = re.sub(r"[؟،؛!,:؛]", " ", text)            # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -------------------------------
# 2️⃣  Prepare Chat Data (from TSV)
# -------------------------------
def prepare_chat_data(input_path="data/final_main_dataset.tsv"):
    """
    Read Urdu dataset (Common Voice style) and create chatbot-like input/response pairs.
    """
    print("Reading dataset...")
    df = pd.read_csv(input_path, sep="\t")

    # Remove rows without sentences
    df = df.dropna(subset=["sentence"])
    print(f"Loaded {len(df)} sentences")

    # Clean Urdu text
    df["sentence"] = df["sentence"].apply(normalize_urdu)

    # Create input–response pairs
    inputs = df["sentence"].iloc[:-1].tolist()
    responses = df["sentence"].iloc[1:].tolist()

    chat_df = pd.DataFrame({"input_text": inputs, "target_text": responses})
    print(f"Created {len(chat_df)} input-response pairs")

    # Train/Val/Test split
    train, temp = train_test_split(chat_df, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    os.makedirs("data/processed", exist_ok=True)
    train.to_csv("data/processed/train.csv", index=False)
    val.to_csv("data/processed/val.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)

    print("Train/Val/Test saved in data/processed/")
    return train


# -------------------------------
# 3️⃣  Build Vocabulary
# -------------------------------
def build_vocab_from_data(csv_path="data/processed/train.csv", min_freq=2, save_path="data/processed/vocab.json"):
    """
    Build word-level vocabulary from Urdu dataset.
    """
    print("Building vocabulary...")
    df = pd.read_csv(csv_path)
    counter = Counter()

    # Count words from both input and target text
    for col in ["input_text", "target_text"]:
        for text in df[col].dropna().tolist():
            words = text.split()
            counter.update(words)

    # Base vocab with special tokens
    vocab = {
        "<pad>": 0,
        "<sos>": 1,
        "<eos>": 2,
        "<unk>": 3
    }

    # Add words meeting minimum frequency
    for word, freq in counter.items():
        if freq >= min_freq and word not in vocab:
            vocab[word] = len(vocab)

    # Save vocab to JSON
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"Vocabulary built with {len(vocab)} tokens")
    print(f"Saved to: {save_path}")
    return vocab


# -------------------------------
# 4️⃣  Run Everything
# -------------------------------
if __name__ == "__main__":
    # Step 1: Prepare dataset
    train = prepare_chat_data()

    # Step 2: Build vocab
    build_vocab_from_data("data/processed/train.csv", min_freq=2, save_path="data/processed/vocab.json")

    print("\n🎉 Preprocessing pipeline complete!")
