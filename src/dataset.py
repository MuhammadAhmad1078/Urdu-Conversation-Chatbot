import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import os


# -------------------------------
# 1️⃣ Urdu Chat Dataset Class
# -------------------------------
class UrduChatDataset(Dataset):
    def __init__(self, csv_path, vocab_path, max_len=50):
        """
        Args:
            csv_path: path to train/val/test CSV
            vocab_path: path to vocab.json
            max_len: max token length for each sentence
        """
        assert os.path.exists(csv_path), f"Dataset not found: {csv_path}"
        assert os.path.exists(vocab_path), f"Vocab file not found: {vocab_path}"

        self.df = pd.read_csv(csv_path)
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.max_len = max_len

    def encode_sentence(self, text):
        """
        Convert Urdu text into token IDs.
        """
        tokens = str(text).split()
        ids = [self.vocab.get(w, self.vocab["<unk>"]) for w in tokens]
        ids = [self.vocab["<sos>"]] + ids[:self.max_len - 2] + [self.vocab["<eos>"]]
        pad_len = self.max_len - len(ids)
        ids += [self.vocab["<pad>"]] * pad_len
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src = self.encode_sentence(row["input_text"])
        tgt = self.encode_sentence(row["target_text"])
        return src, tgt


# -------------------------------
# 2️⃣ Helper Function to Load Data
# -------------------------------
def get_dataloaders(batch_size=32, max_len=50):
    """
    Build train and validation dataloaders.
    """
    train_csv = "data/processed/train.csv"
    val_csv = "data/processed/val.csv"
    vocab_path = "data/processed/vocab.json"

    train_ds = UrduChatDataset(train_csv, vocab_path, max_len)
    val_ds = UrduChatDataset(val_csv, vocab_path, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")
    return train_loader, val_loader


# -------------------------------
# 3️⃣ Quick Sanity Check
# -------------------------------
if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders(batch_size=4, max_len=40)
    for src, tgt in train_loader:
        print("Source batch:", src.shape)
        print("Target batch:", tgt.shape)
        print("Example src[0]:", src[0][:10])  # print first 10 tokens
        break
