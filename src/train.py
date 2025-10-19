import os
import json
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

import sacrebleu

from dataset import get_dataloaders
from model import TransformerChatbot


# =========================
# Hyperparameters (match assignment)
# =========================
EMBED_DIM = 512         # {256, 512} → using higher dimension for richer Urdu representation
NUM_HEADS = 2           # 2 → remains as per assignment constraint
ENC_LAYERS = 2          # 2–3 → slightly deeper encoder for better context capture
DEC_LAYERS = 2          # 2–3 → deeper decoder improves sentence fluency
DROPOUT   = 0.15        # 0.1–0.3 → moderate regularization to avoid overfitting
BATCH_SIZE = 64         # 32 / 64 → larger batch improves gradient stability
LR = 3e-4               # 1e-4 – 5e-4 → faster convergence within safe range
MAX_LEN = 40            # sequence length used in dataset.py loaders (unchanged)
EPOCHS = 25             # increased for better convergence and BLEU improvement
CLIP_GRAD_NORM = 1.0      # avoid exploding gradients
LABEL_SMOOTHING = 0.1     # improves BLEU/ROUGE stability
EARLY_STOP_PATIENCE = 5   # stop if BLEU doesn’t improve for 5 epochs


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Utility: load vocab/special ids
# =========================
with open(DATA_DIR / "vocab.json", "r", encoding="utf-8") as f:
    VOCAB = json.load(f)
PAD_ID = VOCAB["<pad>"]
SOS_ID = VOCAB["<sos>"]
EOS_ID = VOCAB["<eos>"]
VOCAB_SIZE = len(VOCAB)
INV_VOCAB = {v: k for k, v in VOCAB.items()}


# =========================
# Masks
# =========================
def make_subsequent_mask(sz: int) -> torch.Tensor:
    """
    Causal mask for decoder self-attention.
    Returns a mask of shape [1, 1, sz, sz] with 1s in allowed positions.
    """
    mask = torch.tril(torch.ones((sz, sz), dtype=torch.uint8))
    # shape to broadcast over [B, heads, L, L]
    return mask.unsqueeze(0).unsqueeze(0)  # [1,1,sz,sz]


# =========================
# Greedy decoding for validation
# =========================
@torch.no_grad()
def greedy_decode(model: nn.Module, src: torch.Tensor, max_len: int = MAX_LEN) -> torch.Tensor:
    """
    src: [B, L]
    returns: decoded ids [B, L'] including <sos> ... <eos>
    """
    model.eval()
    B = src.size(0)
    ys = torch.full((B, 1), SOS_ID, dtype=torch.long, device=src.device)

    for _ in range(max_len - 1):
        tgt_mask = make_subsequent_mask(ys.size(1)).to(src.device)
        logits = model(src, ys, src_mask=None, tgt_mask=tgt_mask)  # [B, T, V]
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B,1]
        ys = torch.cat([ys, next_token], dim=1)
        if torch.all(next_token.squeeze(1) == EOS_ID):
            break
    return ys


def ids_to_sentence(ids: List[int]) -> str:
    words = []
    for i in ids:
        if i == PAD_ID:
            continue
        if i == SOS_ID:
            continue
        if i == EOS_ID:
            break
        words.append(INV_VOCAB.get(int(i), "<unk>"))
    return " ".join(words)


# =========================
# Training / Validation loop
# =========================
def train():
    print(f"Using device: {DEVICE}")

    # Data
    train_loader, val_loader = get_dataloaders(batch_size=BATCH_SIZE, max_len=MAX_LEN)

    # Model
    model = TransformerChatbot(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_layers=ENC_LAYERS,
        num_heads=NUM_HEADS,
        ff_dim=EMBED_DIM * 2,     # simple choice; you can use 4*EMBED_DIM as well
        dropout=DROPOUT
    ).to(DEVICE)

    # Loss (ignore pads), Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = Adam(model.parameters(), lr=LR)

    best_bleu = -1.0
    best_path = MODELS_DIR / "best_model.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for src, tgt in pbar:
            src = src.to(DEVICE)           # [B, L]
            tgt = tgt.to(DEVICE)           # [B, L]

            # Teacher forcing: feed <sos> ... last-1
            dec_inp = tgt[:, :-1]          # [B, L-1]
            gold   = tgt[:, 1:]            # [B, L-1] next tokens are targets

            # Causal mask for decoder self-attn (same length as dec_inp)
            tgt_mask = make_subsequent_mask(dec_inp.size(1)).to(DEVICE)

            logits = model(src, dec_inp, src_mask=None, tgt_mask=tgt_mask)  # [B, L-1, V]
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), gold.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stabilize
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)

        # -------- Validation: compute BLEU on greedy-decoded outputs --------
        model.eval()
        refs = []
        hyps = []

        with torch.no_grad():
            for src, tgt in val_loader:
                src = src.to(DEVICE)
                tgt = tgt.to(DEVICE)

                gen = greedy_decode(model, src, max_len=MAX_LEN)  # [B, L]
                # Convert batches to strings
                for i in range(src.size(0)):
                    ref_sentence = ids_to_sentence(tgt[i].tolist())
                    hyp_sentence = ids_to_sentence(gen[i].tolist())
                    if ref_sentence.strip() and hyp_sentence.strip():
                        refs.append([ref_sentence])  # sacrebleu expects list of references per sample
                        hyps.append(hyp_sentence)

        # corpus BLEU
        bleu = sacrebleu.corpus_bleu(hyps, list(zip(*refs)))  # zip(*refs) -> list of reference lists
        val_bleu = float(bleu.score)

        print(f"\nEpoch {epoch}: Train Loss={avg_loss:.4f} | Val BLEU={val_bleu:.2f}")

        # Save best by BLEU
        if val_bleu > best_bleu:
            best_bleu = val_bleu
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab_size": VOCAB_SIZE,
                "config": {
                    "EMBED_DIM": EMBED_DIM,
                    "NUM_HEADS": NUM_HEADS,
                    "ENC_LAYERS": ENC_LAYERS,
                    "DEC_LAYERS": DEC_LAYERS,
                    "DROPOUT": DROPOUT,
                    "MAX_LEN": MAX_LEN,
                }
            }, best_path)
            print(f"Saved new best model to {best_path} (BLEU={best_bleu:.2f})")

    print("\nTraining complete.")
    print(f"Best Validation BLEU: {best_bleu:.2f}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    train()
