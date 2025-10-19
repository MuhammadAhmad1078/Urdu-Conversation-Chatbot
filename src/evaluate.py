import torch
import torch.nn as nn
import pandas as pd
import json
from tqdm import tqdm
import math

import sacrebleu
from rouge_score import rouge_scorer

from model import TransformerChatbot
from dataset import UrduChatDataset
from train import make_subsequent_mask, ids_to_sentence, greedy_decode


# -------------------------------
# Load model & vocab
# -------------------------------
def load_model(model_path="models/best_model.pt", vocab_path="data/processed/vocab.json"):
    print("🔄 Loading model and vocabulary...")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    inv_vocab = {v: k for k, v in vocab.items()}

    ckpt = torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    config = ckpt["config"]

    model = TransformerChatbot(
        vocab_size=len(vocab),
        embed_dim=config["EMBED_DIM"],
        num_layers=config["ENC_LAYERS"],
        num_heads=config["NUM_HEADS"],
        ff_dim=config["EMBED_DIM"] * 2,
        dropout=config["DROPOUT"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    return model, vocab, inv_vocab


# -------------------------------
# Compute metrics
# -------------------------------
def compute_metrics(model, csv_path="data/processed/test.csv", vocab=None, inv_vocab=None, max_len=40):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(csv_path).dropna()

    PAD_ID = vocab["<pad>"]
    SOS_ID = vocab["<sos>"]
    EOS_ID = vocab["<eos>"]

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction="sum")
    total_loss = 0.0
    total_tokens = 0

    refs, hyps = [], []

    print("📊 Generating predictions and calculating metrics...")
    for i in tqdm(range(len(df))):
        src_text = str(df.iloc[i]["input_text"])
        tgt_text = str(df.iloc[i]["target_text"])

        # Encode manually
        src_ids = [SOS_ID] + [vocab.get(tok, vocab["<unk>"]) for tok in src_text.split()] + [EOS_ID]
        src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)

        # Greedy decode
        with torch.no_grad():
            pred = greedy_decode(model, src, max_len=max_len)
            logits = model(src, pred[:, :-1])
            loss = criterion(logits.reshape(-1, len(vocab)), pred[:, 1:].reshape(-1))
            total_loss += loss.item()
            total_tokens += (pred[:, 1:] != PAD_ID).sum().item()

        hyp_sentence = ids_to_sentence(pred[0].tolist())
        refs.append([tgt_text])
        hyps.append(hyp_sentence)

    # BLEU
    bleu = sacrebleu.corpus_bleu(hyps, list(zip(*refs)))

    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    rouge_scores = [scorer.score(ref[0], hyp)['rougeL'].fmeasure for hyp, ref in zip(hyps, refs)]
    rouge_l = sum(rouge_scores) / len(rouge_scores)

    # chrF
    chrf = sacrebleu.corpus_chrf(hyps, list(zip(*refs)))

    # Perplexity
    ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")

    print("\n📈 Evaluation Metrics:")
    print(f"BLEU Score     : {bleu.score:.3f}")
    print(f"ROUGE-L (F1)   : {rouge_l:.3f}")
    print(f"chrF Score     : {chrf.score:.3f}")
    print(f"Perplexity (↓) : {ppl:.3f}")

    return {
        "BLEU": bleu.score,
        "ROUGE-L": rouge_l,
        "chrF": chrf.score,
        "Perplexity": ppl,
    }


if __name__ == "__main__":
    model, vocab, inv_vocab = load_model()
    metrics = compute_metrics(model, "data/processed/test.csv", vocab, inv_vocab)
    print("\n✅ Final Metrics Summary:", metrics)
