import torch
import json
from model import TransformerChatbot
from train import make_subsequent_mask, ids_to_sentence
from preprocess import normalize_urdu

# -------------------------------
# Load Model
# -------------------------------
def load_model(model_path="models/best_model.pt", vocab_path="data/processed/vocab.json"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    inv_vocab = {v: k for k, v in vocab.items()}
    ckpt = torch.load(model_path, map_location=device)
    config = ckpt["config"]

    model = TransformerChatbot(
        vocab_size=len(vocab),
        embed_dim=config["EMBED_DIM"],
        num_layers=config["ENC_LAYERS"],
        num_heads=config["NUM_HEADS"],
        ff_dim=config["EMBED_DIM"] * 2,
        dropout=config["DROPOUT"]
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, vocab, inv_vocab, device


# -------------------------------
# Manual Greedy Decoding
# -------------------------------
def greedy_decode(model, src, vocab, inv_vocab, device, max_len=40):
    model.eval()
    src = src.to(device)

    # prepare initial decoder input (<sos> token or first token)
    sos_idx = list(vocab.values())[0]  # first token as start (or define manually)
    eos_idx = list(vocab.values())[-1]  # last token as end (optional)

    dec_input = torch.tensor([[sos_idx]], dtype=torch.long).to(device)

    for _ in range(max_len):
        with torch.no_grad():
            output = model(src, dec_input)
            next_token = output[:, -1, :].argmax(-1).unsqueeze(1)
        dec_input = torch.cat([dec_input, next_token], dim=1)
        if next_token.item() == eos_idx:
            break

    decoded = "".join([inv_vocab.get(int(tok), "") for tok in dec_input[0]])
    return decoded


# -------------------------------
# Generate Reply
# -------------------------------
def generate_reply(text, model, vocab, inv_vocab, device, max_len=40):
    text = normalize_urdu(text)

    src = torch.tensor([[vocab.get(c, 0) for c in text]], dtype=torch.long).to(device)
    reply = greedy_decode(model, src, vocab, inv_vocab, device, max_len)
    return reply


# -------------------------------
# Quick Test (for console)
# -------------------------------
if __name__ == "__main__":
    model, vocab, inv_vocab, device = load_model()
    while True:
        user_input = input("👤 User (Urdu): ")
        if user_input.lower() in ["exit", "quit"]:
            break
        reply = generate_reply(user_input, model, vocab, inv_vocab, device)
        print(f"🤖 Bot: {reply}\n")
