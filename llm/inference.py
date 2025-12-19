import argparse
import torch
from tokenizers import Tokenizer

from tools.utils import get_device, generate
from tools.tokenizer_bpe import decode
from model import GPTConfig, GPT

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/gpt_wikitext2.pt")
    ap.add_argument("--prompt", default="New York is a city in ")
    ap.add_argument("--gen_len", type=int, default=200)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    args = ap.parse_args()

    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device)

    tok = Tokenizer.from_file(ckpt["tokenizer_path"])
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ids = tok.encode(args.prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    out = generate(model, idx, max_new_tokens=args.gen_len, temperature=args.temp, top_k=args.top_k)[0].tolist()
    print(decode(tok, out))

if __name__ == "__main__":
    main()
