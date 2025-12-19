# main_wikitext2.py
import os
import time
import torch

from tools.utils import get_device, estimate_loss, generate
from tools.tokenizer_bpe import train_or_load_tokenizer, decode
from tools.data_tokens import load_wikitext2_text, build_token_dataset
from model import GPTConfig, GPT

TOKENIZER_PATH = os.path.join("data", "wikitext2_tokenizer.json")
CKPT_PATH = os.path.join("checkpoints", "gpt_wikitext2.pt")

BLOCK_SIZE = 256
BATCH_SIZE = 48
MAX_STEPS = 5000
EVAL_INTERVAL = 300
EVAL_ITERS = 20

LR = 3e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

GEN_LEN = 200
TEMPERATURE = 0.9
TOP_K = 50


def save_checkpoint(path, model, cfg: GPTConfig, tokenizer_json_path: str, step: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "step": step,
        "config": cfg.__dict__,
        "model_state_dict": model.state_dict(),
        "tokenizer_path": tokenizer_json_path,
    }
    torch.save(payload, path)
    print(f"[saved] {path} (step={step})")


def main():
    device = get_device()
    print("Device:", device)

    train_texts, _ = load_wikitext2_text()
    tok = train_or_load_tokenizer(train_texts, TOKENIZER_PATH, vocab_size=8000)
    vocab_size = tok.get_vocab_size()
    print("Tokenizer vocab_size:", vocab_size)

    ds = build_token_dataset(tok, device=device, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE)
    print("Token dataset lens:", ds["train_len"], ds["val_len"])

    cfg = GPTConfig(
        block_size=BLOCK_SIZE,
        n_layers=6,
        n_heads=6,
        d_model=384,
        dropout=0.1,
        vocab_size=vocab_size,
    )
    model = GPT(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    t0 = time.time()
    for step in range(1, MAX_STEPS + 1):
        xb, yb = ds["get_batch"]("train", BATCH_SIZE)
        _, loss = model(xb, yb)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        if GRAD_CLIP and GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optim.step()

        if step % EVAL_INTERVAL == 0 or step == 1:
            losses = estimate_loss(model, ds["get_batch"], device, EVAL_ITERS, BATCH_SIZE)
            dt = time.time() - t0
            print(f"step {step:5d}/{MAX_STEPS} | train {losses['train']:.4f} | val {losses['val']:.4f} | {dt:.1f}s")

            bos_id = tok.token_to_id("<bos>")
            idx0 = torch.tensor([[bos_id]], dtype=torch.long, device=device)
            out = generate(model, idx0, max_new_tokens=GEN_LEN, temperature=TEMPERATURE, top_k=TOP_K)[0].tolist()
            print("---- sample ----")
            print(decode(tok, out))
            print("---------------\n")

            save_checkpoint(CKPT_PATH, model, cfg, TOKENIZER_PATH, step)

    save_checkpoint(CKPT_PATH, model, cfg, TOKENIZER_PATH, MAX_STEPS)
    print("Training finished.")


if __name__ == "__main__":
    torch.manual_seed(42)
    main()
