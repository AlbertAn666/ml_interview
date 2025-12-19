import math
import torch
import torch.nn.functional as F

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

@torch.no_grad()
def estimate_loss(model, get_batch_fn, device, eval_iters: int, batch_size: int):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            xb, yb = get_batch_fn(split, batch_size)
            _, loss = model(xb, yb)
            losses[k] = loss
        out[split] = losses.mean().item()
    model.train()
    return out

def top_k_logits(logits: torch.Tensor, k: int):
    if k <= 0:
        return logits
    v, _ = torch.topk(logits, k)
    thresh = v[:, [-1]]
    return torch.where(logits < thresh, torch.full_like(logits, float("-inf")), logits)

@torch.no_grad()
def generate(model, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int = 0):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-8)
        logits = top_k_logits(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_idx], dim=1)
    model.train()
    return idx
