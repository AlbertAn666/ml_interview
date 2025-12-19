import os
import time
import numpy as np
import torch

from tools.emb_data import make_loader
from model.rq_vae import RQVAE, RQVAEConfig


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def code_usage(model: RQVAE, loader, device):
    # returns usage counts per quantizer: list of arrays shape (K,)
    K = model.cfg.codebook_size
    nq = model.cfg.n_quantizers
    counts = [np.zeros((K,), dtype=np.int64) for _ in range(nq)]

    model.eval()
    for x in loader:
        x = x.to(device)
        codes = model.encode_to_codes(x)  # (B,nq)
        codes_np = codes.detach().cpu().numpy()
        for qi in range(nq):
            vals, cts = np.unique(codes_np[:, qi], return_counts=True)
            counts[qi][vals] += cts
    model.train()
    return counts


def save_ckpt(path, model, cfg: RQVAEConfig, step: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "step": step,
        "config": cfg.__dict__,
        "model_state_dict": model.state_dict(),
    }
    torch.save(payload, path)
    print(f"[saved] {path} (step={step})")


@torch.no_grad()
def dump_semantic_ids(model: RQVAE, loader, device, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    all_codes = []
    model.eval()
    for x in loader:
        x = x.to(device)
        codes = model.encode_to_codes(x)  # (B,3)
        all_codes.append(codes.detach().cpu().numpy().astype(np.uint8))
    model.train()
    arr = np.concatenate(all_codes, axis=0)  # (N,3) uint8
    np.save(out_path, arr)
    print(f"[dumped] {out_path} {arr.shape} dtype={arr.dtype}")


def main():
    device = get_device()
    print("Device:", device)

    # paths (after generating CLIP embeddings)
    emb_dir = "data/clip_emb"
    img_train = os.path.join(emb_dir, "img_emb_train2017.npy")
    txt_train = os.path.join(emb_dir, "txt_emb_train2017.npy")

    ds, dl = make_loader(img_train, txt_train, batch_size=1024, shuffle=True)

    _, dl_noshuf = make_loader(img_train, txt_train, batch_size=1024, shuffle=False)

    in_dim = ds.x.shape[1]  # 1024
    cfg = RQVAEConfig(
        in_dim=in_dim,
        latent_dim=256,
        n_quantizers=3,
        codebook_size=256,
        hidden_dim=512,
        dropout=0.0,
        beta_commit=0.25,
        beta_codebook=1.0,
    )
    model = RQVAE(cfg).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    ckpt_path = "checkpoints/rqvae_clip_coco.pt"
    ids_out = "data/semantic_ids/semantic_ids_train2017.npy"

    max_steps = 2000
    log_interval = 100
    eval_interval = 1000

    t0 = time.time()
    step = 0

    it = iter(dl)
    while step < max_steps:
        try:
            x = next(it)
        except StopIteration:
            it = iter(dl)
            x = next(it)

        x = x.to(device)
        _, _, loss, ld = model(x)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        step += 1

        if step == 1 or step % log_interval == 0:
            dt = time.time() - t0
            print(
                f"step {step:5d}/{max_steps} | "
                f"loss {loss.item():.6f} | recon {float(ld['recon_loss']):.6f} | vq {float(ld['vq_loss']):.6f} | "
                f"{dt:.1f}s"
            )

        if step % eval_interval == 0:
            save_ckpt(ckpt_path, model, cfg, step)

            counts = code_usage(model, dl_noshuf, device)
            for qi, c in enumerate(counts):
                used = int((c > 0).sum())
                ent = float((- (c / max(c.sum(), 1)) * np.log((c / max(c.sum(), 1)) + 1e-12)).sum())
                print(f"  Q{qi}: used {used}/256 | entropy {ent:.3f}")

    save_ckpt(ckpt_path, model, cfg, max_steps)

    # dump semantic ids aligned with meta_train2017.jsonl (because of dl_noshuf)
    dump_semantic_ids(model, dl_noshuf, device, ids_out)


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
