import argparse
import numpy as np
import torch

from model.rqvae import RQVAE, RQVAEConfig


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/rqvae_clip_coco.pt")
    ap.add_argument("--img_npy", default="data/clip_emb/img_emb_val2017.npy")
    ap.add_argument("--txt_npy", default="data/clip_emb/txt_emb_val2017.npy")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device)

    cfg = RQVAEConfig(**ckpt["config"])
    model = RQVAE(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    img = np.load(args.img_npy).astype(np.float32)
    txt = np.load(args.txt_npy).astype(np.float32)
    x = np.concatenate([img, txt], axis=1)[: args.n]  # (n, 1024)

    # normalize like training
    D = img.shape[1]
    x[:, :D] = x[:, :D] / (np.linalg.norm(x[:, :D], axis=1, keepdims=True) + 1e-12)
    x[:, D:] = x[:, D:] / (np.linalg.norm(x[:, D:], axis=1, keepdims=True) + 1e-12)

    xt = torch.from_numpy(x).to(device)
    with torch.no_grad():
        codes = model.encode_to_codes(xt).detach().cpu().numpy().astype(np.uint8)

    print("codes (n,3) uint8:")
    print(codes)


if __name__ == "__main__":
    main()
