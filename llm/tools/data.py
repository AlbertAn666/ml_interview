import os
import urllib.request
import torch

def ensure_data(path: str, url: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        return
    print(f"Downloading dataset to {path} ...")
    urllib.request.urlretrieve(url, path)
    print("Done.")

def build_char_dataset(text: str, device: torch.device, block_size: int, split_ratio: float = 0.9):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    def encode(s: str):
        return [stoi[c] for c in s]

    def decode(ids):
        return "".join([itos[int(i)] for i in ids])

    data = torch.tensor(encode(text), dtype=torch.long, device=device)
    n = int(split_ratio * len(data))
    train_data = data[:n]
    val_data = data[n:]

    def get_batch(split: str, batch_size: int):
        source = train_data if split == "train" else val_data
        # random start positions
        ix = torch.randint(0, len(source) - block_size - 1, (batch_size,), device=device)
        x = torch.stack([source[i:i + block_size] for i in ix])
        y = torch.stack([source[i + 1:i + block_size + 1] for i in ix])
        return x, y

    return {
        "stoi": stoi,
        "itos": itos,
        "encode": encode,
        "decode": decode,
        "get_batch": get_batch,
        "vocab_size": len(chars),
        "train_len": len(train_data),
        "val_len": len(val_data),
    }
