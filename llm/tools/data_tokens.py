import torch
from datasets import load_dataset

def load_wikitext2_text():
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_texts = [x["text"] for x in ds["train"] if x["text"].strip()]
    val_texts   = [x["text"] for x in ds["validation"] if x["text"].strip()]
    return train_texts, val_texts

def build_token_dataset(tokenizer, device: torch.device, block_size: int, batch_size: int):
    train_texts, val_texts = load_wikitext2_text()

    train_ids = []
    for t in train_texts:
        train_ids.extend(tokenizer.encode(t).ids + [tokenizer.token_to_id("<eos>")])
    val_ids = []
    for t in val_texts:
        val_ids.extend(tokenizer.encode(t).ids + [tokenizer.token_to_id("<eos>")])

    train_data = torch.tensor(train_ids, dtype=torch.long, device=device)
    val_data   = torch.tensor(val_ids, dtype=torch.long, device=device)

    def get_batch(split: str, bs: int = batch_size):
        source = train_data if split == "train" else val_data
        ix = torch.randint(0, len(source) - block_size - 1, (bs,), device=device)
        x = torch.stack([source[i:i + block_size] for i in ix])
        y = torch.stack([source[i + 1:i + block_size + 1] for i in ix])
        return x, y

    return {
        "get_batch": get_batch,
        "train_len": int(train_data.numel()),
        "val_len": int(val_data.numel()),
        "vocab_size": tokenizer.get_vocab_size(),
    }
