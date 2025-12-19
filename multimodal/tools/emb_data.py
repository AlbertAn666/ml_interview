import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ClipPairEmbeddingDataset(Dataset):
    """
    Each item: x = concat([img_emb, txt_emb]) -> float32 tensor shape (2D,)
    """
    def __init__(self, img_npy: str, txt_npy: str, normalize: bool = True):
        assert os.path.exists(img_npy), f"Missing {img_npy}"
        assert os.path.exists(txt_npy), f"Missing {txt_npy}"

        self.img = np.load(img_npy).astype(np.float32)  # (N, D)
        self.txt = np.load(txt_npy).astype(np.float32)  # (N, D)
        assert self.img.shape == self.txt.shape, f"Shape mismatch: {self.img.shape} vs {self.txt.shape}"

        if normalize:
            # CLIP embeddings are often already L2-normalized, but we keep it safe.
            self.img = self.img / (np.linalg.norm(self.img, axis=1, keepdims=True) + 1e-12)
            self.txt = self.txt / (np.linalg.norm(self.txt, axis=1, keepdims=True) + 1e-12)

        self.x = np.concatenate([self.img, self.txt], axis=1)  # (N, 2D)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx])  # (2D,)


def make_loader(img_npy: str, txt_npy: str, batch_size: int, shuffle: bool, num_workers: int = 0):
    ds = ClipPairEmbeddingDataset(img_npy, txt_npy, normalize=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False)
    return ds, dl
