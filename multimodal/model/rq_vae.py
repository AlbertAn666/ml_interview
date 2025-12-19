from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RQVAEConfig:
    in_dim: int = 1024        # img(512)+txt(512)
    latent_dim: int = 256     # quantize dimension
    n_quantizers: int = 3     # 3 layers
    codebook_size: int = 256  # each layer 0..255
    hidden_dim: int = 512
    dropout: float = 0.0

    # losses
    beta_commit: float = 0.25 # commitment loss weight
    beta_codebook: float = 1.0 # codebook loss weight (often 1.0)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class VectorQuantizer(nn.Module):
    """
    Classic VQ layer (EMA not used here for simplicity).
    """
    def __init__(self, codebook_size: int, dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.codebook = nn.Embedding(codebook_size, dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z_e: torch.Tensor):
        """
        z_e: (B, D)
        return:
          z_q: (B, D) quantized
          codes: (B,) long in [0, K-1]
          vq_loss: scalar (codebook + commit)
          stats: dict
        """
        # compute squared distances to codebook entries:
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2 z·e
        B, D = z_e.shape
        z2 = (z_e ** 2).sum(dim=1, keepdim=True)  # (B,1)
        e2 = (self.codebook.weight ** 2).sum(dim=1).unsqueeze(0)  # (1,K)
        ze = z_e @ self.codebook.weight.t()  # (B,K)
        dist = z2 + e2 - 2 * ze

        codes = torch.argmin(dist, dim=1)  # (B,)
        z_q = self.codebook(codes)         # (B,D)

        return z_q, codes


class ResidualQuantizer(nn.Module):
    """
    RQ: quantize residuals sequentially with multiple codebooks.
    """
    def __init__(self, n_quantizers: int, codebook_size: int, dim: int, beta_commit: float, beta_codebook: float):
        super().__init__()
        self.nq = n_quantizers
        self.beta_commit = beta_commit
        self.beta_codebook = beta_codebook
        self.vqs = nn.ModuleList([VectorQuantizer(codebook_size, dim) for _ in range(n_quantizers)])

    def forward(self, z_e: torch.Tensor):
        """
        z_e: (B,D)
        returns:
          z_q_sum: (B,D)
          codes: (B, nq) long
          loss: scalar
          usage: list of (K,) counts (optional computed outside)
        """
        residual = z_e
        z_q_sum = torch.zeros_like(z_e)
        all_codes = []
        vq_losses = []

        for vq in self.vqs:
            z_q, codes = vq(residual)

            # VQ-VAE losses (no EMA):
            # codebook loss: ||sg[z_e] - z_q||^2
            # commit loss:   ||z_e - sg[z_q]||^2
            codebook_loss = F.mse_loss(z_q, residual.detach())
            commit_loss = F.mse_loss(residual, z_q.detach())

            vq_loss = self.beta_codebook * codebook_loss + self.beta_commit * commit_loss

            # straight-through estimator
            z_q_st = residual + (z_q - residual).detach()

            z_q_sum = z_q_sum + z_q_st
            residual = residual - z_q_st  # residual for next quantizer

            all_codes.append(codes)
            vq_losses.append(vq_loss)

        codes_mat = torch.stack(all_codes, dim=1)  # (B,nq)
        loss = torch.stack(vq_losses).sum()
        return z_q_sum, codes_mat, loss


class RQVAE(nn.Module):
    def __init__(self, cfg: RQVAEConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder = MLP(cfg.in_dim, cfg.hidden_dim, cfg.latent_dim, dropout=cfg.dropout)
        self.rq = ResidualQuantizer(
            n_quantizers=cfg.n_quantizers,
            codebook_size=cfg.codebook_size,
            dim=cfg.latent_dim,
            beta_commit=cfg.beta_commit,
            beta_codebook=cfg.beta_codebook,
        )
        self.decoder = MLP(cfg.latent_dim, cfg.hidden_dim, cfg.in_dim, dropout=cfg.dropout)

    def forward(self, x: torch.Tensor):
        """
        x: (B, in_dim)
        returns:
          x_hat: (B, in_dim)
          codes: (B, 3) long
          loss: scalar (recon + vq)
          loss_dict: dict
        """
        z_e = self.encoder(x)                      # (B, latent_dim)
        z_q, codes, vq_loss = self.rq(z_e)         # (B, latent_dim), (B,3)
        x_hat = self.decoder(z_q)                  # (B, in_dim)

        recon_loss = F.mse_loss(x_hat, x)
        loss = recon_loss + vq_loss

        return x_hat, codes, loss, {
            "recon_loss": recon_loss.detach(),
            "vq_loss": vq_loss.detach(),
        }

    @torch.no_grad()
    def encode_to_codes(self, x: torch.Tensor):
        """
        x: (B, in_dim) -> codes: (B, 3)
        """
        z_e = self.encoder(x)
        _, codes, _ = self.rq(z_e)
        return codes
