# train_logreg_bc.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class LogisticRegression(nn.Module):
    """Binary logistic regression: y = sigmoid(w^T x + b)."""
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)  # (batch,)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0

    criterion = nn.BCEWithLogitsLoss(reduction="sum")

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += float(loss.item())

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        correct += int((preds == yb).sum().item())
        total += yb.numel()

    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
    }


def main():
    # 1) Load data
    X, y = load_breast_cancer(return_X_y=True)
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    # 2) Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3) Standardize features (fit on train only!)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    # 4) Torch datasets/loaders
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    # 5) Model/optim
    device = get_device()
    model = LogisticRegression(in_dim=X_train.shape[1]).to(device)

    # numerically stable binary cross-entropy: input logits + target in {0,1}
    criterion = nn.BCEWithLogitsLoss()

    # weight_decay = L2 norm regularization
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, weight_decay=1e-4)

    # 6) Train loop
    epochs = 200
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = yb.size(0)
            running_loss += float(loss.item()) * bs
            n += bs

        train_loss = running_loss / max(n, 1)
        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:3d} | "
                f"train loss {train_loss:.4f} acc {train_metrics['acc']:.4f} | "
                f"val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.4f} | "
                f"best val acc {best_val_acc:.4f} | device {device}"
            )

    # 7) Final weights (optional)
    w = model.linear.weight.detach().cpu().numpy().reshape(-1)
    b = float(model.linear.bias.detach().cpu().item())
    print("\nLearned params:")
    print("  w[:5] =", np.round(w[:5], 4))
    print("  b     =", round(b, 4))


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
