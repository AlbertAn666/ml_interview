# softmax_fashion_mnist.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.datasets import FashionMNIST
from torchvision import transforms


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SoftmaxRegression(nn.Module):
    """Multiclass linear classifier: logits = W x + b (softmax happens in loss)."""
    def __init__(self, in_dim: int = 784, num_classes: int = 10):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28) -> (B, 784)
        x = x.view(x.size(0), -1)
        return self.linear(x)  # logits: (B, 10)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += float(loss.item())

        preds = torch.argmax(logits, dim=1)
        correct += int((preds == yb).sum().item())
        total += yb.numel()

    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
    }


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = get_device()
    print("Using device:", device)

    # Normalize: Fashion-MNIST is in [0,1] after ToTensor, normalize with common mean/std
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),  # widely-used Fashion-MNIST stats
    ])

    data_dir = os.path.join(".", "data")
    train_ds = FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_ds = FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0, pin_memory=False)

    model = SoftmaxRegression(in_dim=784, num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4)

    epochs = 10
    best_test_acc = 0.0

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
        test_metrics = evaluate(model, test_loader, device)

        if test_metrics["acc"] > best_test_acc:
            best_test_acc = test_metrics["acc"]

        print(
            f"Epoch {epoch:2d} | "
            f"train loss {train_loss:.4f} acc {train_metrics['acc']:.4f} | "
            f"test loss {test_metrics['loss']:.4f} acc {test_metrics['acc']:.4f} | "
            f"best test acc {best_test_acc:.4f}"
        )

    # Show a few learned weights (optional sanity check)
    W = model.linear.weight.detach().cpu().numpy()  # (10, 784)
    b = model.linear.bias.detach().cpu().numpy()    # (10,)
    print("\nParams snapshot:")
    print("  W shape:", W.shape, "b shape:", b.shape)
    print("  W[0, :5]:", np.round(W[0, :5], 4))
    print("  b[:5]:", np.round(b[:5], 4))


if __name__ == "__main__":
    main()
