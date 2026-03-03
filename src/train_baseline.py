import argparse
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.config import ensure_dirs
from src.dataset import MelNpyDataset

class SimpleCNN(nn.Module):
    """
    Small CNN baseline for mel spectrograms.
    Input: [B, 1, n_mels, time]
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # -> [B, 64, 1, 1]
            nn.Flatten(),                  # -> [B, 64]
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, preds = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        preds.extend(pred)
        ys.extend(y.numpy().tolist())
    return accuracy_score(ys, preds)

def main():
    print("Starting training...")
    parser = argparse.ArgumentParser(description="Train a baseline CNN on precomputed mel spectrograms.")
    parser.add_argument("--manifest_processed", type=str, default="data/interim/manifest_processed.csv")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_size", type=float, default=0.2)
    args = parser.parse_args()

    paths = ensure_dirs()
    set_seed(args.seed)

    df = pd.read_csv(args.manifest_processed)
    if len(df) < 20:
        raise ValueError("Not enough processed samples. Need at least ~20 to train/eval.")

    # Stratified split so class balance is preserved
    train_df, val_df = train_test_split(
        df,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=df["label"]
    )

    split_dir = paths.data_interim / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    train_csv = split_dir / "train.csv"
    val_csv = split_dir / "val.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    train_ds = MelNpyDataset(str(train_csv))
    val_ds = MelNpyDataset(str(val_csv))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            epoch_losses.append(loss.item())

        train_acc = evaluate(model, train_loader, device)
        val_acc = evaluate(model, val_loader, device)

        row = {
            "epoch": epoch,
            "loss": float(np.mean(epoch_losses)) if epoch_losses else None,
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
        }
        history.append(row)
        print(row)

    # Save artifacts
    model_path = paths.results / "baseline_cnn.pt"
    torch.save(model.state_dict(), model_path)

    metrics_path = paths.results / "train_history.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nSaved model: {model_path}")
    print(f"Saved training history: {metrics_path}")
    print(f"Saved splits: {train_csv} and {val_csv}")

if __name__ == "__main__":
    main()
