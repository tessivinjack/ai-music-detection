import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset

from src.dataset import MelNpyDataset
from src.train_baseline import SimpleCNN

def main():
    parser = argparse.ArgumentParser(description="Fine-tune baseline CNN by adding external fake generators.")
    parser.add_argument("--base_train_csv", type=str, default="data/interim/splits/train.csv")
    parser.add_argument("--external_train_csv", type=str, default="data/external/splits/external_train.csv")
    parser.add_argument("--init_model_path", type=str, default="results/baseline_cnn.pt")
    parser.add_argument("--out_model_path", type=str, default="results/baseline_cnn_finetuned.pt")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_ds = MelNpyDataset(args.base_train_csv)
    ext_ds = MelNpyDataset(args.external_train_csv)

    # Combine: keep real examples present, add extra fake examples
    train_ds = ConcatDataset([base_ds, ext_ds])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = SimpleCNN().to(device)
    state = torch.load(args.init_model_path, map_location=device)
    model.load_state_dict(state)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        row = {"epoch": epoch, "loss": float(np.mean(losses))}
        history.append(row)
        print(row)

    torch.save(model.state_dict(), args.out_model_path)
    with open("results/finetune_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nSaved fine-tuned model: {args.out_model_path}")
    print("Saved: results/finetune_history.json")

if __name__ == "__main__":
    main()