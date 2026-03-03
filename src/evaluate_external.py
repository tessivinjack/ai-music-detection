import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import MelNpyDataset
from src.train_baseline import SimpleCNN

@torch.no_grad()
def get_probs(model, loader, device):
    model.eval()
    probs = []
    sources = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # P(fake)
        probs.extend(p.tolist())
        # y not needed (all fake), but dataset doesn't return source
    return np.array(probs)

def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline model on external generators (fake-only).")
    parser.add_argument("--external_manifest", type=str, default="data/external/manifest_processed.csv")
    parser.add_argument("--model_path", type=str, default="results/baseline_cnn.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.50, 0.6222071, 0.7004964])
    parser.add_argument("--out_csv", type=str, default="results/external_eval_table.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.external_manifest)

    ds = MelNpyDataset(args.external_manifest)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)

    p_fake = get_probs(model, loader, device)

    # Attach probs back to df (same order as dataset)
    df = df.copy()
    df["p_fake"] = p_fake

    print("\nExternal generator score summary (mean p_fake):")
    print(df.groupby("source")["p_fake"].mean().sort_values(ascending=False))

    for t in args.thresholds:
        df[f"pred_fake_t{t:.3f}"] = (df["p_fake"] >= t).astype(int)

    print("\nExternal generator recall at thresholds (since all external are fake):")
    rows = []
    for src, g in df.groupby("source"):
        row = {"source": src, "n": len(g)}
        for t in args.thresholds:
            row[f"recall@{t:.3f}"] = float(g[f"pred_fake_t{t:.3f}"].mean())
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("source")
    print(out.to_string(index=False))

    # Save table for notebook
    out_path = args.out_csv
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    
if __name__ == "__main__":
    main()