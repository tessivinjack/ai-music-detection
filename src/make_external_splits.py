import argparse
import pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Create train/holdout splits for external generators by source.")
    parser.add_argument("--external_manifest", type=str, default="data/external/manifest_processed.csv")
    parser.add_argument("--holdout_sources", nargs="+", default=["external_mureka", "external_tad_ai"])
    parser.add_argument("--out_dir", type=str, default="data/external/splits")
    args = parser.parse_args()

    df = pd.read_csv(args.external_manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    holdout = df[df["source"].isin(args.holdout_sources)].copy()
    train = df[~df["source"].isin(args.holdout_sources)].copy()

    train_path = out_dir / "external_train.csv"
    holdout_path = out_dir / "external_holdout.csv"
    train.to_csv(train_path, index=False)
    holdout.to_csv(holdout_path, index=False)

    print(f"External train: {len(train)} -> {train_path}")
    print(f"External holdout: {len(holdout)} -> {holdout_path}")
    print("Holdout sources:", args.holdout_sources)

if __name__ == "__main__":
    main()