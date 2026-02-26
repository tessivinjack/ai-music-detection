import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa

from src.config import ensure_dirs, Paths

def audio_to_mel(path: str, sr: int, duration: float, n_mels: int) -> np.ndarray:
    y, _sr = librosa.load(path, sr=sr, mono=True, duration=duration)
    if y.size == 0:
        raise ValueError("Empty audio")
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Normalize to roughly [-1, 1]
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
    return mel_db.astype(np.float32)

def main():
    parser = argparse.ArgumentParser(description="Preprocess audio into mel spectrogram arrays.")
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--n_mels", type=int, default=128)
    args = parser.parse_args()

    paths = ensure_dirs()
    if not paths.manifest_csv.exists():
        raise FileNotFoundError(f"Manifest not found: {paths.manifest_csv}. Run make_manifest first.")

    df = pd.read_csv(paths.manifest_csv)
    out_rows = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = row["path"]
        label = int(row["label"])
        source = row["source"]

        out_name = f"{source}_{i:06d}.npy"
        out_path = paths.data_processed / out_name

        try:
            mel = audio_to_mel(audio_path, sr=args.sr, duration=args.duration, n_mels=args.n_mels)
            np.save(out_path, mel)
            out_rows.append({"feature_path": str(out_path), "label": label, "source": source})
        except Exception:
            # skip unreadable files, but keep pipeline moving
            continue

    out_df = pd.DataFrame(out_rows)
    processed_manifest = paths.data_interim / "manifest_processed.csv"
    out_df.to_csv(processed_manifest, index=False)
    print(f"Wrote processed manifest: {processed_manifest} ({len(out_df)} rows)")

if __name__ == "__main__":
    main()
