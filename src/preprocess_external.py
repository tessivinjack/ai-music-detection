import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import librosa

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}

def list_audio_files(root: Path) -> list[Path]:
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            files.append(p)
    return files

def audio_to_melspec(
    audio_path: Path,
    sr: int,
    duration: float,
    n_mels: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    # Load audio (mono) and clip/pad to duration
    y, _sr = librosa.load(str(audio_path), sr=sr, mono=True, duration=duration)
    target_len = int(sr * duration)

    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    mels = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    mels_db = librosa.power_to_db(mels, ref=np.max).astype(np.float32)  # [n_mels, time]
    return mels_db

def main():
    parser = argparse.ArgumentParser(description="Preprocess external fake generators into mel spectrogram .npy files.")
    parser.add_argument("--external_root", type=str, default="data/raw/external_fake")
    parser.add_argument("--out_dir", type=str, default="data/external/processed")
    parser.add_argument("--out_manifest", type=str, default="data/external/manifest_processed.csv")
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    external_root = Path(args.external_root)
    out_dir = Path(args.out_dir)
    out_manifest = Path(args.out_manifest)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    # Each subfolder is a generator
    generators = [p for p in external_root.iterdir() if p.is_dir()]
    if not generators:
        raise FileNotFoundError(f"No generator folders found under: {external_root}")

    rows = []
    idx = 0

    for gen_dir in sorted(generators):
        gen_name = gen_dir.name
        files = list_audio_files(gen_dir)

        if len(files) == 0:
            continue

        for f in sorted(files):
            try:
                mels_db = audio_to_melspec(
                    f,
                    sr=args.sr,
                    duration=args.duration,
                    n_mels=args.n_mels,
                    n_fft=args.n_fft,
                    hop_length=args.hop_length,
                )
            except Exception as e:
                print(f"SKIP (failed load/featurize): {f} :: {e}")
                continue

            out_path = out_dir / f"external_{gen_name}_{idx:06d}.npy"
            np.save(out_path, mels_db)

            rows.append({
                "feature_path": str(out_path.resolve()),
                "label": 1,  # external are fake
                "source": f"external_{gen_name}",
            })
            idx += 1

    df = pd.DataFrame(rows)
    df.to_csv(out_manifest, index=False)

    print(f"Wrote external processed manifest: {out_manifest} ({len(df)} rows)")
    print(f"Processed features in: {out_dir}")

if __name__ == "__main__":
    main()