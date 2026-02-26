import argparse
from pathlib import Path
import pandas as pd

from src.config import ensure_dirs, Paths

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}

def list_audio_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            files.append(p)
    return files

def main():
    parser = argparse.ArgumentParser(description="Build a manifest CSV for real vs fake audio.")
    parser.add_argument("--max_fake", type=int, default=200)
    parser.add_argument("--max_real", type=int, default=200)
    args = parser.parse_args()

    paths = ensure_dirs()
    fake_files = list_audio_files(paths.sonics_fake_dir)[: args.max_fake]
    real_files = list_audio_files(paths.fma_real_dir)[: args.max_real]

    if len(fake_files) == 0:
        raise FileNotFoundError(f"No fake audio found under: {paths.sonics_fake_dir}")
    if len(real_files) == 0:
        raise FileNotFoundError(f"No real audio found under: {paths.fma_real_dir}")

    rows = []
    for f in fake_files:
        rows.append({"path": str(f), "label": 1, "source": "sonics_fake"})
    for f in real_files:
        rows.append({"path": str(f), "label": 0, "source": "fma_real"})

    df = pd.DataFrame(rows).sample(frac=1.0, random_state=42).reset_index(drop=True)
    df.to_csv(paths.manifest_csv, index=False)
    print(f"Wrote manifest: {paths.manifest_csv} ({len(df)} rows)")

if __name__ == "__main__":
    main()
