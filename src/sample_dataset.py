import argparse
import random
import shutil
from pathlib import Path

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}

def list_audio_files(root: Path) -> list[Path]:
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            files.append(p)
    return files

def copy_subset(files: list[Path], dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dest_dir / f.name)

def main():
    parser = argparse.ArgumentParser(description="Sample subsets of FMA real and SONICs fake into data/raw/")
    parser.add_argument("--fma_source", type=str, required=True, help="Path to full FMA dataset root")
    parser.add_argument("--sonics_source", type=str, required=True, help="Path to folder containing SONICs fake mp3s")
    parser.add_argument("--out_real", type=str, default="data/raw/fma_real", help="Output dir for sampled real audio")
    parser.add_argument("--out_fake", type=str, default="data/raw/sonics_fake", help="Output dir for sampled fake audio")
    parser.add_argument("--n_real", type=int, default=150)
    parser.add_argument("--n_fake", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    fma_root = Path(args.fma_source)
    sonics_root = Path(args.sonics_source)
    out_real = Path(args.out_real)
    out_fake = Path(args.out_fake)

    fma_files = list_audio_files(fma_root)
    if len(fma_files) < args.n_real:
        raise ValueError(f"Not enough FMA files found ({len(fma_files)}) for n_real={args.n_real}")

    sonics_files = list_audio_files(sonics_root)
    if len(sonics_files) < args.n_fake:
        raise ValueError(f"Not enough SONICs files found ({len(sonics_files)}) for n_fake={args.n_fake}")

    real_subset = random.sample(fma_files, args.n_real)
    fake_subset = random.sample(sonics_files, args.n_fake)

    # Clear output dirs to keep runs deterministic
    if out_real.exists():
        shutil.rmtree(out_real)
    if out_fake.exists():
        shutil.rmtree(out_fake)

    copy_subset(real_subset, out_real)
    copy_subset(fake_subset, out_fake)

    print(f"Sampled real: {args.n_real} -> {out_real}")
    print(f"Sampled fake: {args.n_fake} -> {out_fake}")
    print(f"Seed: {args.seed}")

if __name__ == "__main__":
    main()
