from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    repo_root: Path = Path(__file__).resolve().parents[1]
    data_raw: Path = repo_root / "data" / "raw"
    data_interim: Path = repo_root / "data" / "interim"
    data_processed: Path = repo_root / "data" / "processed"
    results: Path = repo_root / "results"

    sonics_fake_dir: Path = data_raw / "sonics_fake"
    fma_real_dir: Path = data_raw / "fma_real"

    manifest_csv: Path = data_interim / "manifest.csv"

def ensure_dirs():
    p = Paths()
    p.data_interim.mkdir(parents=True, exist_ok=True)
    p.data_processed.mkdir(parents=True, exist_ok=True)
    p.results.mkdir(parents=True, exist_ok=True)
    (p.results / "figures").mkdir(parents=True, exist_ok=True)
    return p
