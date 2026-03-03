import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class MelNpyDataset(Dataset):
    """
    Loads mel-spectrogram features saved as .npy files.

    Expected .npy shape: [n_mels, time] (float32)
    Returns:
      x: torch.FloatTensor [1, n_mels, time]
      y: torch.LongTensor scalar (0=real, 1=fake)
    """
    def __init__(self, manifest_csv: str):
        self.df = pd.read_csv(manifest_csv)
        if not {"feature_path", "label"}.issubset(set(self.df.columns)):
            raise ValueError("manifest_csv must have columns: feature_path, label")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        x = np.load(row["feature_path"]).astype(np.float32)  # [n_mels, time]
        # add channel dim -> [1, n_mels, time]
        x = torch.from_numpy(x).unsqueeze(0)
        y = torch.tensor(int(row["label"]), dtype=torch.long)
        return x, y
