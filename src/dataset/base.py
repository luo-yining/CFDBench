from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor
import numpy as np
from torch.utils.data import Dataset


def load_data(data_dir: Path):
    print(f"Loading data from {data_dir}")
    u = np.load(data_dir / "u.npy")
    v = np.load(data_dir / "v.npy")
    mask = np.load(data_dir / "mask.npy")

    # Pad for adding boundary conditions, but no need to set the right edge
    u = np.pad(u, ((0, 0), (1, 1), (1, 0)), mode="constant", constant_values=0)
    v = np.pad(v, ((0, 0), (1, 1), (1, 0)), mode="constant", constant_values=0)
    # Boundaries are 1, interior is 0, but we flip it
    mask = 1 - np.pad(
        mask, ((1, 1), (1, 0)), mode="constant", constant_values=1
    )

    # Set the boundary conditions for u (for v it's all zeros)
    u[:, 1:-1, 0] = 0.5
    u[:, 1:-1, -1] = 0.5

    return u, v, mask


class CfdDataset(Dataset):
    """
    Base class for cfd datasets
    """

    def __geitem__(self, idx: int) -> tuple:
        """
        Returns a tuple of (features, labels, mask)
        """
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class CfdAutoDataset(Dataset):
    """
    Base class for auto-regressive dataset.
    """
    def __init__(self):
        self.all_features = None
        self.case_params = None

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Should return a tuple of (input, labels, mask)"""
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class KarmanDataset(CfdDataset):
    def __init__(self, data_dir: Path, time_step_size: int = 10):
        self.data_dir = data_dir
        self.time_step_size = time_step_size
        u, v, mask = load_data(data_dir)
        u = torch.FloatTensor(u)  # (T, h, w)
        v = torch.FloatTensor(v)  # (T, h, w)
        # u = u[50:170]
        # v = v[50:170]
        self.mask = torch.FloatTensor(mask)
        self.features = torch.stack([u, v], dim=1)  # (T, c, h, w)
        self.labels = self.features[
            time_step_size:
        ]  # (T - time_step_size, c, h, w)
        self.features = self.features[
            :-time_step_size
        ]  # (T - time_step_size, c, h, w)

    def __getitem__(self, idx: int):
        feat = self.features[idx]
        label = self.labels[idx]
        return (feat, self.mask, label)

    def __len__(self):
        return len(self.features)
