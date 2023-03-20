from pathlib import Path
from typing import Optional
import pandas as pd
import torch
from torch.utils.data import Dataset
from functools import lru_cache
import numpy as np


class ActivationDataset(Dataset):
    """
    Loads activations stored at the given directory.
    If activations_path is None, the activations are assumed to be in the same directory as the prompt_csv.
    Each activation tensor is a 2D tensor of shape (num_layers, model_dim).
    If normalize is True, the activations are normalized to have mean 0 and std 1.
    """

    def __init__(
        self,
        prompt_csv: str,
        activations_path: Optional[str] = None,
        layer: Optional[int] = None,
        normalize=True,
    ):
        self.prompts = pd.read_csv(prompt_csv, index_col=0).astype({"label": int})
        self.activations_path = Path(activations_path or Path(prompt_csv).parent)

        self.layer = layer
        self.normalize = normalize

        if normalize:
            # statistics for normalization
            attrs = [
                "plus_mean",
                "plus_std",
                "minus_mean",
                "minus_std",
            ]
            for attr in attrs:
                statistic = np.load(self.activations_path / f"{attr}.npy")
                statistic = torch.from_numpy(statistic).float()
                setattr(self, attr, statistic)

    def __len__(self):
        return len(self.prompts)

    def set_layer(self, layer):
        self.layer = layer

    @lru_cache(maxsize=1024)
    def get_activations(self, sign, idx):
        """Load a single activation tensor from disk."""
        activations = np.load(self.activations_path / sign / f"{idx}.npy")
        activations = torch.from_numpy(activations).float()
        return activations

    def to_tensor(self, sign, idx):
        activations = torch.stack([self.get_activations(sign, i) for i in idx])

        if self.normalize:
            mean = getattr(self, f"{sign}_mean")
            std = getattr(self, f"{sign}_std")
            std = torch.clamp(std, min=1e-6)
            activations = (activations - mean) / std

        if self.layer is not None:
            activations = activations[:, self.layer]

        return activations

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)
        assert isinstance(idx, slice), "Only slices are supported"
        idx = self.prompts.index[idx]
        plus, minus = self.to_tensor("plus", idx), self.to_tensor("minus", idx)
        labels = self.prompts.loc[idx, "label"].values
        labels = torch.from_numpy(labels)
        return plus, minus, labels

def get_statistics(dataset: ActivationDataset, path: str):
    all_pluses_means, all_minuses_means = [], []
    all_pluses_stds, all_minuses_stds = [], []

    for i in range(0, len(dataset), 1024):
        x_plus, x_minus, y = dataset[i:i+1024]
        all_pluses_means.append(x_plus.sum(0))
        all_minuses_means.append(x_minus.sum(0))
        all_pluses_stds.append((x_plus ** 2).sum(0))
        all_minuses_stds.append((x_minus ** 2).sum(0))

    all_pluses_mean = torch.stack(all_pluses_means).sum(0) / len(dataset)
    all_minuses_mean = torch.stack(all_minuses_means).sum(0) / len(dataset)
    all_pluses_std = torch.sqrt(torch.stack(all_pluses_stds).sum(0) / len(dataset) - all_pluses_mean ** 2)
    all_minuses_std = torch.sqrt(torch.stack(all_minuses_stds).sum(0) / len(dataset) - all_minuses_mean ** 2)

    # save the above to disk
    path = Path(path)
    np.save(path / "plus_mean.npy", all_pluses_mean.numpy())
    np.save(path / "minus_mean.npy", all_minuses_mean.numpy())
    np.save(path / "plus_std.npy", all_pluses_std.numpy())
    np.save(path / "minus_std.npy", all_minuses_std.numpy())

    return all_pluses_mean, all_minuses_mean, all_pluses_std, all_minuses_std
