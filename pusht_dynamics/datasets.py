import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from typing import Tuple, List, Dict, Any


class InverseDynamicsDataset(Dataset):
    """
    Samples windows (o_{t-1}, o_t, o_{t+1}) -> a_t
    Expects base[i] to return {"obs": np.ndarray[T,...], "action": np.ndarray[Ta, A]}
    with values already normalized by the base dataset.
    """
    def __init__(self, base_episode_dataset):
        self.base = base_episode_dataset
        self.indices: List[Tuple[int, int]] = []
        for ei in range(len(self.base)):
            ep = self.base[ei]
            obs, act = ep["obs"], ep["action"]
            T, Ta = len(obs), len(act)
            for t in range(1, T - 1):
                if t < Ta and (t + 1) < T:
                    self.indices.append((ei, t))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        ei, t = self.indices[i]
        ep = self.base[ei]
        return {
            "o_prev":  torch.as_tensor(ep["obs"][t - 1], dtype=torch.float32),
            "o_curr":  torch.as_tensor(ep["obs"][t],     dtype=torch.float32),
            "o_next":  torch.as_tensor(ep["obs"][t + 1], dtype=torch.float32),
            "action":  torch.as_tensor(ep["action"][t],  dtype=torch.float32),
            "episode_idx": torch.tensor(ei),
            "t": torch.tensor(t),
        }


class ForwardDynamicsDataset(Dataset):
    """
    Samples (o_t, a_t) -> o_{t+1}
    Works when len(action) is T-1 or T.
    """
    def __init__(self, base_episode_dataset):
        self.base = base_episode_dataset
        self.indices: List[Tuple[int, int]] = []
        for ei in range(len(self.base)):
            ep = self.base[ei]
            obs, act = ep["obs"], ep["action"]
            T, Ta = len(obs), len(act)
            for t in range(0, T - 1):
                if t < Ta:
                    self.indices.append((ei, t))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        ei, t = self.indices[i]
        ep = self.base[ei]
        return {
            "o_curr":  torch.as_tensor(ep["obs"][t],     dtype=torch.float32),
            "o_next":  torch.as_tensor(ep["obs"][t + 1], dtype=torch.float32),
            "action":  torch.as_tensor(ep["action"][t],  dtype=torch.float32),
            "episode_idx": torch.tensor(ei),
            "t": torch.tensor(t),
        }
    
