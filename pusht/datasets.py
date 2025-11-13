"""Dataset utilities for the PushT environment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import zarr

_EPS = 1e-8


@dataclass
class DataStats:
    """Min/max statistics used for symmetric normalisation."""

    min: np.ndarray
    max: np.ndarray

    def normalize(self, data: np.ndarray) -> np.ndarray:
        span = np.maximum(self.max - self.min, _EPS)
        scaled = (data - self.min) / span
        return scaled * 2.0 - 1.0

    def unnormalize(self, data: np.ndarray) -> np.ndarray:
        scaled = (data + 1.0) / 2.0
        return scaled * (self.max - self.min) + self.min


def compute_stats(array: np.ndarray) -> DataStats:
    return DataStats(min=array.min(axis=0), max=array.max(axis=0))


def normalize_data(data: np.ndarray, stats: DataStats) -> np.ndarray:
    return stats.normalize(data)


def unnormalize_data(data: np.ndarray, stats: DataStats) -> np.ndarray:
    return stats.unnormalize(data)


def se2_to_relative_action(obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Convert global agent targets into object-centric displacements."""
    object_xy = obs[:, 2:4]
    object_theta = obs[:, 4]
    cos_theta = torch.cos(object_theta)
    sin_theta = torch.sin(object_theta)
    rotation = torch.stack(
        [
            torch.stack([cos_theta, -sin_theta], dim=-1),
            torch.stack([sin_theta, cos_theta], dim=-1),
        ],
        dim=-1,
    )
    translated = action[:, :2] - object_xy
    return torch.bmm(rotation, translated.unsqueeze(-1)).squeeze(-1)


class PushTEpisodeDataset:
    """Episode-wise PushT demonstrations with symmetric normalisation."""

    def __init__(self, dataset_path: str, use_relative_action: bool = False) -> None:
        root = zarr.open(dataset_path, mode="r")
        actions = root["data"]["action"][:]
        obs = root["data"]["state"][:]

        if use_relative_action:
            obs_t = torch.from_numpy(obs.astype(np.float32))
            act_t = torch.from_numpy(actions.astype(np.float32))
            actions = se2_to_relative_action(obs_t, act_t).numpy()

        episode_ends = root["meta"]["episode_ends"][:]

        self.episodes: list[Dict[str, np.ndarray]] = []
        start = 0
        for end in episode_ends:
            self.episodes.append(
                {
                    "action": actions[start:end].astype(np.float32, copy=False),
                    "obs": obs[start:end].astype(np.float32, copy=False),
                }
            )
            start = int(end)

        self.stats: Dict[str, DataStats] = {
            "action": compute_stats(actions),
            "obs": compute_stats(obs),
        }

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        sample = self.episodes[idx]
        return {
            "action": self.normalize_action(sample["action"]),
            "obs": self.normalize_obs(sample["obs"]),
        }

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        return normalize_data(obs, self.stats["obs"])

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        return normalize_data(action, self.stats["action"])

    def unnormalize_action(self, action_norm: np.ndarray) -> np.ndarray:
        return unnormalize_data(action_norm, self.stats["action"])

    def unnormalize_obs(self, Xn: np.ndarray) -> np.ndarray:
        """Revert normalisation of states back to original scale."""
        return unnormalize_data(Xn, self.stats["obs"])
    
    def relative_action_to_global(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Map object-centric actions back into world coordinates."""
        object_xy = obs[:, 2:4]
        object_theta = obs[:, 4]
        cos_theta = torch.cos(object_theta)
        sin_theta = torch.sin(object_theta)
        rotation_inv = torch.stack(
            [
                torch.stack([cos_theta, sin_theta], dim=-1),
                torch.stack([-sin_theta, cos_theta], dim=-1),
            ],
            dim=-1,
        )
        rotated = torch.bmm(rotation_inv, action[:, :2].unsqueeze(-1)).squeeze(-1)
        return rotated + object_xy

    def distance(self, states: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """Compute PushT-specific distance between normalised states."""
        pos_diff = states[:, :4] - query[:, :4]
        pos_dist_sq = torch.sum(pos_diff ** 2, dim=1)
        angle_diff = states[:, 4] - query[:, 4]
        abs_angle = torch.abs(angle_diff)
        wrapped = torch.min(abs_angle, 2.0 - abs_angle)
        angle_dist_sq = wrapped ** 2
        return torch.sqrt(pos_dist_sq + angle_dist_sq)


def load_episode_dataset(dataset_path: str, use_relative_action: bool) -> PushTEpisodeDataset:
    return PushTEpisodeDataset(dataset_path, use_relative_action=use_relative_action)


class PushTImageDataset(torch.utils.data.Dataset):
    """Sequence dataset exposing stacked observations and images for PushT."""

    def __init__(
        self,
        dataset_path: str,
        obs_horizon: int = 1,
        pred_horizon: int = 1,
        action_horizon: int = 1,
    ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        store = zarr.open(dataset_path, mode="r")
        self.states = store["data"]["state"]
        self.actions = store["data"]["action"] if "action" in store["data"] else None
        self.images = store["data"]["img"]
        episode_ends = store["meta"]["episode_ends"][:].astype(int)
        self._indices: List[int] = []
        window = max(obs_horizon, pred_horizon, action_horizon)
        start = 0
        for end in episode_ends:
            limit = max(start, end - window + 1)
            for idx in range(start, limit):
                self._indices.append(idx)
            start = end

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        start_idx = self._indices[index]
        obs_stop = start_idx + self.obs_horizon
        images = self.images[start_idx:obs_stop]  # (obs_h, H, W, C)
        states = self.states[start_idx:obs_stop]
        images = torch.from_numpy(np.transpose(images, (0, 3, 1, 2))).float()
        states = torch.from_numpy(states.astype(np.float32))
        sample = {
            "image": images,
            "obs_all": states,
        }
        if self.actions is not None:
            act_stop = start_idx + self.action_horizon
            actions = self.actions[start_idx:act_stop]
            sample["action"] = torch.from_numpy(actions.astype(np.float32))
        return sample


__all__ = [
    "DataStats",
    "PushTEpisodeDataset",
    "PushTImageDataset",
    "load_episode_dataset",
    "normalize_data",
    "unnormalize_data",
]
