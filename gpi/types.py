"""Common typing helpers for GPI components."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Protocol

import numpy as np
import torch


class EpisodeDataset(Protocol):
    """Protocol describing the minimal dataset interface required by the policies."""

    stats: Mapping[str, Any]

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]: ...

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray: ...

    def normalize_action(self, action: np.ndarray) -> np.ndarray: ...

    def unnormalize_action(self, action: np.ndarray) -> np.ndarray: ...

    def distance(self, states: torch.Tensor, query: torch.Tensor) -> torch.Tensor: ...
