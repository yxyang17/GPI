"""Common trajectory planning helpers."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from .types import EpisodeDataset


class PlanBuffer:
    """Caches a short horizon of normalised actions/states around κ(x₀)."""

    def __init__(self, dataset: EpisodeDataset, horizon: int) -> None:
        self.dataset = dataset
        self.horizon = max(1, horizon)
        self.clear()

    def clear(self) -> None:
        self.episode_idx: Optional[int] = None
        self.start_timestep: Optional[int] = None
        self.pointer: int = 0
        self.actions: list[np.ndarray] = []
        self.states: list[np.ndarray] = []

    def load(self, episode_idx: int, timestep: int) -> None:
        episode = self.dataset[episode_idx]
        actions = episode["action"]
        states = episode["obs"]
        end = min(timestep + self.horizon, len(actions))
        slice_actions = actions[timestep:end]
        slice_states = states[timestep:end]
        if torch.is_tensor(slice_actions):
            slice_actions = slice_actions.cpu().numpy()
        if torch.is_tensor(slice_states):
            slice_states = slice_states.cpu().numpy()
        self.actions = [np.asarray(a) for a in slice_actions]
        self.states = [np.asarray(s) for s in slice_states]
        self.episode_idx = episode_idx
        self.start_timestep = timestep
        self.pointer = 0

    def pop(self) -> Tuple[np.ndarray, np.ndarray, int]:
        if self.pointer >= len(self.actions):
            raise RuntimeError("PlanBuffer is empty")
        action = self.actions[self.pointer]
        state = self.states[self.pointer]
        timestep = self.start_timestep + self.pointer
        self.pointer += 1
        return action, state, timestep

    def remaining(self) -> int:
        return max(0, len(self.actions) - self.pointer)

    def empty(self) -> bool:
        return self.remaining() == 0


__all__ = ["PlanBuffer"]
