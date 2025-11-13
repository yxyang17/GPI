"""State/action matching utilities shared by GPI policies."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .types import EpisodeDataset

Key = Tuple[int, int]


class StateDatabase:
    """Dense GPU index over demonstrations providing fast GPI distance queries."""

    def __init__(
        self,
        dataset: EpisodeDataset,
        device: Optional[str] = None,
        subset_size: Optional[int] = None,
        batch_size: int = 500_000,
    ) -> None:
        self.dataset = dataset
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size
        self._build_full_buffers()
        self._active_indices = list(range(self._states_full.shape[0]))
        self._subset_size = subset_size
        self._sync_active_buffers()
        if subset_size is not None:
            self.resample_subset(subset_size)

    # ---------------------------------------------------------------------
    # Data preparation
    # ---------------------------------------------------------------------
    def _build_full_buffers(self) -> None:
        states: List[np.ndarray] = []
        actions: List[np.ndarray] = []
        keys: List[Key] = []
        key_to_index: Dict[Key, int] = {}
        for episode_idx in range(len(self.dataset)):
            episode = self.dataset[episode_idx]
            obs_norm = episode["obs"]
            act_norm = episode["action"]
            for timestep, (state, action) in enumerate(zip(obs_norm, act_norm)):
                key = (episode_idx, timestep)
                key_to_index[key] = len(states)
                states.append(state.astype(np.float32))
                actions.append(action.astype(np.float32))
                keys.append(key)
        self._key_to_full_index = key_to_index
        self._states_full = torch.tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        self._actions_full = torch.tensor(np.asarray(actions), dtype=torch.float32, device=self.device)
        self._keys_full = tuple(keys)

    # ---------------------------------------------------------------------
    # Subset control
    # ---------------------------------------------------------------------
    def resample_subset(self, subset_size: Optional[int] = None) -> None:
        total = self._states_full.shape[0]
        size = subset_size if subset_size is not None else self._subset_size
        if size is None or size >= total:
            self._active_indices = list(range(total))
        else:
            choice = np.random.choice(total, size=size, replace=False)
            self._active_indices = sorted(int(i) for i in choice)
        self._sync_active_buffers()

    def _sync_active_buffers(self) -> None:
        if not self._active_indices:
            self._states = torch.empty((0, self._states_full.shape[1]), device=self.device, dtype=torch.float32)
            self._actions = torch.empty((0, self._actions_full.shape[1]), device=self.device, dtype=torch.float32)
            self._keys = []
        else:
            active = torch.tensor(self._active_indices, dtype=torch.long, device=self.device)
            self._states = self._states_full.index_select(0, active)
            self._actions = self._actions_full.index_select(0, active)
            self._keys = [self._keys_full[i] for i in self._active_indices]
        self._key_to_active_idx = {key: idx for idx, key in enumerate(self._keys)}

    # ---------------------------------------------------------------------
    # Mutations
    # ---------------------------------------------------------------------
    def remove(self, key: Key) -> None:
        """Remove a state/action pair from the active buffers."""
        idx = self._key_to_active_idx.get(key)
        if idx is None:
            return
        mask = torch.ones(len(self._keys), dtype=torch.bool, device=self.device)
        mask[idx] = False
        self._states = self._states[mask]
        self._actions = self._actions[mask]
        self._keys.pop(idx)
        self._active_indices.pop(idx)
        self._key_to_active_idx.pop(key, None)
        for position in range(idx, len(self._keys)):
            self._key_to_active_idx[self._keys[position]] = position

    def restore(self, key: Key) -> None:
        """Reinstate a state/action pair using the full buffers."""
        if key in self._keys:
            return
        full_idx = self._key_to_full_index.get(key)
        if full_idx is None:
            return
        self._active_indices.append(full_idx)
        self._active_indices = sorted(set(self._active_indices))
        self._sync_active_buffers()

    # ---------------------------------------------------------------------
    # Queries
    # ---------------------------------------------------------------------
    @property
    def keys(self) -> Sequence[Key]:
        return self._keys

    @property
    def states(self) -> torch.Tensor:
        return self._states

    @property
    def actions(self) -> torch.Tensor:
        return self._actions

    def __len__(self) -> int:
        return self._states.shape[0]

    def distance(self, query: np.ndarray) -> torch.Tensor:
        query_tensor = torch.tensor(query, dtype=torch.float32, device=self.device).unsqueeze(0)
        n = self._states.shape[0]
        if n <= self.batch_size:
            return self.dataset.distance(self._states, query_tensor)
        chunks: List[torch.Tensor] = []
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            chunk = self._states[start:end]
            chunks.append(self.dataset.distance(chunk, query_tensor))
        return torch.cat(chunks, dim=0)

    def nearest(self, query: np.ndarray, k: int = 1, exclude: Optional[Iterable[Key]] = None) -> Tuple[torch.Tensor, List[Key]]:
        """Return distances and keys of the top-k demonstrations (Alg.1, lines 4-10)."""
        distances = self.distance(query)
        num_active = len(self._keys)
        if num_active == 0:
            return torch.empty(0, device=self.device), []
        mask = torch.ones(num_active, dtype=torch.bool, device=self.device)
        if exclude:
            for key in exclude:
                idx = self._key_to_active_idx.get(key)
                if idx is not None:
                    mask[idx] = False
        available = int(mask.sum().item())
        if available == 0:
            return torch.empty(0, device=self.device), []
        filtered = distances.masked_fill(~mask, float("inf"))
        top = min(k, available)
        values, indices = torch.topk(filtered, top, largest=False)
        finite_mask = torch.isfinite(values)
        values = values[finite_mask]
        selected_indices = indices[finite_mask]
        selected_keys = [self._keys[i] for i in selected_indices.cpu().tolist()]
        return values, selected_keys

    def knn_action(
        self,
        query: np.ndarray,
        k: int,
        lambda1: float,
        lambda2: float,
        exclude: Optional[Iterable[Key]] = None,
        prefetched: Optional[Tuple[torch.Tensor, List[Key]]] = None,
    ) -> np.ndarray:
        if prefetched is None:
            distances, keys = self.nearest(query, k=k, exclude=exclude)
        else:
            distances, keys = prefetched
            if exclude:
                # Prefetched distances are assumed to respect the exclusion set.
                pass
        if not keys:
            return np.zeros(self._actions_full.shape[1], dtype=np.float32)
        top = min(k, len(keys))
        distances = distances[:top]
        keys = keys[:top]
        idx = torch.tensor([self._key_to_active_idx[k_] for k_ in keys], dtype=torch.long, device=self.device)
        neighbor_states = self._states.index_select(0, idx)
        neighbor_actions = self._actions.index_select(0, idx)
        # Light-weight surrogate of the softmax weights w_i(x₀) from Alg.1 line 10.
        soft_weights = 1.0 / (distances + 1e-8)
        soft_weights = soft_weights / torch.sum(soft_weights)
        query_agent = torch.tensor(query[:2], dtype=torch.float32, device=self.device)
        neighbor_agent = neighbor_states[:, :2]
        progression = neighbor_actions[:, :2] - neighbor_agent
        attraction = neighbor_agent - query_agent
        displacement = lambda1 * progression + lambda2 * attraction
        blended = query_agent + torch.sum(displacement * soft_weights.unsqueeze(1), dim=0)
        # The dataset stores 2-D actions; keep shape consistent with upstream code
        result = neighbor_actions[0].clone()
        result[:2] = blended
        return result.cpu().numpy()
    
    def knn_object(
        self,
        query: np.ndarray,
        k: int,
        lambda1: float,
        lambda2: float,
        exclude: Optional[Iterable[Key]] = None,
        prefetched: Optional[Tuple[torch.Tensor, List[Key]]] = None,
    ) -> np.ndarray:
        if prefetched is None:
            distances, keys = self.nearest(query, k=k, exclude=exclude)
        else:
            distances, keys = prefetched
            if exclude:
                # Prefetched distances are assumed to respect the exclusion set.
                pass
        if not keys:
            return np.zeros(self._actions_full.shape[1], dtype=np.float32)
        top = min(k, len(keys))
        distances = distances[:top]
        keys = keys[:top]
        idx = torch.tensor([self._key_to_active_idx[k_] for k_ in keys], dtype=torch.long, device=self.device)
        neighbor_states = self._states.index_select(0, idx)
        neighbor_actions = self._actions.index_select(0, idx)
        # Light-weight surrogate of the softmax weights w_i(x₀) from Alg.1 line 10.
        soft_weights = 1.0 / (distances + 1e-8)
        soft_weights = soft_weights / torch.sum(soft_weights)
        
        query_agent = torch.tensor(query[:2], dtype=torch.float32, device=self.device)
        query_obs = torch.tensor(query, dtype=torch.float32, device=self.device)
        
        neighbor_agent = neighbor_states[:, :2]
        neighbor_obs = neighbor_states
        
        # progression = neighbor_actions[:, :2] - neighbor_agent
        progression = neighbor_obs - query_obs

        # attraction = neighbor_agent - query_agent
        attraction = neighbor_obs - query_obs

        displacement = lambda1 * progression + lambda2 * attraction
        # blended = query_agent + torch.sum(displacement * soft_weights.unsqueeze(1), dim=0)
        blended = query_obs + torch.sum(displacement * soft_weights.unsqueeze(1), dim=0)
        # The dataset stores 2-D actions; keep shape consistent with upstream code
        # result = neighbor_actions[0].clone()
        # result[:2] = blended
        result = neighbor_obs[0].clone()
        result += blended 
        return result.cpu().numpy()


__all__ = ["StateDatabase", "Key"]
