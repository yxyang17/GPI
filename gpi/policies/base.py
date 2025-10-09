"""Shared utilities for geometry-aware policy imitation (GPI) controllers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from ..database import StateDatabase
from ..planning import PlanBuffer
from ..types import EpisodeDataset


@dataclass
class GPIConfig:
    dataset_path: str
    dataset_loader: Optional[Callable[[str, bool], EpisodeDataset]] = None
    to_global_action: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
    use_relative_action: bool = True
    action_smoothing: float = 0.0
    k_neighbors: int = 3
    obs_noise_std: float = 0.01
    enable_obs_noise: bool = True
    noise_decay_rate: float = 0.995
    min_noise_std: float = 0.001
    device: Optional[str] = None
    batch_size: int = 500_000
    subset_size: Optional[int] = None
    random_seed: int = 0
    action_horizon: int = 1
    fixed_lambda1: Optional[float] = None
    fixed_lambda2: Optional[float] = None


class GPIPolicyBase:
    """Base class providing noise handling, statistics, and planning helpers.

    The subclasses follow Algorithm 1 from Geometry-Aware Policy Imitation (GPI):
    1. Encode the query observation.
    2. Measure its distance to every demonstration state.
    3. Blend the demonstration progression flow and attraction flow using λ1/λ2.
    """

    def __init__(self, config: GPIConfig) -> None:
        self.config = config
        if config.dataset_loader is None:
            raise ValueError("GPIConfig.dataset_loader must be provided")
        self.dataset = config.dataset_loader(
            config.dataset_path,
            use_relative_action=config.use_relative_action,
        )
        self.to_global_action = config.to_global_action
        self.database = StateDatabase(
            self.dataset,
            device=config.device,
            subset_size=config.subset_size,
            batch_size=config.batch_size,
        )
        self.plan = PlanBuffer(self.dataset, config.action_horizon)
        self.random_seed = config.random_seed
        self.rng = np.random.default_rng(config.random_seed)
        self.use_relative_action = config.use_relative_action
        self.k_neighbors = config.k_neighbors
        self.enable_obs_noise = config.enable_obs_noise
        self.obs_noise_std = config.obs_noise_std
        self.noise_decay_rate = config.noise_decay_rate
        self.min_noise_std = config.min_noise_std
        self.action_horizon = max(1, config.action_horizon)
        self.fixed_lambda1 = config.fixed_lambda1
        self.fixed_lambda2 = config.fixed_lambda2
        self.action_smoothing = float(config.action_smoothing) if config.action_smoothing is not None else 0.0
        if self.action_smoothing < 0.0:
            raise ValueError("action_smoothing must be non-negative")
        if self.use_relative_action:
            if self.to_global_action is None and hasattr(self.dataset, "relative_action_to_global"):
                self.to_global_action = getattr(self.dataset, "relative_action_to_global")
            if self.to_global_action is None:
                raise ValueError("Relative action conversion required when use_relative_action=True")
        self.reset()

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.previous_action = None
        self.step_count = 0
        self.current_noise_std = self.obs_noise_std
        self.plan.clear()
        self.inference_times: list[float] = []
        self.total_inference_time = 0.0
        self._post_reset()

    def _post_reset(self) -> None:  # hook for subclasses
        pass

    # ------------------------------------------------------------------
    # Noise and scheduling
    # ------------------------------------------------------------------
    def add_observation_noise(self, normalized_obs: np.ndarray) -> np.ndarray:
        if not self.enable_obs_noise or self.current_noise_std <= 0:
            return normalized_obs
        noise = self.rng.normal(0.0, self.current_noise_std, size=normalized_obs.shape)
        noisy = np.clip(normalized_obs + noise, -1.0, 1.0)
        self.current_noise_std = max(self.current_noise_std * self.noise_decay_rate, self.min_noise_std)
        return noisy

    @staticmethod
    def calculate_dynamic_lambda2(step_in_plan: int, total_steps: int) -> float:
        if total_steps <= 1:
            return 1.0
        progress = min(step_in_plan / (total_steps - 1), 1.0)
        # Maps the remaining plan length to the λ₂ attraction weight (Alg.1, line 7/8).
        return 0.5 + 0.5 * progress

    def get_inference_stats(self) -> dict:
        if not self.inference_times:
            return {
                "mean_ms": 0.0,
                "std_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "total_s": 0.0,
                "count": 0,
            }
        values = np.array(self.inference_times) * 1000.0
        return {
            "mean_ms": float(np.mean(values)),
            "std_ms": float(np.std(values)),
            "min_ms": float(np.min(values)),
            "max_ms": float(np.max(values)),
            "total_s": float(self.total_inference_time),
            "count": len(self.inference_times),
        }

    # ------------------------------------------------------------------
    # Helper conversions
    # ------------------------------------------------------------------
    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        return self.dataset.normalize_obs(obs)

    def _unnormalize_action(self, action_norm: np.ndarray) -> np.ndarray:
        return self.dataset.unnormalize_action(action_norm)

    def _apply_action_smoothing(self, action: np.ndarray) -> np.ndarray:
        if self.action_smoothing <= 0.0 or self.previous_action is None:
            return action
        smoothing = float(self.action_smoothing)
        previous = np.asarray(self.previous_action, dtype=np.float32)
        smoothed = (action + smoothing * previous) / (1.0 + smoothing)
        return smoothed.astype(np.float32, copy=False)

    def _to_global_if_needed(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        if not self.use_relative_action:
            return action
        if self.to_global_action is None:
            raise RuntimeError("to_global_action not set despite use_relative_action=True")
        obs_arr = np.asarray(obs, dtype=np.float32)
        action_arr = np.asarray(action, dtype=np.float32)
        try:
            result = self.to_global_action(obs_arr, action_arr)
        except Exception as exc:
            obs_t = torch.from_numpy(obs_arr).unsqueeze(0)
            action_t = torch.from_numpy(action_arr).unsqueeze(0)
            attempts = self.to_global_action(obs_t, action_t)
            if isinstance(attempts, torch.Tensor):
                result = attempts.squeeze(0).detach().cpu().numpy()
            else:
                # If the fallback still fails, surface the original error.
                try:
                    result = np.asarray(attempts, dtype=np.float32)
                except Exception as inner_exc:  # pragma: no cover - defensive
                    raise exc from inner_exc
        else:
            result = np.asarray(result, dtype=np.float32)
        if result.ndim > 1:
            if result.shape[0] == 1:
                result = result[0]
            else:
                raise RuntimeError(f"to_global_action returned unexpected shape {result.shape}")
        return result.astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # Timing helpers
    # ------------------------------------------------------------------
    def _record_inference_time(self, duration: float) -> None:
        self.inference_times.append(duration)
        self.total_inference_time += duration


__all__ = ["GPIConfig", "GPIPolicyBase"]
