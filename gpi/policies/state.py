"""State-based GPI policy implementation."""
from __future__ import annotations

from typing import Optional

import numpy as np
import time

from .base import GPIPolicyBase, GPIConfig


class StateGPIPolicy(GPIPolicyBase):
    """Implements Algorithm 1 (GPI) on state observations."""
    def __init__(self, config: GPIConfig, memory_length: Optional[int] = None) -> None:
        self.memory_length = memory_length
        self.recent_keys: list[tuple[int, int]] = []
        self.recent_set: set[tuple[int, int]] = set()
        super().__init__(config)

    def _post_reset(self) -> None:
        self.recent_keys.clear()
        self.recent_set.clear()

    def _consume_key(self, key: tuple[int, int]) -> None:
        if key in self.recent_set:
            return
        self.recent_set.add(key)
        self.recent_keys.append(key)
        if self.memory_length is not None:
            while len(self.recent_keys) > self.memory_length:
                oldest = self.recent_keys.pop(0)
                self.recent_set.discard(oldest)

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        inference_start = time.time()

        # Step 1: project raw observation to the normalised latent (Alg.1 line 1).
        current_obs = observation[-1] if observation.ndim > 1 else observation
        current_obs = np.asarray(current_obs, dtype=np.float32)
        normalized_obs = self._normalize_obs(current_obs)

        # Optional exploration noise keeps the multi-modal behaviour.
        noisy_obs = self.add_observation_noise(normalized_obs)

        # Step 2-11: geometry-aware policy synthesis in normalised space.
        action_norm = self._compute_action_from_normalized(noisy_obs)
        if action_norm is None:
            return np.zeros(2, dtype=np.float32)
        action_raw = self._unnormalize_action(action_norm)
        final_action = self._to_global_if_needed(current_obs, action_raw)
        final_action = self._apply_action_smoothing(final_action)
        duration = time.time() - inference_start
        self._record_inference_time(duration)
        self.previous_action = final_action
        self.step_count += 1
        return final_action

    def _compute_action_from_normalized(self, normalized_obs: np.ndarray) -> Optional[np.ndarray]:
        if len(self.database) == 0:
            return None
        lambda1 = float(self.fixed_lambda1) if self.fixed_lambda1 is not None else 1.0
        if self.action_horizon == 1:
            distances, keys = self.database.nearest(
                normalized_obs, k=self.k_neighbors, exclude=self.recent_set
            )
            if not keys:
                return None
            # Local GPI policy: blend progression/attraction flows of the nearest demo.
            action_norm = self.database.knn_action(
                normalized_obs,
                k=self.k_neighbors,
                lambda1=lambda1,
                lambda2=float(self.fixed_lambda2) if self.fixed_lambda2 is not None else 1.0,
                exclude=self.recent_set,
                prefetched=(distances, keys),
            )
            self._consume_key(keys[0])
            return action_norm
        if self.plan.empty():
            # Step 5-6: pick the demonstration/time with minimal combined distance.
            _, keys = self.database.nearest(normalized_obs, k=1, exclude=self.recent_set)
            if not keys:
                return None
            start_key = keys[0]  # κ(x₀) = argmin d_t^{(i)}
            self.plan.load(start_key[0], start_key[1])
        total_steps = len(self.plan.actions)
        action_norm, state_norm, timestep = self.plan.pop()
        step_in_plan = self.plan.pointer - 1
        if total_steps == 0:
            return None
        if self.fixed_lambda2 is not None:
            lambda2 = float(self.fixed_lambda2)
        else:
            lambda2 = self.calculate_dynamic_lambda2(step_in_plan, total_steps)
        query_agent = normalized_obs[:2]
        neighbor_agent = state_norm[:2]

        # Step 6: progression flow u_prog = ẋ κ (x₀) following the local tangent.
        u_prog = action_norm[:2] - neighbor_agent
        # Step 7: attraction flow u_att = -∇ d_rob that steers toward the demo point.
        u_att = neighbor_agent - query_agent

        # Step 8-11: local policy π_i(x₀) = λ₁ u_prog + λ₂ u_att.
        agent_flow = query_agent + lambda1 * u_prog + lambda2 * u_att
        local_policy = action_norm.copy()
        local_policy[:2] = agent_flow
        executed_key = (self.plan.episode_idx, timestep)
        if executed_key is not None:
            self._consume_key(executed_key)
        return local_policy


__all__ = ["StateGPIPolicy"]
