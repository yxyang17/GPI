"""Vision-based GPI policy implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import time
import torch

from .base import GPIPolicyBase, GPIConfig
from ..vision import load_models, predict_object_state


@dataclass
class VisionPolicyConfig(GPIConfig):
    vision_checkpoint: str = ""
    dynamic_subset: bool = False
    subset_change_interval: int = 50
    memory_length: int = 500


class VisionGPIPolicy(GPIPolicyBase):
    """GPI policy that uses an image encoder to supply the query latent."""
    def __init__(self, config: VisionPolicyConfig) -> None:
        self.dynamic_subset = config.dynamic_subset
        self.subset_change_interval = config.subset_change_interval
        self.memory_length = config.memory_length
        self.vision_inference_times: list[float] = []
        self.total_vision_inference_time = 0.0
        self.recent_queue: list[tuple[int, int]] = []
        self.recent_set: set[tuple[int, int]] = set()
        super().__init__(config)
        if not config.vision_checkpoint:
            raise ValueError("vision_checkpoint must be provided")
        device = self.database.states.device
        self.vision_encoder, self.state_predictor = load_models(config.vision_checkpoint, device)
        stats = self.dataset.stats["obs"]
        self.agent_min = stats.min[:2]
        self.agent_max = stats.max[:2]
        self.agent_span = np.maximum(self.agent_max - self.agent_min, 1e-6)
        self.object_min = stats.min[2:]
        self.object_max = stats.max[2:]
        self.object_span = np.maximum(self.object_max - self.object_min, 1e-6)

    def _post_reset(self) -> None:
        self.vision_inference_times.clear()
        self.total_vision_inference_time = 0.0
        self.recent_queue.clear()
        self.recent_set.clear()

    def _maybe_resample_subset(self) -> None:
        if not self.dynamic_subset or self.config.subset_size is None:
            return
        if self.step_count > 0 and self.step_count % self.subset_change_interval == 0:
            self.database.resample_subset()

    def _remember_key(self, key: tuple[int, int]) -> None:
        if key in self.recent_set:
            return
        self.recent_set.add(key)
        self.recent_queue.append(key)
        while len(self.recent_queue) > self.memory_length:
            oldest = self.recent_queue.pop(0)
            self.recent_set.discard(oldest)

    def _build_normalized_observation(self, agent_pos: np.ndarray, object_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        agent_pos = np.asarray(agent_pos, dtype=np.float32)
        object_state = np.asarray(object_state, dtype=np.float32)
        agent_norm = (agent_pos - self.agent_min) / self.agent_span
        agent_norm = agent_norm * 2.0 - 1.0
        clipped_object = np.clip(object_state, self.object_min, self.object_max)
        object_norm = (clipped_object - self.object_min) / self.object_span
        object_norm = object_norm * 2.0 - 1.0
        normalized = np.concatenate([agent_norm, object_norm], axis=0)
        return normalized.astype(np.float32), clipped_object.astype(np.float32)

    def get_action(self, image_obs: np.ndarray | torch.Tensor, agent_pos: np.ndarray | torch.Tensor) -> np.ndarray:
        inference_start = time.time()
        self._maybe_resample_subset()

        # Step 1: obtain the most recent camera frame / agent pose and encode the object pose.
        image = image_obs[-1] if hasattr(image_obs, "ndim") and image_obs.ndim > 3 else image_obs
        agent = agent_pos[-1] if hasattr(agent_pos, "ndim") and agent_pos.ndim > 1 else agent_pos
        if isinstance(agent, torch.Tensor):
            agent = agent.detach().cpu().numpy()
        vision_start = time.time()
        object_state = predict_object_state(image, self.vision_encoder, self.state_predictor, self.database.states.device)
        vision_duration = time.time() - vision_start
        self.vision_inference_times.append(vision_duration)
        self.total_vision_inference_time += vision_duration

        # Build GPI latent (projection + feature encoder Ψ).
        normalized_obs, object_raw = self._build_normalized_observation(agent, object_state)
        noisy_obs = self.add_observation_noise(normalized_obs)

        # Step 2-11: run geometry-aware blending identical to the state variant.
        action_norm = self._compute_action_from_normalized(noisy_obs)
        if action_norm is None:
            return np.zeros(2, dtype=np.float32)
        action_raw = self._unnormalize_action(action_norm)
        agent_raw = np.asarray(agent, dtype=np.float32)
        full_obs_raw = np.concatenate([agent_raw, object_raw])
        final_action = self._to_global_if_needed(full_obs_raw, action_raw)
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
        exclude = self.recent_set
        if self.action_horizon == 1:
            distances, keys = self.database.nearest(
                normalized_obs, k=self.k_neighbors, exclude=exclude
            )
            if not keys:
                return None
            # Local GPI policy: blend progression/attraction flows of the nearest demo.
            action_norm = self.database.knn_action(
                normalized_obs,
                k=self.k_neighbors,
                lambda1=lambda1,
                lambda2=float(self.fixed_lambda2) if self.fixed_lambda2 is not None else 1.0,
                exclude=exclude,
                prefetched=(distances, keys),
            )
            self._remember_key(keys[0])
            return action_norm
        if self.plan.empty():
            _, keys = self.database.nearest(normalized_obs, k=1, exclude=exclude)
            if not keys:
                return None
            start_key = keys[0]  # κ(x₀) = argmin d_t^{(i)}
            self.plan.load(start_key[0], start_key[1])
        total_steps = len(self.plan.actions)
        action_norm, state_norm, timestep = self.plan.pop()
        step_in_plan = self.plan.pointer - 1
        if self.fixed_lambda2 is not None:
            lambda2 = float(self.fixed_lambda2)
        else:
            lambda2 = self.calculate_dynamic_lambda2(step_in_plan, total_steps)
        query_agent = normalized_obs[:2]
        neighbor_agent = state_norm[:2]
        u_prog = action_norm[:2] - neighbor_agent
        u_att = neighbor_agent - query_agent
        agent_flow = query_agent + lambda1 * u_prog + lambda2 * u_att
        local_policy = action_norm.copy()
        local_policy[:2] = agent_flow
        executed_key = (self.plan.episode_idx, timestep)
        if executed_key is not None:
            self._remember_key(executed_key)
        return local_policy

    def get_full_inference_stats(self) -> dict:
        total_stats = self.get_inference_stats()
        if not self.vision_inference_times:
            vision_stats = {
                "mean_ms": 0.0,
                "std_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "total_s": 0.0,
                "count": 0,
            }
        else:
            values = np.array(self.vision_inference_times) * 1000.0
            vision_stats = {
                "mean_ms": float(np.mean(values)),
                "std_ms": float(np.std(values)),
                "min_ms": float(np.min(values)),
                "max_ms": float(np.max(values)),
                "total_s": float(self.total_vision_inference_time),
                "count": len(self.vision_inference_times),
            }
        return {"policy": total_stats, "vision": vision_stats}


__all__ = ["VisionGPIPolicy", "VisionPolicyConfig"]
