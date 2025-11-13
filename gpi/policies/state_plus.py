"""State-based GPI policy implementation."""

from __future__ import annotations

from typing import Optional, List, Dict, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import time

import torch

from .base import GPIPolicyBase, GPIConfig
from ...pusht_dynamics.models import InverseDynamics, ForwardDynamics

# fix me: need another way to handle device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# fix me: throw this function to util
def load_inverse_model(ckpt_path: str, obs_dim: int, act_dim: int) -> InverseDynamics:
    model = InverseDynamics(obs_dim, act_dim).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt.get("model", ckpt)  # support raw or wrapped
    model.load_state_dict(state, strict=True)
    model.eval()
    return model
def load_forward_model(ckpt_path: str, obs_dim: int, act_dim: int) -> ForwardDynamics:
    model = ForwardDynamics(obs_dim, act_dim).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt.get("model", ckpt)  # support both raw state_dict and wrapped
    model.load_state_dict(state)
    model.eval()
    return model


class StateGPIPolicyPlus(GPIPolicyBase):
    """Implements Algorithm 1 (GPI) on state observations."""

    def __init__(self, config: GPIConfig, memory_length: Optional[int] = None) -> None:
        self.memory_length = memory_length
        self.recent_keys: list[tuple[int, int]] = []
        self.recent_set: set[tuple[int, int]] = set()
        # fix me: assign model directly, but it's better get from argument or config
        forward_model = "/home/ynyg/yuxuan/GPI/GPI/runs/forward_fp16_bs256_20251113_000413"
        inverse_model = "/home/ynyg/yuxuan/GPI/GPI/runs/inverse_fp16_bs256_20251113_005904"
        self.inverse_dynamics_model = load_inverse_model(inverse_model, 5*3, 2)
        self.forward_dynamics_model = load_forward_model(forward_model, 5+2, 2)
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

    def get_prediction(self, observation: np.ndarray) -> np.ndarray:
        inference_start = time.time()

        # Step 1: project raw observation to the normalised latent (Alg.1 line 1).
        current_obs = observation[-1] if observation.ndim > 1 else observation
        current_obs = np.asarray(current_obs, dtype=np.float32)
        normalized_obs = self._normalize_obs(current_obs)

        # Optional exploration noise keeps the multi-modal behaviour.
        noisy_obs = self.add_observation_noise(normalized_obs)

        # Step 2-11: geometry-aware policy synthesis in normalised space.
        trajs = self.get_knn_state_trajectories(
            noisy_obs,
            k=self.k_neighbors,
            exclude=self.recent_set, # not sure what exclude does
            horizon=self.action_horizon,
            return_raw=True,
        )
        if trajs is None:
            return None # np.zeros(2, dtype=np.float32)
        
        # prepare input for inverse model
        print(trajs)
        action_norm = self.inverse_dynamics_model()
        # maybe calculate action from inverse model, adding forward jacobian for feedback

        
        action_raw = self._unnormalize_action(action_norm)
        final_action = self._to_global_if_needed(current_obs, action_raw)
        final_action = self._apply_action_smoothing(final_action)
        duration = time.time() - inference_start
        self._record_inference_time(duration)
        self.previous_action = final_action
        self.step_count += 1
        return final_action

        
    def _compute_action_from_normalized(
        self, normalized_obs: np.ndarray
    ) -> Optional[np.ndarray]:
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
                lambda2=(
                    float(self.fixed_lambda2) if self.fixed_lambda2 is not None else 1.0
                ),
                exclude=self.recent_set,
                prefetched=(distances, keys),
            )
            self._consume_key(keys[0])
            return action_norm
        if self.plan.empty():
            # Step 5-6: pick the demonstration/time with minimal combined distance.
            _, keys = self.database.nearest(
                normalized_obs, k=1, exclude=self.recent_set
            )
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

    # ============================================================
    # KNN TRAJECTORY UTILITIES (fetch & plot)
    # ============================================================

    # --- Internal helpers ---
    def _fetch_episode_states_norm(self, epi: int) -> np.ndarray:
        """
        Get (T, Dx) normalized states for episode `epi` from the *dataset* that
        StateDatabase wraps. Your EpisodeDataset __getitem__ already returns normalized arrays.
        """
        sample = self.database.dataset[
            epi
        ]  # {'obs': normalized_obs, 'action': normalized_action}
        Xn = np.asarray(sample["obs"], dtype=np.float32)
        return Xn

    def _unnormalize_states_batch(self, Xn: np.ndarray) -> np.ndarray:
        """
        Best-effort inverse of database.normalize_obs using database.stats['obs'].
        Handles common field names: mean/std, mu/sigma, or dict with keys.
        If stats are unavailable, returns Xn as-is.
        """
        stats = getattr(self.database.dataset, "stats", None)
        if not isinstance(stats, dict) or "obs" not in stats:
            print("Warning: cannot unnormalize states; stats unavailable.")
            return Xn.astype(np.float32)

        s = stats["obs"]
        # Accept a variety of shapes / keys
        mean = s.get("mean", s.get("mu", None))
        std = s.get("std", s.get("sigma", s.get("stddev", None)))

        if mean is None or std is None:
            # Could be nested {'obs': {'stats': {'mean':..., 'std':...}}}
            inner = s.get("stats") if isinstance(s, dict) else None
            if isinstance(inner, dict):
                mean = inner.get("mean", mean)
                std = inner.get("std", std)

        if mean is None or std is None:
            # Fallback: unknown structure; cannot invert safely
            return Xn.astype(np.float32)

        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        return (Xn * std) + mean

    # --- Public: KNN trajectories ---
    def get_knn_state_trajectories(
        self,
        observation: np.ndarray,
        k: int | None = None,
        exclude=None,
        mode: str = "from_t",  # "full" | "from_t" | "window"
        horizon: int = 200,  # used when mode == "window"
        return_raw: bool = True,
    ):
        """
        Returns a list of dicts, one per neighbor:
        { 'episode': epi, 't': t, 'distance': d, 'states': (Ti, Dx) array }
        """
        # Normalize query with the policy's normalizer (consistent with database.normalize_obs)
        x_raw = observation[-1] if observation.ndim > 1 else observation
        x_norm = self._normalize_obs(np.asarray(x_raw, dtype=np.float32))

        # Nearest neighbors in normalized space
        k = k or getattr(self, "k_neighbors", 5)
        distances, keys = self.database.nearest(
            x_norm, k=k, exclude=exclude or self.recent_set
        )
        if not keys:
            return []

        results = []
        for d, (epi, t) in zip(distances, keys):
            Xn = self._fetch_episode_states_norm(
                epi
            )  # normalized episode array (T, Dx)
            if Xn.ndim != 2 or Xn.size == 0:
                continue

            if mode == "full":
                Xsel_n = Xn
            elif mode == "from_t":
                t0 = int(np.clip(t, 0, len(Xn) - 1))
                Xsel_n = Xn[t0:]
            elif mode == "window":
                t0 = int(np.clip(t, 0, len(Xn) - 1))
                t1 = int(min(len(Xn), t0 + max(1, int(horizon))))
                Xsel_n = Xn[t0:t1]
            else:
                raise ValueError(f"Unknown mode: {mode}")

            Xout = self._unnormalize_obs(Xsel_n) if return_raw else Xsel_n
            results.append(
                {
                    "episode": int(epi),
                    "t": int(t),
                    "distance": float(d),
                    "states": Xout.astype(np.float32),
                }
            )

        return results

    @staticmethod
    def align_and_weight_trajs(
        trajs: List[Dict[str, Any]],
        target_len: Optional[int] = None,
        p: float = 2.0,
        eps: float = 1e-6,
    ):
        """
        trajs: list of {'states': (Ti, Dx), 'distance': d, ...}
        Returns:
          X_stack: (K, T, Dx), weights: (K,), mean_traj: (T, Dx)
        """
        if not trajs:
            return None, None, None

        # Choose length
        if target_len is None:
            target_len = min(t["states"].shape[0] for t in trajs)  # safest
        K = len(trajs)
        Dx = trajs[0]["states"].shape[1]
        X_stack = np.zeros((K, target_len, Dx), dtype=np.float32)

        for i, t in enumerate(trajs):
            X = t["states"]
            Ti = X.shape[0]
            if Ti >= target_len:
                X_stack[i] = X[:target_len]
            else:
                X_stack[i, :Ti] = X
                X_stack[i, Ti:] = X[-1]  # pad by repeating last

        # Inverse-distance weights (larger weight for closer neighbors)
        d = np.array([max(eps, t["distance"]) for t in trajs], dtype=np.float32)
        w = 1.0 / (d**p)
        w = w / np.sum(w)

        # Weighted mean path
        mean_traj = np.tensordot(w, X_stack, axes=(0, 0))  # (T, Dx)
        return X_stack, w, mean_traj

    # --- Public: plotting ---
    def plot_knn_state_trajectories(
        self,
        observation: np.ndarray,
        k: int | None = None,
        exclude=None,
        mode: str = "from_t",
        horizon: int = 400,
        show_points: bool = False,
        show_mean: bool = True,
        align_len_for_mean: int = 200,  # set None to auto-min length
        pos_slice: slice = slice(0, 2),  # change if XY dims differ
    ) -> None:

        trajs = self.get_knn_state_trajectories(
            observation,
            k=k,
            exclude=exclude,
            mode=mode,
            horizon=horizon,
            return_raw=True,
        )
        if not trajs:
            print("No neighbors found.")
            return

        fig, ax = plt.subplots()

        for t in trajs:
            X = t["states"]
            xy = X[:, pos_slice]
            ax.plot(
                xy[:, 0],
                xy[:, 1],
                linewidth=1.8,
                label=f"epi {t['episode']} (d={t['distance']:.3f})",
            )
            if show_points:
                ax.scatter(xy[:, 0], xy[:, 1], s=8, alpha=0.6)

        if show_mean:
            X_stack, w, mean_traj = self.align_and_weight_trajs(
                trajs, target_len=align_len_for_mean
            )
            if mean_traj is not None:
                mxy = mean_traj[:, pos_slice]
                ax.plot(
                    mxy[:, 0],
                    mxy[:, 1],
                    linewidth=3.0,
                    linestyle="--",
                    label="weighted mean",
                )

        # Plot query point
        q = observation[-1] if observation.ndim > 1 else observation
        q = np.asarray(q, dtype=np.float32)
        qxy = q[pos_slice]
        ax.scatter([qxy[0]], [qxy[1]], marker="*", s=140, label="query", zorder=5)

        # Axis settings
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # --- FIXED VIEW SETTINGS ---
        ax.set_xlim(0, 512)
        ax.set_ylim(512, 0)  # invert y by ordering high->low (better than invert_yaxis)
        ax.set_aspect("equal", adjustable="box")
        ax.margins(0)
        ax.set_autoscale_on(False)

        ax.legend()
        ax.set_title(f"KNN state trajectories (mode={mode})")

        # Show non-blocking and wait for a key/mouse press to close the figure
        plt.show(block=False)
        print("Press any key or mouse button in the figure to close...")
        try:
            plt.waitforbuttonpress()  # blocks until a key or mouse button is pressed
        except Exception:
            pass
        plt.close(fig)
        


__all__ = ["StateGPIPolicy"]
