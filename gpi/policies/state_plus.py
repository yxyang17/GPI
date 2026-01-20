"""State-based GPI policy implementation."""

from __future__ import annotations

from typing import Optional, List, Dict, Any, Tuple
from collections import deque
import os, glob, torch, re

from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

import matplotlib.pyplot as plt
import numpy as np
import time

import torch

from .base import GPIPolicyBase, GPIConfig
from pusht_dynamics.models import InverseDynamics, ForwardDynamics
from pusht.datasets import detect_contact_pusht, detect_contact_pusht, draw_pusht_T_rectangles, create_pusht_pts

# fix me: need another way to handle device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# fix me: throw this function to util
def pick_checkpoint(ckpt_dir: str):
    """Pick the best (lowest val loss) checkpoint from a directory."""
    pattern = os.path.join(ckpt_dir, "*.pt")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No checkpoints found under {ckpt_dir}")
    def loss_from_name(path):
        m = re.search(r"loss([0-9eE\+\-\.]+)\.pt$", os.path.basename(path))
        if m:
            try: return float(m.group(1))
            except ValueError: pass
        return float("inf")
    files.sort(key=loss_from_name)
    return files[0]  # best

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
        forward_model = "/home/ynyg/yuxuan/GPI/GPI/runs/forward_abs_contact_True_bs512_lr0.001_20260120_003613"
        
        if config.use_relative_action:
            print("Using relative action inverse model.")
            inverse_model = "/home/ynyg/yuxuan/GPI/GPI/runs/inverse_re_bs512_20251215_225843"
        elif not config.use_relative_action:
            print("Using absolute action inverse model.")
            inverse_model = "/home/ynyg/yuxuan/GPI/GPI/runs/inverse_abs_contact_True_bs512_lr0.001_20260120_002813"
        else:
            raise ValueError("config.use_relative_action must be bool")
        forward_ckpt_dir = os.path.join(forward_model, "checkpoints")
        forward_ckpt_path = pick_checkpoint(forward_ckpt_dir)
        inverse_ckpt_dir = os.path.join(inverse_model, "checkpoints")
        inverse_ckpt_path = pick_checkpoint(inverse_ckpt_dir)
        self.inverse_dynamics_model = load_inverse_model(inverse_ckpt_path, 5, 2)
        self.forward_dynamics_model = load_forward_model(forward_ckpt_path, 5, 2)

        self.prev_action = None
        self.prev_obs = None
        self.planned_traj = None
        self.temporal_plan = deque()
        self.temporal_plan_next = deque()
        self.curr_key_adjusted = None
        self.prev_key = None
        self.curr_key_original = None
        self.magic_episode = None
        self.magic_step = None

        
        super().__init__(config)


        self.detect_contact = config.detect_contact
        if self.detect_contact:
            print("Contact detection is enabled in StateGPIPolicyPlus.")
            self.object_pcd = create_pusht_pts(1024 * 5, self.random_seed)

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

    # original get_action function
    # def get_action(self, observation: np.ndarray) -> np.ndarray:
    #     inference_start = time.time()

    #     # Step 1: project raw observation to the normalised latent (Alg.1 line 1).
    #     current_obs = observation[-1] if observation.ndim > 1 else observation
    #     current_obs = np.asarray(current_obs, dtype=np.float32)
    #     normalized_obs = self._normalize_obs(current_obs)

    #     # Optional exploration noise keeps the multi-modal behaviour.
    #     noisy_obs = self.add_observation_noise(normalized_obs)

    #     # Step 2-11: geometry-aware policy synthesis in normalised space.
    #     action_norm = self._compute_action_from_normalized(noisy_obs)
    #     if action_norm is None:
    #         return np.zeros(2, dtype=np.float32)
    #     action_raw = self._unnormalize_action(action_norm)
    #     final_action = self._to_global_if_needed(current_obs, action_raw)
    #     final_action = self._apply_action_smoothing(final_action)
    #     duration = time.time() - inference_start
    #     self._record_inference_time(duration)
    #     self.previous_action = final_action
    #     self.step_count += 1
    #     return final_action

    # my version
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        inference_start = time.time()

        # Step 1: project raw observation to the normalised latent (Alg.1 line 1).
        current_obs = observation[-1] if observation.ndim > 1 else observation
        current_obs = np.asarray(current_obs, dtype=np.float32)

        if self.dataset.use_object_centric_frame:
            # fix me: it is better to handle this in the dataset class
            # query = ...  so keep the same interface and array is handled inside different dataset class
            print("[get_action]: Dataset use_object_centric_frame is True, converting current_obs to object centric frame.")
            current_obs[:2] = self.dataset.global_state_to_relative(torch.from_numpy(current_obs.astype(np.float32)[None,:]))[0].numpy()            


        normalized_obs = self._normalize_obs(current_obs)

        # Optional exploration noise keeps the multi-modal behaviour.
        noisy_obs = self.add_observation_noise(normalized_obs)

        # Step 2-11: geometry-aware policy synthesis in normalised space.
        plan, pred_traj = self._compute_plan_from_normalized(noisy_obs)
        if plan is None:
            return None # np.zeros(2, dtype=np.float32)
        
        # prepare input for inverse model
        if self.prev_obs is None:
            self.prev_obs = noisy_obs
        
        prev_obs_ts = torch.as_tensor(self.prev_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        noisy_obs_ts = torch.as_tensor(noisy_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        # print("prev_obs_ts", self._unnormalize_obs(prev_obs_ts.detach().cpu().numpy()))
        # print("noisy_obs_ts", self._unnormalize_obs(noisy_obs_ts.detach().cpu().numpy()))
        # print("plan", self._unnormalize_obs(plan.detach().cpu().numpy()))

        # ynyg test: use inverse model to get actio
        # plan_ts = torch.as_tensor(plan, dtype=torch.float32, device=DEVICE).unsqueeze(0)        
        # action_norm = self.inverse_dynamics_model(prev_obs_ts, noisy_obs_ts, plan_ts)
        # action_norm = action_norm.detach().cpu().numpy().squeeze(0)
        # action_raw = self._unnormalize_action(action_norm)

        # ynyg test: use gpi to get action
        action_raw = self._unnormalize_obs(plan[None,:])[0,:2]
        self.prev_obs = noisy_obs


        # maybe calculate action from inverse model, adding forward jacobian for feedback


        
        # print("action_raw", action_raw)
        final_action = self._to_global_if_needed(current_obs, action_raw)
        # print("final_action", final_action)
        final_action = self._apply_action_smoothing(final_action)
        # print("final_action after smoothing", final_action)
        duration = time.time() - inference_start
        self._record_inference_time(duration)
        self.previous_action = final_action
        self.step_count += 1

        if self.detect_contact:
            ax, ay, ox, oy, oa = current_obs
            is_c, _ = detect_contact_pusht(
                finger_pos=np.array([ax, ay], dtype=np.float64),
                obj_pos_world=np.array([ox, oy], dtype=np.float64),
                obj_rad=float(oa),
                fin_rad=float(15),
                margin=float(3),
                pts_num=int(1024 * 5),
                use_object_centric_frame=self.dataset.use_object_centric_frame,
                seed=int(self.random_seed),
                object_pcd=self.object_pcd,
            )
        # show plan and final action in plot
        if self.debug:
            print(f"is contact: {is_c}")
            print(f"current key: {self.curr_key_adjusted}; original: {self.curr_key_original}; previous: {self.prev_key};\n")
            fig, ax = plt.subplots(figsize=(6,6))
            plan_np = plan
            plan_np = self._unnormalize_obs(plan_np)
            agent_pos = current_obs[:2]
            object_pos = current_obs[2:4]
            object_ori = current_obs[4]
            object_pos_in_demo = plan_np[2:4]
            object_ori_in_demo = plan_np[4]
            ax.plot(pred_traj[:, 0], pred_traj[:, 1], color = 'orange', label='agent traj')
            ax.plot(pred_traj[:, 2], pred_traj[:, 3], color = 'green', label='object traj')
            ax.scatter(action_raw[0], action_raw[1], c='r', marker='x', s=100, label='Final Action')
            if is_c:
                ax.scatter(agent_pos[0], agent_pos[1], c='r', marker='o', s=100, label='Agent')
            else:
                ax.scatter(agent_pos[0], agent_pos[1], c='orange', marker='o', s=100, label='Agent')
            ax.scatter(object_pos[0], object_pos[1], c='g', marker='o', s=100, label='Object')
            ax.scatter(plan_np[0], plan_np[1], c='orange', marker='x', s=100, label='Plan agent')
            ax.scatter(plan_np[2], plan_np[3], c='blue', marker='o', s=100, label='Plan object')

            # # draw T block
            # def draw_pusht_T_rectangles(ax, ox, oy, oa, facecolor="lightblue", alpha=0.5, edgecolor=None):
            #     """
            #     Draw PushT T-block as two rectangles in world frame.
            #     Object-frame geometry matches create_pusht_pts():
            #     - bottom bar: x [-60,60], y [0,30]
            #     - stem:       x [-15,15], y [30,120]
            #     """
            #     # Object-frame rectangles (defined by lower-left corner, width, height)
            #     rect_bottom = Rectangle((-60, 0), 120, 30, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor)
            #     rect_stem   = Rectangle((-15, 30), 30, 90, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor)

            #     # Transform: rotate around object-frame origin (0,0), then translate to (ox,oy)
            #     # print(f"affine2d: {Affine2D().rotate(oa).translate(ox, oy)}")
            #     T = Affine2D().rotate(oa).translate(ox, oy) + ax.transData    
            #     # print(f"ax.transData: {ax.transData}")
            #     # print(f"T: {T}")

            #     rect_bottom.set_transform(T)
            #     rect_stem.set_transform(T)

            #     ax.add_patch(rect_bottom)
            #     ax.add_patch(rect_stem)
            draw_pusht_T_rectangles(ax, object_pos[0], object_pos[1], object_ori, facecolor="lightblue", alpha=0.5)
            # draw target block
            draw_pusht_T_rectangles(ax, 256, 256, np.pi / 4, facecolor="LightGreen", alpha=0.5)
            # draw block in demo
            draw_pusht_T_rectangles(ax, object_pos_in_demo[0], object_pos_in_demo[1], object_ori_in_demo, facecolor="blue", alpha=0.5)

            ax.legend()
            ax.set_title('Plan and Final Action')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.grid(True)
            ax.set_xlim(0, 512)
            ax.set_ylim(512, 0)  # invert y by ordering high->low (better than invert_yaxis)
            # ax.set_xlim(-256, 256)
            # ax.set_ylim(256, -256)  # invert y by ordering high->low (better than invert_yaxis)
            ax.set_aspect("equal", adjustable="box")
            ax.margins(0)
            ax.set_autoscale_on(False)
            plt.show()

        return final_action

    def _compute_plan_from_normalized(
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
            # fix me: try one trajectory to the end
            if self.magic_episode is None:
                print("triggering magic episode and step")
                self.magic_episode = keys[0][0]
                self.magic_step = keys[0][1]
            else:
                self.magic_step += 1
                if (self.magic_episode, self.magic_step) not in self.database._key_to_active_idx:
                    self.magic_step -= 1  # stay at the end
            # keys = [(self.magic_episode, self.magic_step)]

            # for dist, key in zip(distances, keys):
            #     print(f"Distance: {dist:.5f}, Key: {key}")
            # print("self.recent_keys:", self.recent_keys)
            # Local GPI policy: blend progression/attraction flows of the nearest demo.
            plan_norm, pred_traj = self.database.knn_object(
                normalized_obs,
                k=self.k_neighbors,
                lambda1=lambda1,
                lambda2=(
                    float(self.fixed_lambda2) if self.fixed_lambda2 is not None else 1.0
                ),
                exclude=self.recent_set,
                prefetched=(distances, keys),
            )

            # fix me: need better hanlding of skip recent set
            # if pred_traj.shape[0] > 4:
            self._consume_key(keys[0])
            # else:
            #     print("Pred traj length <=1, not consuming key.")
            self.prev_key = self.curr_key_adjusted
            self.curr_key_adjusted = keys[0]
            
            return plan_norm, pred_traj
        # if self.plan.empty():
        if len(self.temporal_plan) <= 1:  # fix me: at the begining, it is empty, afterwards, it always has one key left after action horizion
            # Step 5-6: pick the demonstration/time with minimal combined distance.
            _, keys = self.database.nearest(
                normalized_obs, k=1, exclude=self.recent_set
            )
            if not keys:
                return None
            start_key = keys[0]  # κ(x₀) = argmin d_t^{(i)}
            # self.plan.load(start_key[0], start_key[1])
            self.temporal_plan.clear()
            self.temporal_plan_next.clear()
            self.temporal_plan.append(start_key)
            next_key = start_key

            # get trajectory for debug and visualization
            epi, start_t = start_key
            self.prev_key = self.curr_key_adjusted
            self.curr_key_adjusted = start_key
            sample = self.dataset[epi]  # {'obs': normalized_obs, 'action': normalized_action}
            Xn = np.asarray(sample["obs"], dtype=np.float32)
            t0 = int(np.clip(start_t, 0, len(Xn) - 1))
            Xsel_n = Xn[t0:]
            Xsel_n = self.dataset.unnormalize_obs(Xsel_n)
            self.planned_traj = Xsel_n
            for i in range(self.action_horizon):
                prev_key = next_key
                next_key = (prev_key[0], prev_key[1] + 1)  # only use the nearest neighbor for future state
                if next_key not in self.database._key_to_active_idx:
                    next_key = prev_key
                    # fix me: todo, handle multiples keys
                self.temporal_plan_next.append(next_key)
                self.temporal_plan.append(next_key)
                
            
        # total_steps = len(self.plan.actions)
        total_steps = self.action_horizon
        # action_norm, state_norm, timestep = self.plan.pop()
        # step_in_plan = self.plan.pointer - 1
        neighbor_key = self.temporal_plan.popleft()
        neighbor_next_key = self.temporal_plan_next.popleft()
        # if self.debug:
        #     print("popped neighbor key:", neighbor_key)
        #     print("popped neighbor next key:", neighbor_next_key)
        #     print("remaining temporal plan keys length:", len(self.temporal_plan))
        #     print("remaining temporal plan next keys length:", len(self.temporal_plan_next))
        neighbor_idx = torch.tensor([self.database._key_to_active_idx[neighbor_key]], dtype=torch.long, device=self.database.device)
        neighbor_next_idx = torch.tensor([self.database._key_to_active_idx[neighbor_next_key]], dtype=torch.long, device=self.database.device)
        neighbor_states = self.database._states.index_select(0, neighbor_idx)
        neighbor_states_future = self.database._states.index_select(0, neighbor_next_idx)


        step_in_plan = self.action_horizon - (len(self.temporal_plan))
        if total_steps == 0:
            return None
        if self.fixed_lambda2 is not None:
            lambda2 = float(self.fixed_lambda2)
        else:
            lambda2 = self.calculate_dynamic_lambda2(step_in_plan, total_steps)
        # query_agent = normalized_obs[:2]
        query_obs = torch.tensor(normalized_obs, dtype=torch.float32, device=self.database.device)

        # neighbor_agent = state_norm[:2]
        neighbor_obs = neighbor_states

        # Step 6: progression flow u_prog = ẋ κ (x₀) following the local tangent.
        # u_prog = action_norm[:2] - neighbor_agent
        progression = neighbor_states_future - neighbor_obs
        # Step 7: attraction flow u_att = -∇ d_rob that steers toward the demo point.
        # u_att = neighbor_agent - query_agent
        attraction = neighbor_obs - query_obs

        # Step 8-11: local policy π_i(x₀) = λ₁ u_prog + λ₂ u_att.
        # agent_flow = query_agent + lambda1 * u_prog + lambda2 * u_att
        displacement = lambda1 * progression + lambda2 * attraction
        local_policy = query_obs + displacement[0]

        # local_policy = action_norm.copy()
        # local_policy[:2] = agent_flow
        executed_key = (neighbor_key[0], neighbor_key[1])
        if executed_key is not None:
            self._consume_key(executed_key)

        return local_policy, self.planned_traj # fix me: not very good to return self.planned_traj   
        
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
            action_norm, pred_traj = self.database.knn_action(
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
