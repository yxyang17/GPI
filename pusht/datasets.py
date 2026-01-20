"""Dataset utilities for the PushT environment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import zarr

from matplotlib.patches import Rectangle, Circle
from matplotlib.transforms import Affine2D
import matplotlib.pyplot as plt

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
    print(f"array min, max: {array.min(axis=0)}, {array.max(axis=0)}")
    return DataStats(min=array.min(axis=0), max=array.max(axis=0))


def normalize_data(data: np.ndarray, stats: DataStats) -> np.ndarray:
    return stats.normalize(data)


def unnormalize_data(data: np.ndarray, stats: DataStats) -> np.ndarray:
    return stats.unnormalize(data)


#################################
# PushT contact detection
#################################
def create_pusht_pts(pts_num: int = 1024, seed: int = 0) -> np.ndarray:
    """
    Sample interior points of a PushT T-shape in the object frame.
    Units match the repo: x in [-60,60], y in [0,120].
    Deterministic if seed is fixed.
    Returns: (N,2)
    """
    rng = np.random.default_rng(seed)

    size_b = 120 * 30   # bottom bar area
    size_t = 90 * 30    # top stem area
    num_b = int(pts_num * size_b / (size_b + size_t))
    num_t = pts_num - num_b

    pts_b = rng.random((num_b, 2))
    pts_b[:, 0] = pts_b[:, 0] * 120 - 60   # [-60, 60]
    pts_b[:, 1] = pts_b[:, 1] * 30         # [0, 30]

    pts_t = rng.random((num_t, 2))
    pts_t[:, 0] = pts_t[:, 0] * 30 - 15    # [-15, 15]
    pts_t[:, 1] = pts_t[:, 1] * 90 + 30    # [30, 120]

    return np.concatenate([pts_b, pts_t], axis=0)



# -----------------------------
# 2D transform helpers
# -----------------------------

def posrad_to_Tmat(pos_xy: np.ndarray, rad: float) -> np.ndarray:
    """World-from-object transform T_W_O (3x3) for 2D pose (x,y,theta)."""
    c, s = float(np.cos(rad)), float(np.sin(rad))
    x, y = float(pos_xy[0]), float(pos_xy[1])
    return np.array([[c, -s, x],
                     [s,  c, y],
                     [0,  0, 1]], dtype=np.float64)

def trans_pt_2d(pt_xy: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply 3x3 homogeneous transform to a 2D point."""
    p = np.array([float(pt_xy[0]), float(pt_xy[1]), 1.0], dtype=np.float64)
    q = T @ p
    return q[:2]



def detect_contact_pusht(
    finger_pos: np.ndarray,  # (2,)
    obj_pos_world: np.ndarray,     # (2,)
    obj_rad: float,
    fin_rad: float,
    margin: float = 3.0,
    pts_num: int = 1024 * 5,
    use_object_centric_frame: bool = False,
    seed: int = 0,
    object_pcd: np.ndarray = None,
) -> tuple[bool, float]:
    """
    Repo-style PushT contact heuristic.
    Returns (is_contact, min_dist) where min_dist is in object-frame units.
    """
    if object_pcd is None:
        object_pcd = create_pusht_pts(pts_num, seed)

    if not use_object_centric_frame:
        # finger -> object frame
        T_W_O = posrad_to_Tmat(obj_pos_world, obj_rad)
        T_O_W = np.linalg.inv(T_W_O)
        finger_pos_obj = trans_pt_2d(finger_pos, T_O_W)
    else:
        finger_pos_obj = finger_pos
    # min distance to sampled interior points
    d = object_pcd - finger_pos_obj[None, :]
    min_dist = float(np.min(np.sqrt(np.sum(d * d, axis=1))))

    # threshold
    is_contact = (min_dist < float(fin_rad + margin))
    return is_contact, min_dist

# function name needs change
def append_is_contact_to_traj(
    traj: np.ndarray,      # (N,5) [ax,ay,ox,oy,oa]
    fin_rad: float,
    margin: float = 3.0,
    pts_num: int = 1024 * 5,
    use_object_centric_frame: bool = False,
    seed: int = 0,
    dtype=np.float32,
) -> np.ndarray:
    """
    Returns (N,6): [ax, ay, ox, oy, oa, is_contact]
    is_contact stored as 0.0/1.0
    """
    assert traj.ndim == 2 and traj.shape[1] == 5, f"traj dim is {traj.shape} but traj must be (N,5) [ax,ay,ox,oy,oa]"
    N = traj.shape[0]

    out = np.empty((N, 1), dtype=dtype)

    for i in range(N):
        ax, ay, ox, oy, oa = traj[i]
        is_c, _ = detect_contact_pusht(
            finger_pos=np.array([ax, ay], dtype=np.float64),
            obj_pos_world=np.array([ox, oy], dtype=np.float64),
            obj_rad=float(oa),
            fin_rad=float(fin_rad),
            margin=float(margin),
            pts_num=int(pts_num),
            use_object_centric_frame=use_object_centric_frame,
            seed=int(seed),
        )
        out[i, 0] = 1.0 if is_c else 0.0

    return out
#################################################

def draw_pusht_T_rectangles(ax, ox, oy, oa, facecolor="lightblue", alpha=0.5, edgecolor=None, label=None):
    """
    Draw PushT T-block as two rectangles in world frame.
    Object-frame geometry matches create_pusht_pts():
      - bottom bar: x [-60,60], y [0,30]
      - stem:       x [-15,15], y [30,120]
    """
    # Object-frame rectangles (defined by lower-left corner, width, height)
    rect_bottom = Rectangle((-60, 0), 120, 30, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor)
    rect_stem   = Rectangle((-15, 30), 30, 90, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor)

    # Transform: rotate around object-frame origin (0,0), then translate to (ox,oy)
    # print(f"affine2d: {Affine2D().rotate(oa).translate(ox, oy)}")
    T = Affine2D().rotate(oa).translate(ox, oy) + ax.transData    
    # print(f"ax.transData: {ax.transData}")
    # print(f"T: {T}")

    rect_bottom.set_transform(T)
    rect_stem.set_transform(T)

    ax.add_patch(rect_bottom)
    ax.add_patch(rect_stem)

def visualize_pusht_obs(
    obs,                 # [ax, ay, ox, oy, oa]
    pusher_radius=15.0,
    show_pusher=True,
    use_object_centric_frame=False,
    xlim=((0, 512)),
    ylim=((512, 0)),       # invert y like your other plots; use (0,512) if you don't want invert
    target_pos=[256, 256, np.pi / 4]
):
    ax_x, ax_y, ox, oy, oa, is_contact = [float(v) for v in obs]

    fig, ax = plt.subplots(figsize=(6, 6))
    if use_object_centric_frame:
        ax_title = "(Object-Centric Frame)"        
        draw_pusht_T_rectangles(ax, 0, 0, 0, facecolor="lightblue", alpha=0.5)
        ax.scatter([0], [0], s=15, color="lightblue")  # object center
    else:
        ax_title = "(World Frame)"
        draw_pusht_T_rectangles(ax, ox, oy, oa, facecolor="lightblue", alpha=0.5)
        ax.scatter([ox], [oy], s=15, color="lightblue")  # object center

    # target T block
    draw_pusht_T_rectangles(ax, target_pos[0], target_pos[1], target_pos[2], facecolor="LightGreen", alpha=0.5)
    ax.scatter([target_pos[0]], [target_pos[1]], s=15, color="LightGreen")  # object center
    
    if show_pusher:
        # pusher as a circle (like your code)qqqq
        print(f"ax_x={ax_x}, ax_y={ax_y}")
        if is_contact > 0.5:
            ax.add_patch(Circle((ax_x, ax_y), pusher_radius, color="red"))
        else:
            ax.add_patch(Circle((ax_x, ax_y), pusher_radius, color="black"))
        
        ax.scatter([ax_x], [ax_y], s=15, color="red")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_title(f"{ax_title} ax,ay=({ax_x:.1f},{ax_y:.1f})  ox,oy=({ox:.1f},{oy:.1f})  oa={oa:.3f}")
    plt.show()
    return fig, ax


# def se2_to_relative_action(obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
#     """Convert global agent targets into object-centric displacements."""
#     object_xy = obs[:, 2:4]
#     object_theta = obs[:, 4]
#     cos_theta = torch.cos(object_theta)
#     sin_theta = torch.sin(object_theta)
#     rotation = torch.stack(
#         [
#             torch.stack([cos_theta, -sin_theta], dim=-1),
#             torch.stack([sin_theta, cos_theta], dim=-1),
#         ],
#         dim=-1,
#     )
#     translated = action[:, :2] - object_xy
#     return torch.bmm(rotation, translated.unsqueeze(-1)).squeeze(-1)


class PushTEpisodeDataset:
    """Episode-wise PushT demonstrations with symmetric normalisation."""

    def __init__(self, dataset_path: str, use_relative_action: bool = False, use_object_centric_frame: bool = False, calculate_contact: bool = False) -> None:
        root = zarr.open(dataset_path, mode="r")
        actions = root["data"]["action"][:]
        obs = root["data"]["state"][:]

        # create a movement mask: 1 = object is moving, 0 = object not moving
        # movement_epsilon = 1e-3
        # obs_f = obs.astype(np.float32)
        # if obs_f.shape[0] > 1:
        #     diffs = np.linalg.norm(obs_f[1:, 2:] - obs_f[:-1, 2:], axis=1)
        #     mask = np.ones(obs_f.shape[0], dtype=np.uint8)
        #     mask[:-1] = (diffs >= movement_epsilon).astype(np.uint8)
        #     mask[-1] = 0
        # else:
        #     mask = np.zeros(obs_f.shape[0], dtype=np.uint8)
        print("Dataset use_relative_action:", use_relative_action, "use_object_centric_frame:", use_object_centric_frame, "calculate_contact:", calculate_contact)
        print("obs shape:", obs.shape, "actions shape:", actions.shape)
        

        # expose mask on the dataset instance for later use (and keep original obs/actions unchanged)
        
        self.use_relative_action = use_relative_action
        self.use_object_centric_frame = use_object_centric_frame
        self.calculate_contact = calculate_contact
        
        if self.use_relative_action:
            obs_t = torch.from_numpy(obs.astype(np.float32))
            act_t = torch.from_numpy(actions.astype(np.float32))
            actions = self.global_action_to_relative(obs_t, act_t).numpy()

        if self.use_object_centric_frame: # fix me: condition logical of use_object_centric_frame and use_relative_action need improvement
            obs_t = torch.from_numpy(obs.astype(np.float32))
            # fix me: I should handle this slice in the function, not here
            # this part should be obs = ...
            obs[:, :2] = self.global_state_to_relative(obs_t, obs_t[:, :2]).numpy()

        if calculate_contact:
        # not the most efficient way, but simple trail to implement
            detected_contacts = append_is_contact_to_traj(
                traj=obs,
                fin_rad=15.0,
                margin=3.0,
                pts_num=1024 * 5,
                use_object_centric_frame=use_object_centric_frame,
                seed=0,
                dtype=np.float32,
            )
            # print("test 0", obs[:161,-1])
            # visualize for debug    
            # for t in range(0, obs.shape[0],5):
            #     print(obs[t], 'is_contact =', obs[t, 5])
            #     if use_object_centric_frame:
            #         target_pos = np.array([256, 256, np.pi / 4]) - obs[t, 2:5]  # ox, oy, oa
            #     else:
            #         target_pos = np.array([256, 256, np.pi / 4])
            #     visualize_pusht_obs(obs[t], pusher_radius=15.0, use_object_centric_frame=use_object_centric_frame, xlim=[-256, 256], ylim=[256, -256], target_pos=target_pos)

        episode_ends = root["meta"]["episode_ends"][:]

        self.episodes: list[Dict[str, np.ndarray]] = []
        start = 0
        # print("first episode:", obs[start:episode_ends[0],-1].astype(np.float32, copy=False))
        for end in episode_ends:
            self.episodes.append(
                {
                    "action": actions[start:end].astype(np.float32, copy=False),
                    "obs": obs[start:end].astype(np.float32, copy=False),
                    "is_contact": detected_contacts[start:end].astype(bool, copy=False) if calculate_contact else None,
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
        # print("Getting item", sample["obs"][:,-1])
        return {
            "action": self.normalize_action(sample["action"]),
            "obs": self.normalize_obs(sample["obs"]),
            "is_contact": sample["is_contact"],
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
        rotation = torch.stack(
            [
                torch.stack([cos_theta, -sin_theta], dim=-1),
                torch.stack([sin_theta, cos_theta], dim=-1),
            ],
            dim=-2,
        )
        
        rotated = torch.bmm(rotation, action[:, :2].unsqueeze(-1)).squeeze(-1)
        return rotated + object_xy
    
    def relative_state_to_global(self, obs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Map object-centric observations back into world coordinates."""
        object_xy = obs[:, 2:4]
        object_theta = obs[:, 4]
        cos_theta = torch.cos(object_theta)
        sin_theta = torch.sin(object_theta)
        rotation = torch.stack(
            [
                torch.stack([cos_theta, -sin_theta], dim=-1),
                torch.stack([sin_theta, cos_theta], dim=-1),
            ],
            dim=-2,
        )
        
        rotated = torch.bmm(rotation, state[:, :2].unsqueeze(-1)).squeeze(-1)
        if state.shape[1] > 2:
            # append other state dimensions unchanged
            object_xy_global = rotated + object_xy
            theta_global = state[:, 2:3] + object_theta.unsqueeze(-1)
            return torch.cat([object_xy_global, theta_global], dim=-1)
        else:
            return rotated + object_xy
    
    def global_state_to_relative(self, obs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Convert global states into object-centric observations using the inverse (transposed) rotation."""
        object_xy = obs[:, 2:4]
        object_theta = obs[:, 4]
        cos_theta = torch.cos(object_theta)
        sin_theta = torch.sin(object_theta)
        rotation = torch.stack(
            [
                torch.stack([cos_theta, -sin_theta], dim=-1),
                torch.stack([sin_theta, cos_theta], dim=-1),
            ],
            dim=-2,
        )
        # For orthonormal rotation matrices the inverse is the transpose.
        rotation_inv = rotation.transpose(-1, -2)
        translated = state[:, :2] - object_xy
        state_rel = torch.bmm(rotation_inv, translated.unsqueeze(-1)).squeeze(-1)
        if state.shape[1] > 2:
            # append other state dimensions unchanged            
            theta_rel = state[:, 2:3] - object_theta.unsqueeze(-1)
            return torch.cat([state_rel, theta_rel], dim=-1)
        else:
            return state_rel
    
    def global_action_to_relative(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
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
            dim=-2,
        )
        rotation_inv = rotation.transpose(-1, -2)
        translated = action[:, :2] - object_xy
        return torch.bmm(rotation_inv, translated.unsqueeze(-1)).squeeze(-1)

    def distance(self, states: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """Compute PushT-specific distance between normalised states."""
        pos_diff = states[:, :4] - query[:, :4]
        pos_dist_sq = torch.sum(pos_diff ** 2, dim=1)
        angle_diff = states[:, 4] - query[:, 4]
        abs_angle = torch.abs(angle_diff)
        wrapped = torch.min(abs_angle, 2.0 - abs_angle)
        angle_dist_sq = wrapped ** 2
        return torch.sqrt(pos_dist_sq + angle_dist_sq)


def load_episode_dataset(
    dataset_path: str,
    use_relative_action: bool = False,
    use_object_centric_frame: bool = False,
    calculate_contact: bool = False,
) -> PushTEpisodeDataset:
    return PushTEpisodeDataset(
        dataset_path,
        use_relative_action=use_relative_action,
        use_object_centric_frame=use_object_centric_frame,
        calculate_contact=calculate_contact,
    )


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


if __name__ == "__main__":
    # simple test
    dataset_path = "models/pusht_cchi_v7_replay.zarr.zip"
    use_relative_action = False
    use_object_centric_frame = False
    dataset = PushTEpisodeDataset(dataset_path, use_relative_action=use_relative_action, use_object_centric_frame=use_object_centric_frame, calculate_contact=True)
    