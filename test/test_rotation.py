
import torch
import numpy as np

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

def global_state_to_relative(obs: torch.Tensor) -> torch.Tensor:
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
        translated = obs[:, :2] - object_xy
        return torch.bmm(rotation_inv, translated.unsqueeze(-1)).squeeze(-1)


ob_ = np.array([300.0, 200.0, 320.0, 220.0, np.pi/4])  # global state: pusher x,y; object x,y; object theta
obj_pos_world = ob_[2:4]
obj_rad = ob_[4]
finger_pos = ob_[0:2]

T_W_O = posrad_to_Tmat(obj_pos_world, obj_rad)
T_O_W = np.linalg.inv(T_W_O)
finger_pos_obj = trans_pt_2d(finger_pos, T_O_W)
print(finger_pos_obj)

obs_tensor = torch.from_numpy(ob_.astype(np.float32)).unsqueeze(0)  # [1, obs_dim]
finger_pos_obj_torch = global_state_to_relative(obs_tensor).squeeze(0).numpy()
print(finger_pos_obj_torch)
