# jacobian_demo.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datetime import datetime
from tqdm import tqdm
from pusht.datasets import load_episode_dataset

# ----------------------------
# Utilities
# ----------------------------
def wrap_angle(theta):
    """wrap to [-pi, pi]"""
    return (theta + np.pi) % (2*np.pi) - np.pi

def angle_diff(theta_next, theta):
    """wrapped difference"""
    return wrap_angle(theta_next - theta)

def rotate_into_object_frame(vec_xy, theta):
    """
    vec_xy: (..., 2) in world frame
    theta:  (...,) object orientation
    returns vec in object frame
    """
    c = np.cos(theta)
    s = np.sin(theta)
    # R(-theta) * v
    x =  c * vec_xy[..., 0] + s * vec_xy[..., 1]
    y = -s * vec_xy[..., 0] + c * vec_xy[..., 1]
    return np.stack([x, y], axis=-1)

# ----------------------------
# Dataset wrapper
# ----------------------------
class JacobianDataset(Dataset):
    """
    Expects numpy arrays:
      agent_xy_t:  (N,2)
      agent_xy_tp1:(N,2)
      obj_xy_t:    (N,2)
      obj_th_t:    (N,)
      obj_xy_tp1:  (N,2)
      obj_th_tp1:  (N,)
    """
    def __init__(self, agent_xy_t, agent_xy_tp1, obj_xy_t, obj_th_t, obj_xy_tp1, obj_th_tp1):
        self.agent_xy_t   = agent_xy_t.astype(np.float32)
        self.agent_xy_tp1 = agent_xy_tp1.astype(np.float32)
        self.obj_xy_t     = obj_xy_t.astype(np.float32)
        self.obj_th_t     = obj_th_t.astype(np.float32)
        self.obj_xy_tp1   = obj_xy_tp1.astype(np.float32)
        self.obj_th_tp1   = obj_th_tp1.astype(np.float32)

        # Action as displacement u = a_{t+1}-a_t
        self.u = (self.agent_xy_tp1 - self.agent_xy_t).astype(np.float32)  # (N,2)

        # Task delta: Δy = [Δx_obj, Δy_obj, Δtheta]
        dxy = (self.obj_xy_tp1 - self.obj_xy_t).astype(np.float32)  # (N,2)
        dth = angle_diff(self.obj_th_tp1, self.obj_th_t).astype(np.float32)  # (N,)
        self.dy = np.concatenate([dxy, dth[:, None]], axis=1).astype(np.float32)  # (N,3)

        # Features x_t (general-ish for planar tasks):
        # - relative agent position in object frame
        rel_world = (self.agent_xy_t - self.obj_xy_t).astype(np.float32)  # (N,2)
        rel_obj = rotate_into_object_frame(rel_world, self.obj_th_t)      # (N,2)

        # - object orientation as sin/cos (avoid angle discontinuity)
        sin_th = np.sin(self.obj_th_t)[:, None].astype(np.float32)
        cos_th = np.cos(self.obj_th_t)[:, None].astype(np.float32)

        # Feature vector: [rel_obj_x, rel_obj_y, sin(theta), cos(theta)]
        self.x = np.concatenate([rel_obj, sin_th, cos_th], axis=1).astype(np.float32)  # (N,4)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.x[idx]),
            torch.from_numpy(self.u[idx]),
            torch.from_numpy(self.dy[idx]),
        )

# ----------------------------
# Jacobian model: x -> J(x) in R^{3x2}
# ----------------------------
class JacobianNet(nn.Module):
    def __init__(self, in_dim=4, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3*2),
        )

    def forward(self, x):
        # x: (B,in_dim)
        J_flat = self.mlp(x)         # (B,6)
        J = J_flat.view(-1, 3, 2)    # (B,3,2)
        return J

# ----------------------------
# Train + quick test
# ----------------------------
def train_jacobian_model(dataset, epochs=20, batch_size=256, lr=1e-3, weight_decay=1e-5, device="cpu"):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model = JacobianNet(in_dim=dataset.x.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for ep in range(1, epochs+1):
        model.train()
        total = 0.0
        for x, u, dy in loader:
            x, u, dy = x.to(device), u.to(device), dy.to(device)

            J = model(x)                        # (B,3,2)
            dy_pred = torch.bmm(J, u.unsqueeze(-1)).squeeze(-1)  # (B,3)

            # Weighted loss trick (optional): ignore tiny-action samples
            # (helps when your dataset has many "move agent, object static" frames)
            u_norm = torch.linalg.norm(u, dim=1)  # (B,)
            w = torch.sigmoid((u_norm - 0.01) / 0.01).detach()   # tune thresholds if needed
            loss = (w[:, None] * (dy_pred - dy)**2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        print(f"epoch {ep:03d} | loss {total/len(loader):.6f}")

    return model

@torch.no_grad()
def evaluate_and_show_control_step(model, sample_x, sample_u, sample_dy, rho=1e-3):
    """
    Demonstrate:
      1) predict dy from J(x)u
      2) compute u to realize a desired dy via damped least squares
    """
    model.eval()
    x = sample_x.unsqueeze(0)  # (1,in_dim)
    u = sample_u.unsqueeze(0)  # (1,2)
    dy = sample_dy.unsqueeze(0) # (1,3)

    J = model(x)[0]  # (3,2)
    dy_pred = (J @ u[0]).cpu().numpy()

    print("True dy     :", dy[0].cpu().numpy())
    print("Pred dy(Ju) :", dy_pred)

    # Suppose we want some desired dy_des (e.g., from object-centric GPI step)
    dy_des = dy[0].clone()
    # Example tweak: ask for slightly more rotation
    dy_des[2] += 0.05
    print("Desired dy  :", dy_des.cpu().numpy())

    # Solve u* = argmin ||J u - dy_des||^2 + rho||u||^2
    # closed form: u = (J^T J + rho I)^-1 J^T dy_des
    JTJ = J.T @ J
    A = JTJ + rho * torch.eye(2)
    b = J.T @ dy_des
    u_star = torch.linalg.solve(A, b)
    print("u* (DLS)    :", u_star.cpu().numpy())

# ----------------------------
# Example usage (replace with your real arrays)
# ----------------------------
if __name__ == "__main__":
    # TODO: replace these with your real arrays loaded from file
    # Here we generate a tiny synthetic placeholder dataset just so the script runs.
    # N = 5000
    # rng = np.random.default_rng(0)

    # agent_xy_t   = rng.normal(size=(N,2)).astype(np.float32)
    # u_true       = 0.05 * rng.normal(size=(N,2)).astype(np.float32)
    # agent_xy_tp1 = agent_xy_t + u_true

    # obj_xy_t = rng.normal(size=(N,2)).astype(np.float32)
    # obj_th_t = rng.uniform(-np.pi, np.pi, size=(N,)).astype(np.float32)

    # # Fake object response: depends on relative x and action (just for demo)
    # rel = agent_xy_t - obj_xy_t
    # dxy = 0.2 * u_true + 0.05 * np.stack([rel[:,0], rel[:,1]], axis=1) * (np.linalg.norm(u_true, axis=1)[:,None])
    # dth = 0.5 * (rel[:,0]*u_true[:,1] - rel[:,1]*u_true[:,0])  # torque-like cross product

    # obj_xy_tp1 = obj_xy_t + dxy.astype(np.float32)
    # obj_th_tp1 = (obj_th_t + dth.astype(np.float32))

    # ---- config ----
    dataset_path = "models/pusht_cchi_v7_replay.zarr.zip"
    use_relative_action = True
    use_object_centric_frame = True

    base = load_episode_dataset(dataset_path, use_relative_action=use_relative_action)


    ds = JacobianDataset(agent_xy_t, agent_xy_tp1, obj_xy_t, obj_th_t, obj_xy_tp1, obj_th_tp1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = train_jacobian_model(ds, epochs=10, device=device)

    # show a random sample control step
    idx = 0
    x, u, dy = ds[idx]
    evaluate_and_show_control_step(model.to("cpu"), x, u, dy)
