# experiments/eval_forward_on_episode.py
import os, glob, torch
import torch.nn as nn
import numpy as np

from pusht_dynamics.models import ForwardDynamics
# from pusht.datasets import load_episode_dataset
from pusht.datasets import load_episode_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def pick_checkpoint(ckpt_dir: str):
    """Pick the best (lowest val loss) checkpoint from a directory."""
    pattern = os.path.join(ckpt_dir, "*.pt")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No checkpoints found under {ckpt_dir}")
    def loss_from_name(path):
        name = os.path.basename(path)
        try:
            return float(name.split("loss")[-1].replace(".pt", ""))
        except Exception:
            return float("inf")
    files.sort(key=loss_from_name)
    return files[0]  # best

def load_forward_model(ckpt_path: str, obs_dim: int, act_dim: int) -> ForwardDynamics:
    model = ForwardDynamics(obs_dim, act_dim).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt.get("model", ckpt)  # support both raw state_dict and wrapped
    model.load_state_dict(state)
    model.eval()
    return model

@torch.no_grad()
def evaluate_episode(model: torch.nn.Module, dataset, epi_idx: int, print_every: int = 20):
    """
    episode: {"obs": np.ndarray[T,...], "action": np.ndarray[Ta, A]} (already normalized by your base dataset)
    Returns dict with per-step MSE and overall MSE.
    """
    episode = dataset[epi_idx]
    obs_np = episode["obs"]
    act_np = episode["action"]
    T, Ta = len(obs_np), len(act_np)

    # Flatten obs for the MLP; if yours are images, replace with your encoder before comparison.
    def flatten(x):
        return x.reshape(x.shape[0], -1) if x.ndim > 1 else x

    o = torch.as_tensor(flatten(obs_np), dtype=torch.float32, device=DEVICE)
    a = torch.as_tensor(act_np,          dtype=torch.float32, device=DEVICE)

    valid_T = min(T - 1, Ta)  # we need (o_t, a_t, o_{t+1})
    mse = nn.MSELoss(reduction="none")

    pred_next_list = []

    for t in range(valid_T):
        o_t   = o[t].unsqueeze(0)           # [1, obs_dim]
        a_t   = a[t].unsqueeze(0)           # [1, act_dim]
        gt_np = o[t+1].unsqueeze(0)         # [1, obs_dim]

        pred_next = model(o_t, a_t)         # [1, obs_dim]
        pred_next_list.append(pred_next.detach().cpu().numpy())
        diff = mse(pred_next, gt_np).mean(dim=1)  # scalar per sample
        # per_step_mse.append(diff.item())

        if (t % print_every) == 0:
            print(f"t={t:04d}: MSE={diff.item():.6f}")

    per_step_mse = []
        
    # Convert predictions to numpy for visualization
    pred_next_array = np.concatenate(pred_next_list, axis=0)  # [valid_T, obs_dim]
    
    # Create visualization
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot action trajectory
    act_np_plot = act_np[:valid_T]
    act_np_plot = dataset.unnormalize_action(act_np_plot)
    obs_np = dataset.unnormalize_obs(obs_np[:valid_T])
    pred_next_array = dataset.unnormalize_obs(pred_next_array)

    # Plot 1: Object trajectories (ground truth vs predicted)
    axes[0].plot(obs_np[:, 0], obs_np[:, 1], 'b-', linewidth=2, label='GT Object 1', alpha=0.7)
    axes[0].plot(obs_np[:, 2], obs_np[:, 3], 'g-', linewidth=2, label='GT Object 2', alpha=0.7)
    axes[0].plot(pred_next_array[:, 0], pred_next_array[:, 1], 'r--', linewidth=2, label='Pred Object 1')
    axes[0].plot(pred_next_array[:, 2], pred_next_array[:, 3], 'm--', linewidth=2, label='Pred Object 2')
    axes[0].scatter(obs_np[0, 0], obs_np[0, 1], c='blue', s=100, marker='o', zorder=5)
    axes[0].scatter(obs_np[0, 2], obs_np[0, 3], c='green', s=100, marker='o', zorder=5)
    axes[0].scatter(obs_np[-1, 0], obs_np[-1, 1], c='blue', s=100, marker='x', zorder=5)
    axes[0].scatter(obs_np[-1, 2], obs_np[-1, 3], c='green', s=100, marker='x', zorder=5)
    axes[0].set_title('Object Trajectories (Ground Truth vs Predicted)')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].legend(loc='best')
    axes[0].grid(True)
    axes[0].axis('equal')
    
    # Plot 2: Action trajectory
    axes[1].plot(act_np_plot[:, 0], act_np_plot[:, 1], 'b-', linewidth=2, label='Action Trajectory')
    axes[1].scatter(act_np_plot[0, 0], act_np_plot[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    axes[1].scatter(act_np_plot[-1, 0], act_np_plot[-1, 1], c='red', s=100, marker='x', label='End', zorder=5)
    axes[1].set_title('Action Trajectory (X-Y Coordinates)')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].legend(loc='best')
    axes[1].grid(True)
    axes[1].axis('equal')
    
    plt.tight_layout()
    # plt.savefig(os.path.join(run_dir, "eval", f"epi{epi_idx}_trajectories.png"))
    plt.show()

    per_step_mse = np.array(per_step_mse, dtype=np.float64)
    overall = float(per_step_mse.mean()) if len(per_step_mse) else float("nan")
    print(f"\nEpisode summary: steps={len(per_step_mse)}, overall MSE={overall:.6f}")
    return {
        "per_step_mse": per_step_mse,
        "overall_mse": overall,
    }

def main():
    # ---- config ----
    dataset_path = "models/pusht_cchi_v7_replay.zarr.zip"
    use_relative_action = False
    # Point to your run directory created by the training script:
    run_dir = "runs/forward_fp16_bs256_20251113_000413"   # <-- set this
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    ckpt_path = pick_checkpoint(ckpt_dir)  # picks the best by filename loss
    print(f"Using checkpoint: {ckpt_path}")

    # ---- load a base episode dataset (normalized already) ----
    base_ds = load_episode_dataset(dataset_path, use_relative_action=use_relative_action)

    # Pick one episode index to evaluate:
    epi_idx = 0  # change as needed
    ep = base_ds[epi_idx]

    # ---- infer dims from this episode ----
    obs_dim = int(np.prod(ep["obs"][0].shape))  # flatten for the MLP version
    act_dim = int(np.prod(ep["action"][0].shape))
    print(f"Episode {epi_idx}: obs_dim={obs_dim}, act_dim={act_dim}")

    # ---- load model and evaluate ----
    model = load_forward_model(ckpt_path, obs_dim, act_dim)
    results = evaluate_episode(model, base_ds, epi_idx, print_every=50)

    # Optional: save per-step MSE to disk
    out_dir = os.path.join(run_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"epi{epi_idx:04d}_per_step_mse.npy"), results["per_step_mse"])
    with open(os.path.join(out_dir, f"epi{epi_idx:04d}_summary.txt"), "w") as f:
        f.write(f'overall_mse: {results["overall_mse"]:.6f}\n')

if __name__ == "__main__":
    main()
