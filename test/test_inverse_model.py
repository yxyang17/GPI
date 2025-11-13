# experiments/eval_inverse_on_episode.py
import os, glob, torch, re
import torch.nn as nn
import numpy as np

from pusht_dynamics.models import InverseDynamics
from pusht.datasets import load_episode_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

# def _clean_state_dict(sd):
#     # unwrap torch.compile("_orig_mod.") / DataParallel("module.")
#     if any(k.startswith("_orig_mod.") for k in sd.keys()):
#         sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
#     if any(k.startswith("module.") for k in sd.keys()):
#         sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
#     return sd

def load_inverse_model(ckpt_path: str, obs_dim: int, act_dim: int) -> InverseDynamics:
    model = InverseDynamics(obs_dim, act_dim).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt.get("model", ckpt)  # support raw or wrapped
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

@torch.no_grad()
def evaluate_episode(model: torch.nn.Module, dataset, epi_idx: int, print_every: int = 20):
    """
    Uses (o_{t-1}, o_t, o_{t+1}) -> a_t over one episode from the dataset.
    Returns dict with per-step MSE and overall MSE.
    """
    episode = dataset[epi_idx]  # already normalized by base dataset
    obs_np = episode["obs"]
    act_np = episode["action"]
    T, Ta = len(obs_np), len(act_np)

    # valid window needs t-1, t, t+1 and a_t
    t_start = 1
    t_end_exclusive = min(T - 1, Ta)  # ensure t+1 exists and a_t exists
    if t_end_exclusive <= t_start:
        raise ValueError("Episode too short for inverse dynamics evaluation.")

    # flatten observations for MLP (swap with encoder if you used CNNs)
    def flatten(x):
        return x.reshape(x.shape[0], -1) if x.ndim > 1 else x

    o = torch.as_tensor(flatten(obs_np), dtype=torch.float32, device=DEVICE)  # [T, obs_dim]
    a = torch.as_tensor(act_np,          dtype=torch.float32, device=DEVICE)  # [Ta, act_dim]

    mse = nn.MSELoss(reduction="none")

    pred_act_list = []
    per_step_mse = []

    for t in range(t_start, t_end_exclusive):
        o_prev = o[t-1].unsqueeze(0)  # [1, obs_dim]
        o_curr = o[t  ].unsqueeze(0)
        o_next = o[t+1].unsqueeze(0)
        a_t    = a[t  ].unsqueeze(0)  # [1, act_dim]

        pred_a = model(o_prev, o_curr, o_next)  # [1, act_dim]
        pred_act_list.append(pred_a.detach().cpu().numpy())

        diff = mse(pred_a, a_t).mean(dim=1)  # scalar per sample
        per_step_mse.append(diff.item())

        if ((t - t_start) % print_every) == 0:
            print(f"t={t:04d}: action MSE={diff.item():.6f}")

    # numpy arrays for plotting / metrics
    pred_actions = np.concatenate(pred_act_list, axis=0)                     # [N, act_dim]
    gt_actions   = a[t_start:t_end_exclusive].detach().cpu().numpy()         # [N, act_dim]
    per_step_mse = np.asarray(per_step_mse, dtype=np.float64)
    overall = float(per_step_mse.mean()) if len(per_step_mse) else float("nan")
    print(f"\nEpisode summary: steps={len(per_step_mse)}, overall action MSE={overall:.6f}")

    # ---- Visualization: predicted vs GT actions ----
    # Unnormalize for human-readable plots if dataset exposes unnormalize_action
    if hasattr(dataset, "unnormalize_action"):
        gt_plot   = dataset.unnormalize_action(gt_actions)
        pred_plot = dataset.unnormalize_action(pred_actions)
    else:
        gt_plot, pred_plot = gt_actions, pred_actions

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: XY trajectory (first two action dims)
    if gt_plot.shape[1] >= 2:
        axes[0].plot(gt_plot[:, 0],   gt_plot[:, 1],  'b-',  linewidth=2, label='GT', alpha=0.8)
        axes[0].plot(pred_plot[:, 0], pred_plot[:, 1], 'r--', linewidth=2, label='Pred', alpha=0.8)
        axes[0].scatter(gt_plot[0, 0],   gt_plot[0, 1],   c='green', s=80, marker='o', label='Start', zorder=5)
        axes[0].scatter(gt_plot[-1, 0],  gt_plot[-1, 1],  c='red',   s=80, marker='x', label='End',   zorder=5)
        axes[0].set_title('Action Trajectory (XY)')
        axes[0].set_xlabel('Action X'); axes[0].set_ylabel('Action Y')
        axes[0].axis('equal'); axes[0].grid(True); axes[0].legend(loc='best')
    else:
        axes[0].plot(gt_plot[:, 0], 'b-', linewidth=2, label='GT')
        axes[0].plot(pred_plot[:, 0], 'r--', linewidth=2, label='Pred')
        axes[0].set_title('Action dim0 over time')
        axes[0].set_xlabel('t'); axes[0].set_ylabel('Action'); axes[0].grid(True); axes[0].legend()

    # Right: per-dimension time series overlay
    steps = np.arange(gt_plot.shape[0])
    for d in range(gt_plot.shape[1]):
        axes[1].plot(steps, gt_plot[:, d],  label=f'GT d{d}', linewidth=2)
        axes[1].plot(steps, pred_plot[:, d], '--', label=f'Pred d{d}', linewidth=2)
    axes[1].set_title('Action per Dimension')
    axes[1].set_xlabel('t'); axes[1].set_ylabel('Action'); axes[1].grid(True); axes[1].legend(ncol=2)

    plt.tight_layout()
    # plt.savefig(os.path.join(run_dir, "eval_inverse", f"epi{epi_idx}_actions.png"))
    plt.show()

    return {
        "per_step_mse": per_step_mse,
        "overall_mse": overall,
        "pred_actions": pred_actions,
        "gt_actions": gt_actions,
    }

def main():
    # ---- config ----
    dataset_path = "models/pusht_cchi_v7_replay.zarr.zip"
    use_relative_action = False
    run_dir = "runs/inverse_fp16_bs256_20251113_005904"   # <-- set to your inverse run
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    ckpt_path = pick_checkpoint(ckpt_dir)
    print(f"Using checkpoint: {ckpt_path}")

    # ---- load dataset (normalized) ----
    base_ds = load_episode_dataset(dataset_path, use_relative_action=use_relative_action)

    # which episode
    epi_idx = 0

    # ---- infer dims from that episode ----
    ep = base_ds[epi_idx]
    obs_dim = int(np.prod(ep["obs"][0].shape))
    act_dim = int(np.prod(ep["action"][0].shape))

    print(f"Episode {epi_idx}: obs_dim={obs_dim}, act_dim={act_dim}")

    # ---- load model & evaluate ----
    model = load_inverse_model(ckpt_path, obs_dim, act_dim)
    results = evaluate_episode(model, base_ds, epi_idx, print_every=50)

    # Optional: save metrics
    out_dir = os.path.join(run_dir, "eval_inverse")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"epi{epi_idx:04d}_per_step_action_mse.npy"), results["per_step_mse"])
    with open(os.path.join(out_dir, f"epi{epi_idx:04d}_summary.txt"), "w") as f:
        f.write(f'overall_action_mse: {results["overall_mse"]:.6f}\n')

if __name__ == "__main__":
    main()
