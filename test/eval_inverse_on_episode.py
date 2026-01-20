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

def load_inverse_model(ckpt_path: str, obs_dim: int, act_dim: int) -> InverseDynamics:
    model = InverseDynamics(obs_dim, act_dim).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt.get("model", ckpt)  # support raw or wrapped
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

@torch.no_grad()
def evaluate_episode(
    model: torch.nn.Module,
    dataset,
    epi_idx: int,
    print_every: int = 20,
    # ---- NEW: contact settings ----
    AGENT_XY_IDXS=(0, 1),   # indices in obs vector for agent xy in object frame
    CONTACT_TH=0.03,        # contact threshold (tune to your object size / units)
    PLOT_CONTACT_ERROR=True
):
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

    # ---- NEW: store distance/contact per step ----
    distances = []
    contact_flags = []

    def agent_distance_from_obs(o_t_1d: torch.Tensor) -> torch.Tensor:
        """
        o_t_1d: [obs_dim] tensor.
        Returns scalar distance of agent to object (in object frame) using AGENT_XY_IDXS.
        """
        ax = o_t_1d[AGENT_XY_IDXS[0]]
        ay = o_t_1d[AGENT_XY_IDXS[1]]
        return torch.sqrt(ax * ax + ay * ay)

    def object_moved_between_obs(o_prev: torch.Tensor, o_curr: torch.Tensor) -> bool:
        """
        Heuristic to determine if object moved between two observations.
        Can be used to filter out no-contact frames if needed.
        """
        obj_prev = o_prev[2:]  # fix me: hardcoded for two-object pusht
        obj_curr = o_curr[2:]
        dist = torch.linalg.norm(obj_curr - obj_prev)
        return dist.item() > 0.01  # tune threshold if needed
    
    for t in range(t_start, t_end_exclusive):
        o_prev = o[t-1].unsqueeze(0)  # [1, obs_dim]
        o_curr = o[t  ].unsqueeze(0)
        o_next = o[t+1].unsqueeze(0)
        a_t    = a[t  ].unsqueeze(0)  # [1, act_dim]

        pred_a = model(o_prev, o_curr, o_next)  # [1, act_dim]
        pred_act_list.append(pred_a.detach().cpu().numpy())

        diff = mse(pred_a, a_t).mean(dim=1)  # scalar per sample
        per_step_mse.append(diff.item())

        # ---- NEW: contact heuristic ----
        dist = agent_distance_from_obs(o_curr[0])
        distances.append(dist.item())
        object_moved = object_moved_between_obs(o_curr[0], o_next[0])
        contact_flags.append(object_moved)

        if ((t - t_start) % print_every) == 0:
            print(f"t={t:04d}: action MSE={diff.item():.6f} | dist={dist.item():.4f} | contact={dist.item() < CONTACT_TH}")

    # numpy arrays for plotting / metrics
    pred_actions = np.concatenate(pred_act_list, axis=0)                     # [N, act_dim]
    gt_actions   = a[t_start:t_end_exclusive].detach().cpu().numpy()         # [N, act_dim]
    per_step_mse = np.asarray(per_step_mse, dtype=np.float64)
    overall = float(per_step_mse.mean()) if len(per_step_mse) else float("nan")
    print(f"\nEpisode summary: steps={len(per_step_mse)}, overall action MSE={overall:.6f}")

    distances = np.asarray(distances, dtype=np.float64)
    contact_flags = np.asarray(contact_flags, dtype=bool)

    # ---- Visualization: predicted vs GT actions ----
    # Unnormalize for human-readable plots if dataset exposes unnormalize_action
    if hasattr(dataset, "unnormalize_action"):
        gt_plot   = dataset.unnormalize_action(gt_actions)
        pred_plot = dataset.unnormalize_action(pred_actions)
    else:
        gt_plot, pred_plot = gt_actions, pred_actions

    if dataset.use_relative_action:
        pred_plot = dataset.relative_action_to_global(
            o[1:-1].detach().cpu(), torch.tensor(pred_plot)
        ).numpy()
        gt_plot = dataset.relative_action_to_global(
            o[1:-1].detach().cpu(), torch.tensor(gt_plot)
        ).numpy()

    import matplotlib.pyplot as plt

    # ---- Figure 1: predicted vs GT actions ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: XY trajectory (first two action dims)
    if gt_plot.shape[1] >= 2:
        axes[0].plot(gt_plot[:, 0],   gt_plot[:, 1],  'b-',  linewidth=2, label='GT', alpha=0.8)
        axes[0].plot(pred_plot[:, 0], pred_plot[:, 1], 'r--', linewidth=2, label='Pred', alpha=0.8)
        axes[0].scatter(gt_plot[0, 0],   gt_plot[0, 1],   c='green', s=80, marker='o', label='Start', zorder=5)
        axes[0].scatter(gt_plot[-1, 0],  gt_plot[-1, 1],  c='red',   s=80, marker='x', label='End',   zorder=5)
        axes[0].scatter(pred_plot[0, 0],   pred_plot[0, 1],   c='green', s=80, marker='o', zorder=5)
        axes[0].scatter(pred_plot[-1, 0],  pred_plot[-1, 1],  c='red',   s=80, marker='x', zorder=5)
        axes[0].set_title('Action Trajectory (XY)')
        axes[0].set_xlabel('Action X'); axes[0].set_ylabel('Action Y')
        axes[0].invert_yaxis()  # THIS LINE FLIPS THE Y-AXIS
        axes[0].axis('equal'); axes[0].grid(True); axes[0].legend(loc='best')
    else:
        axes[0].plot(gt_plot[:, 0], 'b-', linewidth=2, label='GT')
        axes[0].plot(pred_plot[:, 0], 'r--', linewidth=2, label='Pred')
        axes[0].set_title('Action dim0 over time')
        axes[0].set_xlabel('t'); axes[0].set_ylabel('Action'); axes[0].grid(True); axes[0].legend()

    # Right: per-dimension time series overlay
    steps = np.arange(gt_plot.shape[0])
    colors = ["b", "g"]
    for d in range(gt_plot.shape[1]):
        axes[1].plot(steps, gt_plot[:, d],  colors[d % len(colors)], label=f'GT d{d}', linewidth=2)
        axes[1].plot(steps, pred_plot[:, d], colors[d % len(colors)]+'--', label=f'Pred d{d}', linewidth=2)
    axes[1].set_title('Action per Dimension')
    axes[1].set_xlabel('t'); axes[1].set_ylabel('Action'); axes[1].grid(True); axes[1].legend(ncol=2)

    plt.tight_layout()
    plt.show()

    # ---- NEW Figure 2: error vs contact/no-contact ----
    if PLOT_CONTACT_ERROR:
        # A) Error over time with contact coloring
        fig2 = plt.figure(figsize=(12, 4))
        ts = np.arange(len(per_step_mse))

        plt.plot(ts, per_step_mse, 'k-', alpha=0.35, linewidth=1, label='Action MSE')

        # mark contact/no-contact
        plt.scatter(ts[contact_flags], per_step_mse[contact_flags], s=12, label='Contact')
        plt.scatter(ts[~contact_flags], per_step_mse[~contact_flags], s=12, label='No Contact')

        plt.yscale('log')
        plt.xlabel('timestep')
        plt.ylabel('action MSE (log scale)')
        plt.title(f'Inverse Error vs Contact (CONTACT_TH={CONTACT_TH})')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # B) Error vs distance to object (even more diagnostic)
        fig3 = plt.figure(figsize=(6, 4))
        plt.scatter(distances, per_step_mse, s=12, alpha=0.7)
        plt.axvline(CONTACT_TH, linestyle='--', label='contact threshold')

        plt.yscale('log')
        plt.xlabel('agent-object distance (object frame)')
        plt.ylabel('action MSE (log scale)')
        plt.title('Inverse Error vs Distance to Object')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Print numeric summary
        if contact_flags.any():
            print("Mean MSE (contact)   :", float(per_step_mse[contact_flags].mean()))
        if (~contact_flags).any():
            print("Mean MSE (no contact):", float(per_step_mse[~contact_flags].mean()))

    return {
        "per_step_mse": per_step_mse,
        "overall_mse": overall,
        "pred_actions": pred_actions,
        "gt_actions": gt_actions,
        "distances": distances,
        "contact_flags": contact_flags,
    }

def main():
    # ---- config ----
    dataset_path = "models/pusht_cchi_v7_replay.zarr.zip"
    use_relative_action = True
    use_object_centric_frame = True

    run_dir = "runs/inverse_fp16_bs256_20251113_005904"
    run_dir = "runs/inverse_fp16_bs256_20251113_183159"
    run_dir = "runs/inverse_re_bs512_20251210_235709"
    run_dir = "runs/inverse_re_bs512_20251215_225843"
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    ckpt_path = pick_checkpoint(ckpt_dir)
    print(f"Using checkpoint: {ckpt_path}")

    # ---- load dataset (normalized) ----
    base_ds = load_episode_dataset(
        dataset_path,
        use_relative_action=use_relative_action,
        use_object_centric_frame=use_object_centric_frame
    )

    # which episode
    epi_idx = 10

    # ---- infer dims from that episode ----
    ep = base_ds[epi_idx]
    obs_dim = int(np.prod(ep["obs"][0].shape))
    act_dim = int(np.prod(ep["action"][0].shape))
    print(f"Episode {epi_idx}: obs_dim={obs_dim}, act_dim={act_dim}")

    # ---- load model & evaluate ----
    model = load_inverse_model(ckpt_path, obs_dim, act_dim)

    # ---- NEW: set these two based on your obs layout / units ----
    AGENT_XY_IDXS = (0, 1)   # if agent xy in object frame is not obs[0:2], change this
    CONTACT_TH = 0.03        # tune this (object size in your normalized units)

    results = evaluate_episode(
        model,
        base_ds,
        epi_idx,
        print_every=50,
        AGENT_XY_IDXS=AGENT_XY_IDXS,
        CONTACT_TH=CONTACT_TH,
        PLOT_CONTACT_ERROR=True
    )

    # Optional: save metrics
    out_dir = os.path.join(run_dir, "eval_inverse")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"epi{epi_idx:04d}_per_step_action_mse.npy"), results["per_step_mse"])
    np.save(os.path.join(out_dir, f"epi{epi_idx:04d}_distances.npy"), results["distances"])
    np.save(os.path.join(out_dir, f"epi{epi_idx:04d}_contact_flags.npy"), results["contact_flags"])

    with open(os.path.join(out_dir, f"epi{epi_idx:04d}_summary.txt"), "w") as f:
        f.write(f'overall_action_mse: {results["overall_mse"]:.6f}\n')
        if results["contact_flags"].any():
            f.write(f'mean_mse_contact: {results["per_step_mse"][results["contact_flags"]].mean():.6f}\n')
        if (~results["contact_flags"]).any():
            f.write(f'mean_mse_no_contact: {results["per_step_mse"][~results["contact_flags"]].mean():.6f}\n')
        f.write(f'contact_threshold: {CONTACT_TH}\n')
        f.write(f'agent_xy_idxs: {AGENT_XY_IDXS}\n')

if __name__ == "__main__":
    main()
