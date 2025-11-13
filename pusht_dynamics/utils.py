import os
import torch
from torch.utils.data import DataLoader, random_split
from typing import Tuple
from torch.utils.tensorboard import SummaryWriter
import re

def make_fast_loader(dataset, batch_size: int = 256, shuffle=True) -> DataLoader:
    num_workers = max(2, (os.cpu_count() or 4) // 2)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=4, drop_last=True
    )


def setup_optimizer(model, lr=1e-3):
    try:
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), fused=True)
    except TypeError:
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    
def make_writer(run_name: str, root: str = "runs") -> SummaryWriter:
    """
    Creates a SummaryWriter at {root}/{run_name}. Example run_name: 'inverse_fp16_bs256'.
    """
    path = os.path.join(root, run_name)
    os.makedirs(path, exist_ok=True)
    return SummaryWriter(log_dir=path)

def get_lr(optimizer) -> float:
    for pg in optimizer.param_groups:
        return float(pg.get("lr", 0.0))
    return 0.0

def split_episodes(base_episode_dataset, val_ratio=0.2, seed=42):
    N = len(base_episode_dataset)
    val_len = int(round(N * val_ratio))
    train_len = N - val_len
    gen = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(base_episode_dataset, [train_len, val_len], generator=gen)
    return train_subset, val_subset

def save_topk(state_dict, run_dir, k: int = 5, fname_prefix="ckpt", metric="val_loss"):
    """
    Save a checkpoint and keep only the top-k with lowest validation loss.

    Args:
        state_dict (dict): checkpoint dict (must include metric and epoch).
        run_dir (str): run directory containing /checkpoints subfolder.
        k (int): how many top checkpoints to keep.
        fname_prefix (str): prefix for saved checkpoint filenames.
        metric (str): key in state_dict used as the loss/metric value.
    """
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # --- current file name ---
    val_loss = float(state_dict.get(metric, float("inf")))
    epoch = int(state_dict.get("epoch", 0))
    fname = f"{fname_prefix}_e{epoch:03d}_loss{val_loss:.6f}.pt"
    path = os.path.join(ckpt_dir, fname)
    torch.save(state_dict, path)

    # --- gather and sort all checkpoints ---
    files = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".pt")]

    def loss_from_name(fn: str) -> float:
        # match 'loss<number>.pt'
        m = re.search(r"loss([0-9eE\+\-\.]+)\.pt", fn)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return float("inf")
        return float("inf")

    # sort ascending (lowest loss first)
    files = sorted(files, key=lambda f: (loss_from_name(f), os.path.getmtime(f)))

    # --- remove extras if > k ---
    if len(files) > k:
        for f in files[k:]:
            try:
                os.remove(f)
            except OSError:
                pass

    return path

def get_plain_state_dict(model):
    # If compiled, prefer the original module’s weights
    if hasattr(model, "_orig_mod"):
        return model._orig_mod.state_dict()
    # If DataParallel, prefer the wrapped module’s weights
    if hasattr(model, "module"):
        return model.module.state_dict()
    return model.state_dict()

