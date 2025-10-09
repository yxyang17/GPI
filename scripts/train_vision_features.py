#!/usr/bin/env python3
"""Train the vision encoder + state regressor used by the GPI vision policy and export features."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from pusht.datasets import PushTImageDataset  # noqa: E402
from gpi.vision import create_models  # noqa: E402
from pusht.downloads import ensure_resource  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PushT vision encoder for GPI")
    parser.add_argument(
        "--dataset",
        type=str,
        default="models/pusht_cchi_v7_replay.zarr.zip",
        help="Path to the original PushT replay dataset (.zarr or .zarr.zip). Downloads land in models/ by default.",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--device", type=str, default=None, help="torch device (defaults to cuda if available)."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models/checkpoints",
        help="Directory to store {vision_state_predictor_epoch_XX.ckpt}.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=20,
        help="How often (in epochs) to save a checkpoint.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="torch DataLoader worker processes."
    )
    parser.add_argument(
        "--obs-horizon",
        type=int,
        default=1,
        help="Observation horizon to sample per data point.",
    )
    parser.add_argument(
        "--pred-horizon",
        type=int,
        default=1,
        help="Prediction horizon provided by the dataset.",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=1,
        help="Action horizon provided by the dataset.",
    )
    parser.add_argument(
        "--output-dataset",
        type=str,
        default="models/pusht_cchi_v7_replay_imgs_feature_epoch_200.zarr",
        help="Destination path for the feature dataset (directory or .zarr.zip).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="models/vision_state_predictor_epoch_200.ckpt",
        help="Final checkpoint path to save the trained encoder/regressor.",
    )
    return parser.parse_args()


def build_dataloader(args: argparse.Namespace) -> Tuple[PushTImageDataset, DataLoader]:
    dataset_path = ensure_resource(args.dataset)
    dataset = PushTImageDataset(
        dataset_path=dataset_path,
        pred_horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    return dataset, loader


def create_optimisers(
    encoder: nn.Module, predictor: nn.Module, lr: float
) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    enc_opt = torch.optim.Adam(encoder.parameters(), lr=lr)
    pred_opt = torch.optim.Adam(predictor.parameters(), lr=lr)
    return enc_opt, pred_opt


def save_checkpoint(path: Path, epoch: int, encoder: nn.Module, predictor: nn.Module, loss_value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "loss": loss_value,
            "vision_encoder": encoder.state_dict(),
            "state_predictor": predictor.state_dict(),
        },
        path,
    )
    print(f"[checkpoint] saved {path}")


def open_store(path: str, mode: str):
    import zarr
    if path.endswith(".zip"):
        return zarr.ZipStore(path, mode=mode)
    return zarr.DirectoryStore(path)


def export_feature_dataset(
    args: argparse.Namespace,
    dataset: PushTImageDataset,
    encoder: nn.Module,
    device: torch.device,
) -> None:
    """Run the trained encoder across the replay buffer and persist features."""
    import numpy as np
    import zarr

    encoder.eval()
    src_store = open_store(args.dataset, "r")
    dst_store = open_store(args.output_dataset, "w")
    src_root = zarr.open_group(store=src_store, mode="r")
    dst_root = zarr.open_group(store=dst_store, mode="w")
    data_dst = dst_root.create_group("data")
    meta_dst = dst_root.create_group("meta")

    # Copy demonstration state/action (and optionally images if present) for completeness.
    for key in ["state", "action", "img"]:
        if key in src_root["data"]:
            src_arr = src_root["data"][key]
            dst_arr = data_dst.create_dataset(
                key,
                shape=src_arr.shape,
                chunks=src_arr.chunks,
                dtype=src_arr.dtype,
                compressor=src_arr.compressor,
            )
            dst_arr[:] = src_arr[:]

    if "episode_ends" in src_root["meta"]:
        ep_src = src_root["meta"]["episode_ends"]
        ep_dst = meta_dst.create_dataset(
            "episode_ends",
            shape=ep_src.shape,
            chunks=ep_src.chunks,
            dtype=ep_src.dtype,
            compressor=ep_src.compressor,
        )
        ep_dst[:] = ep_src[:]

    total_samples = data_dst["state"].shape[0]
    feature_dim = encoder(torch.zeros(1, 3, 96, 96, device=device)).shape[-1]
    chunk_size = (
        data_dst["img"].chunks[0]
        if "img" in data_dst and data_dst["img"].chunks is not None
        else min(total_samples, 1024)
    )
    feature_arr = data_dst.create_dataset(
        "img_feature",
        shape=(total_samples, feature_dim),
        chunks=(chunk_size, feature_dim),
        dtype=np.float32,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False,
    )
    idx = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Exporting image features"):
            images = batch["image"].to(device)
            bsz, obs_h = images.shape[:2]
            feats = encoder(images.flatten(end_dim=1))
            feats = feats.view(bsz, obs_h, -1)[:, 0, :]
            feature_arr[idx : idx + bsz] = feats.cpu().numpy()
            idx += bsz

    src_store.close()
    dst_store.close()
    print(f"[export] wrote features to {args.output_dataset}")


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[setup] using device: {device}")
    dataset, dataloader = build_dataloader(args)
    vision_encoder, state_predictor = create_models(device)
    criterion = nn.MSELoss()
    enc_opt, pred_opt = create_optimisers(vision_encoder, state_predictor, args.lr)

    for epoch in range(1, args.epochs + 1):
        vision_encoder.train()
        state_predictor.train()
        epoch_loss = 0.0
        batches = 0
        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in progress:
            images = batch["image"].to(device)  # (B, obs_horizon, 3, H, W)
            states = batch["obs_all"].to(device)  # (B, obs_horizon, state_dim)

            batch_size, obs_horizon = images.shape[:2]
            # Flatten horizon dimension for encoder, then restore.
            features = vision_encoder(images.flatten(end_dim=1))
            features = features.view(batch_size, obs_horizon, -1)

            predicted = state_predictor(features[:, 0, :])
            target = states[:, 0, 2:5]  # object pose: x, y, theta

            loss = criterion(predicted, target)

            enc_opt.zero_grad()
            pred_opt.zero_grad()
            loss.backward()
            enc_opt.step()
            pred_opt.step()

            epoch_loss += loss.item()
            batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        mean_loss = epoch_loss / max(batches, 1)
        print(f"[epoch {epoch}] mean regression loss: {mean_loss:.6f}")
        if args.checkpoint_dir:
            checkpoint_dir = Path(args.checkpoint_dir)
            if epoch % args.save_interval == 0:
                periodic_path = checkpoint_dir / f"vision_state_predictor_epoch_{epoch}.ckpt"
                save_checkpoint(periodic_path, epoch, vision_encoder, state_predictor, mean_loss)

    # Always save the final weights to the requested checkpoint path.
    final_path = Path(args.checkpoint_path)
    save_checkpoint(final_path, args.epochs, vision_encoder, state_predictor, mean_loss)

    # Export the learned image descriptors for downstream usage.
    export_feature_dataset(args, dataset, vision_encoder, device)


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
