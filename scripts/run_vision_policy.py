#!/usr/bin/env python3
"""CLI entry point for the vision-based GPI policy."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpi.policies.base import GPIConfig
from gpi.policies.vision import VisionGPIPolicy, VisionPolicyConfig
from pusht.datasets import load_episode_dataset
from pusht.evaluation import VisionEvaluator
from pusht.downloads import ensure_resource


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the vision-based GPI policy on PushT")
    parser.add_argument("--dataset", type=str, default="models/pusht_cchi_v7_replay_imgs_feature_epoch_200.zarr", help="Path to PushT vision dataset (downloaded into models/ by default)")
    parser.add_argument("--vision-checkpoint", type=str, default="models/vision_state_predictor_epoch_200.ckpt", help="Checkpoint containing vision encoder and predictor weights (stored under models/)")
    parser.add_argument("--seed", type=int, default=500, help="Environment seed")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum evaluation steps")
    parser.add_argument("--k-neighbors", type=int, default=3, help="Number of neighbours for smoothing")
    parser.add_argument("--action-horizon", type=int, default=8, help="Plan horizon in steps")
    parser.add_argument("--subset-size", type=int, default=None, help="Optional random subset of database states")
    parser.add_argument("--dynamic-subset", action="store_true", help="Resample subset periodically during rollout")
    parser.add_argument("--subset-change-interval", type=int, default=50, help="Number of steps between subset resamples")
    parser.add_argument("--batch-size", type=int, default=500_000, help="GPU batch size for distance queries")
    parser.add_argument("--device", type=str, default=None, help="torch device override (cpu/cuda)")
    parser.add_argument("--obs-noise-std", type=float, default=0.01, help="Initial observation noise stddev")
    parser.add_argument("--noise-decay", type=float, default=0.99, help="Noise decay factor per step")
    parser.add_argument("--min-noise-std", type=float, default=0.000, help="Lower bound on observation noise")
    parser.add_argument("--random-seed", type=int, default=0, help="Random seed for subset sampling")
    parser.add_argument("--memory-length", type=int, default=500, help="Length of recent-state memory to avoid loops")
    parser.add_argument("--fixed-lambda1", type=float, default=1.0, help="Optional fixed lambda1 weighting")
    parser.add_argument("--fixed-lambda2", type=float, default=1.0, help="Optional fixed lambda2 weighting")
    parser.add_argument("--action-smoothing", type=float, default=0.0, help="Weight for first-order action smoothing (0 disables smoothing)")
    parser.add_argument("--video-path", type=str, default=None, help="Optional output path for rendered video")
    parser.add_argument("--no-video", action="store_true", help="Disable video recording")
    parser.add_argument("--no-live-render", dest="live_render", action="store_false", help="Disable live window display during rollout")
    parser.add_argument("--quiet", action="store_true", help="Disable progress bar")
    parser.add_argument("--relative-action", dest="use_relative_action", action="store_true", default=False)
    parser.add_argument("--disable-noise", dest="enable_obs_noise", action="store_false", default=True)
    parser.set_defaults(live_render=True)
    return parser.parse_args()
def main() -> None:
    args = parse_args()
    ensure_resource(args.dataset)
    ensure_resource(args.vision_checkpoint)
    base_config = GPIConfig(
        dataset_path=args.dataset,
        dataset_loader=load_episode_dataset,
        use_relative_action=args.use_relative_action,
        k_neighbors=args.k_neighbors,
        obs_noise_std=args.obs_noise_std,
        enable_obs_noise=args.enable_obs_noise,
        noise_decay_rate=args.noise_decay,
        min_noise_std=args.min_noise_std,
        device=args.device,
        batch_size=args.batch_size,
        subset_size=args.subset_size,
        random_seed=args.random_seed,
        action_horizon=args.action_horizon,
        fixed_lambda1=args.fixed_lambda1,
        fixed_lambda2=args.fixed_lambda2,
        action_smoothing=args.action_smoothing,
    )
    vision_config = VisionPolicyConfig(**base_config.__dict__, vision_checkpoint=args.vision_checkpoint, dynamic_subset=args.dynamic_subset, subset_change_interval=args.subset_change_interval, memory_length=args.memory_length)
    policy = VisionGPIPolicy(vision_config)
    evaluator = VisionEvaluator(env_seed=args.seed, max_steps=args.max_steps)
    results = evaluator.evaluate(
        policy,
        render_video=not args.no_video,
        video_path=args.video_path,
        verbose=not args.quiet,
        live_display=args.live_render,
    )
    print("\nEvaluation complete")
    print(f"  Steps: {results.steps}")
    print(f"  Max reward: {results.max_reward:.3f}")
    if isinstance(results.inference_stats, dict):
        policy_stats = results.inference_stats.get("policy", results.inference_stats)
        print(f"  Policy inference mean (ms): {policy_stats['mean_ms']:.2f}")
    if isinstance(results.inference_stats, dict) and "vision" in results.inference_stats:
        vision_stats = results.inference_stats["vision"]
        print(f"  Vision inference mean (ms): {vision_stats['mean_ms']:.2f}")
    if results.video_path:
        print(f"  Video saved to: {results.video_path}")


if __name__ == "__main__":
    main()
