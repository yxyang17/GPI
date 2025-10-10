#!/usr/bin/env python3
"""Batch runner for vision-based GPI evaluations with concise reporting."""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "models" / "pusht_cchi_v7_replay_imgs_feature_epoch_200.zarr"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "vision_state_predictor_epoch_200.ckpt"
LOG_PATH = PROJECT_ROOT / "results" / "logs" / "vision_policy_runs.log"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from gpi.policies.base import GPIConfig
from gpi.policies.vision import VisionGPIPolicy, VisionPolicyConfig
from pusht.datasets import load_episode_dataset
from pusht.evaluation import VisionEvaluator


@dataclass(frozen=True)
class VisionEvaluationConfig:
    name: str
    seed: int
    k_neighbors: int
    action_horizon: int
    obs_noise_std: float
    random_seed: int
    action_smoothing: float


@dataclass
class VisionEvaluationResult:
    final_reward: float
    max_reward: float
    mean_inference_ms: float
    gpu_memory_mb: float
    video_path: Optional[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run randomised vision-based GPI evaluations.")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET), help="Path to PushT vision dataset (.zarr).")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Path to trained vision checkpoint.")
    parser.add_argument("--count", type=int, default=10, help="Number of evaluations to run.")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum steps per evaluation.")
    parser.add_argument("--random-seed", type=int, default=None, help="Seed for stable reproduction.")
    parser.add_argument("--no-save-video", dest="save_video", action="store_false", default=True, help="Disable mp4 capture for each rollout.")
    parser.add_argument("--video-dir", type=str, default=None, help="Directory to store evaluation videos (defaults to results/).")
    return parser.parse_args()


def ensure_resource_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} '{path}' not found. Please place it under 'models/' before running.")


def set_global_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def generate_configs(count: int, seed: Optional[int]) -> list[VisionEvaluationConfig]:
    rng_seed = seed if seed is not None else random.randrange(1 << 30)
    rng = random.Random(rng_seed)
    configs: list[VisionEvaluationConfig] = []
    env_seed_pool = list(range(400, 901))
    if count <= len(env_seed_pool):
        env_seeds = rng.sample(env_seed_pool, count)
    else:
        rng.shuffle(env_seed_pool)
        repeats = (count // len(env_seed_pool)) + 1
        env_seeds = (env_seed_pool * repeats)[:count]
    for idx, env_seed in enumerate(env_seeds):
        configs.append(
            VisionEvaluationConfig(
                name=f"vision_run_{idx:02d}",
                seed=env_seed,
                k_neighbors=rng.choice([3, 4, 5, 6, 7, 8]),
                action_horizon=rng.randint(1, 16),
                obs_noise_std=round(rng.uniform(0.0, 0.02), 4),
                random_seed=rng.randint(0, 1000),
                action_smoothing=round(rng.uniform(0.0, 0.4), 3),
            )
        )
    return configs


def gpu_allocated_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    try:
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / (1024**2)
    except Exception:
        return 0.0


def resolve_video_path(
    config: VisionEvaluationConfig,
    base_dir: Optional[Path],
) -> Optional[str]:
    if base_dir is None:
        return None
    base_dir.mkdir(parents=True, exist_ok=True)
    return str(base_dir / f"{config.name}.mp4")


def run_evaluation_run(
    config: VisionEvaluationConfig,
    dataset_path: Path,
    checkpoint_path: Path,
    max_steps: int,
    save_video: bool,
    video_dir: Optional[Path],
) -> VisionEvaluationResult:
    baseline_gpu = gpu_allocated_mb()
    base_config = GPIConfig(
        dataset_path=str(dataset_path),
        dataset_loader=load_episode_dataset,
        use_relative_action=False,
        k_neighbors=config.k_neighbors,
        obs_noise_std=config.obs_noise_std,
        enable_obs_noise=config.obs_noise_std > 0.0,
        random_seed=config.random_seed,
        action_horizon=config.action_horizon,
        action_smoothing=config.action_smoothing,
        subset_size=None,
        device="cuda" if torch is not None and torch.cuda.is_available() else None,
    )
    vision_config = VisionPolicyConfig(
        **base_config.__dict__,
        vision_checkpoint=str(checkpoint_path),
    )
    policy = VisionGPIPolicy(vision_config)
    evaluator = VisionEvaluator(env_seed=config.seed, max_steps=max_steps)
    gpu_after_init = gpu_allocated_mb()
    gpu_memory_mb = max(0.0, gpu_after_init - baseline_gpu)

    explicit_video_path = resolve_video_path(config, video_dir) if save_video else None
    result = evaluator.evaluate(
        policy,
        render_video=save_video,
        video_path=explicit_video_path,
        verbose=False,
        live_display=False,
    )
    stats = result.inference_stats if isinstance(result.inference_stats, dict) else {}
    if "policy" in stats and isinstance(stats["policy"], dict):
        mean_ms = float(stats["policy"].get("mean_ms", 0.0))
    else:
        mean_ms = float(stats.get("mean_ms", 0.0))
    max_reward = float(result.max_reward)
    final_reward = float(result.rewards[-1]) if result.rewards else max_reward
    return VisionEvaluationResult(
        final_reward=final_reward,
        max_reward=max_reward,
        mean_inference_ms=mean_ms,
        gpu_memory_mb=gpu_memory_mb,
        video_path=result.video_path if save_video else None,
    )


def ensure_log_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_log(summaries: Iterable[tuple[VisionEvaluationConfig, VisionEvaluationResult]]) -> None:
    ensure_log_dir(LOG_PATH)
    timestamp = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"# Vision policy batch {timestamp}\n")
        for config, result in summaries:
            log_file.write(
                json.dumps(
                    {
                        "config": asdict(config),
                        "metrics": {
                            "final_reward": result.final_reward,
                            "max_reward": result.max_reward,
                            "mean_inference_ms": result.mean_inference_ms,
                            "gpu_memory_mb": result.gpu_memory_mb,
                            "video_path": result.video_path,
                        },
                    }
                )
                + "\n"
            )
        log_file.write("\n")


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset)
    checkpoint = Path(args.checkpoint)
    if not dataset.is_absolute():
        dataset = (PROJECT_ROOT / dataset).resolve()
    if not checkpoint.is_absolute():
        checkpoint = (PROJECT_ROOT / checkpoint).resolve()
    ensure_resource_exists(dataset, "Dataset")
    ensure_resource_exists(checkpoint, "Checkpoint")

    video_dir = Path(args.video_dir).expanduser() if args.video_dir else None
    if video_dir and not video_dir.is_absolute():
        video_dir = (PROJECT_ROOT / video_dir).resolve()

    seed = args.random_seed
    set_global_seed(seed)

    configs = generate_configs(args.count, seed)
    summaries: list[tuple[VisionEvaluationConfig, VisionEvaluationResult]] = []
    for config in configs:
        set_global_seed(config.random_seed)
        result = run_evaluation_run(config, dataset, checkpoint, args.max_steps, args.save_video, video_dir)
        summaries.append((config, result))
        message = (
            f"{config.name}: Reward: {result.final_reward:.3f}\t"
            f"Inference Time: {result.mean_inference_ms:.2f} ms\t"
            f"Memory: {result.gpu_memory_mb:.3f} MB"
        )
        if result.video_path:
            message += f"\tVideo: {result.video_path}"
        print(message)

    append_log(summaries)
    print("Completed vision policy batch.")


if __name__ == "__main__":
    main()
