"""Evaluation helpers for GPI policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import os
import collections
import numpy as np
import cv2
from tqdm.auto import tqdm
from skvideo.io import vwrite

from .envs import PushTEnv, PushTImageEnv


@dataclass
class EvaluationResult:
    max_reward: float
    rewards: list[float]
    steps: int
    total_time: float
    inference_stats: dict
    video_path: Optional[str]


class StateEvaluator:
    def __init__(self, env_seed: int = 500, max_steps: int = 500) -> None:
        self.env_seed = env_seed
        self.max_steps = max_steps
        self.env = PushTEnv()

    def evaluate(
        self,
        policy,
        render_video: bool = True,
        video_path: Optional[str] = None,
        verbose: bool = True,
        live_display: bool = False,
    ) -> EvaluationResult:
        self.env.seed(self.env_seed)
        obs, _ = self.env.reset()
        self.env.reset_kp_kv(100, 20)
        policy.reset()
        rewards: list[float] = []
        capture_frames = render_video or live_display
        initial_frame = self.env.render(mode="rgb_array") if capture_frames else None
        frames: list[np.ndarray] = (
            [initial_frame] if (render_video and initial_frame is not None) else []
        )
        if live_display and initial_frame is not None:
            cv2.imshow(
                "PushT State Policy", cv2.cvtColor(initial_frame, cv2.COLOR_RGB2BGR)
            )
            cv2.waitKey(1)
        done = False
        step_idx = 0
        obs_deque = collections.deque([obs], maxlen=1)
        with tqdm(total=self.max_steps, desc="State GPI", disable=not verbose) as pbar:
            while not done and step_idx < self.max_steps:
                current_obs = np.stack(obs_deque)
                action = policy.get_action(current_obs)
                action_slice = slice(0, 2)
                object_slice = slice(2, 4)
                # policy.plot_knn_state_trajectories(
                #     obs, k=2, pos_slice=object_slice, show_points=True, show_mean=True
                # )
                # policy.plot_knn_state_trajectories(
                #     obs, k=2, pos_slice=action_slice, show_points=True, show_mean=True
                # )
                obs, reward, done, _, _ = self.env.step(action)
                obs_deque.append(obs)
                rewards.append(float(reward))
                if capture_frames:
                    frame = self.env.render(mode="rgb_array")
                    if render_video:
                        frames.append(frame)
                    if live_display:
                        cv2.imshow(
                            "PushT State Policy", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        )
                        cv2.waitKey(1)
                step_idx += 1
                if verbose:
                    pbar.update(1)
                    pbar.set_postfix(reward=f"{reward:.3f}")
        inference_stats = policy.get_inference_stats()
        video_out = None
        if render_video and frames:
            if video_path is None:
                video_path = f"results/state_policy_{self.env_seed}.mp4"
            directory = os.path.dirname(video_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            vwrite(video_path, np.array(frames))
            video_out = video_path
        if live_display:
            cv2.destroyWindow("PushT State Policy")
        return EvaluationResult(
            max_reward=max(rewards) if rewards else 0.0,
            rewards=rewards,
            steps=step_idx,
            total_time=sum(policy.inference_times),
            inference_stats=inference_stats,
            video_path=video_out,
        )


class VisionEvaluator:
    def __init__(self, env_seed: int = 500, max_steps: int = 200) -> None:
        self.env_seed = env_seed
        self.max_steps = max_steps
        self.env = PushTImageEnv()

    def evaluate(
        self,
        policy,
        render_video: bool = True,
        video_path: Optional[str] = None,
        verbose: bool = True,
        live_display: bool = False,
    ) -> EvaluationResult:
        self.env.seed(self.env_seed)
        obs, _ = self.env.reset()
        self.env.reset_kp_kv(100, 20)
        policy.reset()
        capture_frames = render_video or live_display
        initial_frame = self.env.render(mode="rgb_array") if capture_frames else None
        frames: list[np.ndarray] = (
            [initial_frame] if (render_video and initial_frame is not None) else []
        )
        if live_display and initial_frame is not None:
            cv2.imshow(
                "PushT Vision Policy", cv2.cvtColor(initial_frame, cv2.COLOR_RGB2BGR)
            )
            cv2.waitKey(1)
        rewards: list[float] = []
        done = False
        step_idx = 0
        image_deque = collections.deque([obs["image"]], maxlen=1)
        agent_deque = collections.deque([obs["agent_pos"]], maxlen=1)
        with tqdm(total=self.max_steps, desc="Vision GPI", disable=not verbose) as pbar:
            while not done and step_idx < self.max_steps:
                current_image = np.stack(image_deque)
                current_agent = np.stack(agent_deque)
                action = policy.get_action(current_image, current_agent)
                obs, reward, done, _, _ = self.env.step(action)
                image_deque.append(obs["image"])
                agent_deque.append(obs["agent_pos"])
                rewards.append(float(reward))
                if capture_frames:
                    frame = self.env.render(mode="rgb_array")
                    if render_video:
                        frames.append(frame)
                    if live_display:
                        cv2.imshow(
                            "PushT Vision Policy",
                            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                        )
                        cv2.waitKey(1)
                step_idx += 1
                if verbose:
                    pbar.update(1)
                    pbar.set_postfix(reward=f"{reward:.3f}")
        inference_stats = (
            policy.get_full_inference_stats()
            if hasattr(policy, "get_full_inference_stats")
            else policy.get_inference_stats()
        )
        video_out = None
        if render_video and frames:
            if video_path is None:
                video_path = f"results/vision_policy_{self.env_seed}.mp4"
            directory = os.path.dirname(video_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            vwrite(video_path, np.array(frames))
            video_out = video_path
        total_inference = sum(policy.inference_times)
        if live_display:
            cv2.destroyWindow("PushT Vision Policy")
        return EvaluationResult(
            max_reward=max(rewards) if rewards else 0.0,
            rewards=rewards,
            steps=step_idx,
            total_time=total_inference,
            inference_stats=inference_stats,
            video_path=video_out,
        )


__all__ = ["StateEvaluator", "VisionEvaluator", "EvaluationResult"]
