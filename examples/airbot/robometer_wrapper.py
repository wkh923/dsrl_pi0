"""Dense-reward integration: Robometer adapter for DSRL Airbot.

robometer/robometer (https://github.com/robometer/robometer, MIT) is a VLM-based
reward model (Robometer-4B) that scores task progress + success directly from a
rollout video and a task-language description — no reference demo clip needed,
unlike the Reward-Model-MVP variants in `rm_wrapper*.py`.

This module wraps it behind the same `compute_rewards(rollout_frames,
num_query_steps) -> np.ndarray` contract those wrappers use, so it can be
selected via `--use_rm --rm_variant robometer` in train_airbot.py without
touching the training loop.

Setup (not done by this file):
  git clone https://github.com/robometer/robometer /home/jpy/RM/Airbot-VLA-RL/robometer
  pip install -e /home/jpy/RM/Airbot-VLA-RL/robometer   # installs the `robometer` package
First run downloads --robometer_model_path (default 'robometer/Robometer-4B')
from Hugging Face via `load_model_from_hf`; set HF_HOME to redirect the cache
off of the already-crowded ~/.cache if needed.

TODO before first real run (unverified against the actual installed package —
based on scripts/example_inference_local.py's documented usage):
  - Confirm compute_batch_outputs's "progress_pred" is per-frame (one score per
    input frame) vs. a single clip-level score; the per-query-step sampling
    below assumes per-frame.
  - Confirm Trajectory.frames expects uint8 HxWx3 RGB, matching rollout_frames.
"""
import os
import sys
from typing import List, Optional, Sequence

import numpy as np


def _add_robometer_repo_to_sys_path(robometer_repo_path: str) -> None:
    """Fallback for sys.path if `pip install -e` wasn't run. Idempotent."""
    if not os.path.isdir(robometer_repo_path):
        raise FileNotFoundError(
            f"Robometer repo not found: {robometer_repo_path} "
            "(check --robometer_repo_path, or `pip install -e` it so this "
            "path check can be skipped entirely).")
    if robometer_repo_path not in sys.path:
        sys.path.insert(0, robometer_repo_path)


class RobometerRewardModel:
    """Scores DSRL rollout clips against a task instruction via Robometer-4B.

    Reward semantics mirror `rm_wrapper.AirbotRewardModel.compute_rewards`:
    returns one float per query step, each in roughly [-0.5, 0.5], derived from
    the progress delta since the previous query step (0.0 delta on the first
    step). The +1.0 success override on the final step is still applied by the
    caller (train_utils_airbot.collect_traj), same as the other RM variants.
    """

    def __init__(
        self,
        instruction: str,
        robometer_repo_path: str,
        robometer_model_path: str = 'robometer/Robometer-4B',
        max_timesteps: int = 1000,
        query_freq: int = 25,
        num_query_steps: int = 40,
        capture_stride: int = 1,
        device: Optional[str] = None,
        **_unused,
    ):
        _add_robometer_repo_to_sys_path(robometer_repo_path)

        import torch
        from robometer.utils.save import load_model_from_hf
        from robometer.utils.setup_utils import setup_batch_collator

        self.instruction = instruction
        self.max_timesteps = max_timesteps
        self.query_freq = query_freq
        self.num_query_steps = num_query_steps
        self.capture_stride = capture_stride
        self.device = torch.device(
            device or ('cuda' if torch.cuda.is_available() else 'cpu'))

        print(f"[Robometer] loading {robometer_model_path} on {self.device} ...")
        self.exp_config, self.tokenizer, self.processor, self.reward_model = (
            load_model_from_hf(model_path=robometer_model_path, device=self.device))
        self.batch_collator = setup_batch_collator(
            self.processor, self.tokenizer, self.exp_config, is_eval=True)
        self.reward_model.eval()
        print("[Robometer] model loaded.")

    def compute_rewards(
        self,
        rollout_frames: Sequence[np.ndarray],
        num_query_steps: Optional[int] = None,
        traj_id: Optional[int] = None,
        is_success: bool = False,
    ) -> np.ndarray:
        """Args mirror rm_wrapper.AirbotRewardModel.compute_rewards.

        rollout_frames: HxWx3 uint8 RGB frames captured every `capture_stride`
            env-steps (rollout_frames[i] <-> env_step i*capture_stride).
        """
        import torch
        from robometer.data.dataset_types import ProgressSample, Trajectory
        from robometer.evals.eval_server import compute_batch_outputs

        if num_query_steps is None:
            num_query_steps = self.num_query_steps

        rewards = np.zeros(num_query_steps, dtype=np.float32)
        if len(rollout_frames) == 0:
            return rewards

        frames = np.stack([np.asarray(f) for f in rollout_frames], axis=0)

        traj = Trajectory(
            frames=frames,
            frames_shape=frames.shape,
            task=self.instruction,
            id=f"dsrl_rollout_{traj_id if traj_id is not None else '?'}",
            metadata={},
            video_embeddings=None,
        )
        sample = ProgressSample(trajectory=traj, sample_type='progress')
        batch = self.batch_collator([sample])

        with torch.inference_mode():
            out = compute_batch_outputs(
                self.reward_model, self.tokenizer, batch['progress_inputs'],
                sample_type='progress', is_discrete_mode=False, num_bins=None)

        # progress_pred assumed per-frame, aligned to `frames` order (see TODO
        # in module docstring). Sample it at each query step's frame index.
        progress_per_frame = np.asarray(out['progress_pred']).reshape(-1)
        query_freq_capture = max(1, self.query_freq // self.capture_stride)
        max_idx = len(progress_per_frame) - 1

        prev_progress = 0.0
        for k in range(num_query_steps):
            idx = min(k * query_freq_capture, max_idx)
            curr_progress = float(progress_per_frame[idx])
            delta = curr_progress - prev_progress
            rewards[k] = np.clip(delta, -0.5, 0.5)
            prev_progress = curr_progress

        tid = f"{traj_id}" if traj_id is not None else "?"
        print(f"[Robometer] rollout {tid}: final progress={prev_progress:.3f}, "
              f"reward range=[{rewards.min():.3f}, {rewards.max():.3f}]")
        return rewards
