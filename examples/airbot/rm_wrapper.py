"""Dense-reward integration: BinaryProgressRewardModel adapter for DSRL Airbot.

The Reward-Model-MVP repo (`/home/jpy/RM/Airbot-VLA-RL/Reward-Model-MVP`) implements
`BinaryProgressRewardModel`: a DINOv3 ViT-B16 encoder + cross-attn comparator that
compares rollout clips to demonstration clips and returns 1.0/0.0 per clip based on
high-water-mark progress.

This module wraps it for DSRL's training loop:
  - On init, load demo frames, pad them so the RM library auto-constructs exactly
    `num_demo_clips` clips covering the task span (mirrors the cap-to-last-frame
    behavior used for rollout clip k=7 at frame index 200).
  - On `compute_rewards(rollout_frames, num_query_steps)`, build per-query-step
    clips with the same cap-to-last-frame rule, encode and score each clip vs all
    demo clips, apply BinaryProgressClipRewardModel's hit logic (max_sim >=
    per_clip_threshold AND demo_progress > previous_high_water), and return
    `num_query_steps` floats: 0.0 on hit, -1.0 on miss.

The wrapper bypasses RM's `step()` (which strides one frame at a time) and instead
calls `_prepare_frame` + `_encode_clip` + comparator directly, so we can use
user-defined query-step clip indices that don't align with RM's natural stride.
"""
import hashlib
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np


def _add_rm_repo_to_sys_path(rm_repo_path: str) -> None:
    """Add RM repo paths to sys.path so that
    `from reward_model_baseline.MetaWorld.RewardModels.BinaryProgressRewardModel
    import BinaryProgressRewardModel` resolves (this is the same import style
    SERL uses — see SERL/serl_robot_infra/airbot_env/envs/progress_reward_wrapper.py).
    BinaryProgressRewardModel.py itself adds `<repo>/Reference-Anchored_RM` to
    sys.path for the internal `from scripts.reward_model import ClipRewardModel`.
    Idempotent — safe to call multiple times.
    """
    if not os.path.isdir(rm_repo_path):
        raise FileNotFoundError(
            f"RM repo not found: {rm_repo_path} (check --rm_repo_path)")
    metaworld_dir = os.path.join(rm_repo_path, 'reward_model_baseline', 'MetaWorld')
    if not os.path.isdir(metaworld_dir):
        raise FileNotFoundError(
            f"RM repo missing reward_model_baseline/MetaWorld at: {rm_repo_path}")
    if rm_repo_path not in sys.path:
        sys.path.insert(0, rm_repo_path)


def _build_padded_demo_dir(
    demo_path: str,
    target_total_frames: int,
    cache_root: str = '/tmp',
) -> str:
    """Read frame_*.jpg from `demo_path` and produce a folder of exactly
    `target_total_frames` JPEGs, padding with copies of the last real frame if
    fewer were provided. Result is cached under
    `<cache_root>/dsrl_rm_padded_<hash>/` so re-init across wrapper restarts
    skips re-writing.

    Returns the absolute path to the padded-demo directory.
    """
    src = Path(demo_path)
    if not src.is_dir():
        raise FileNotFoundError(f"demo_path is not a directory: {demo_path}")

    src_frames = sorted(src.glob("frame_*.jpg"))
    if not src_frames:
        src_frames = sorted(src.glob("frame_*.png"))
    if not src_frames:
        raise ValueError(
            f"No frame_*.jpg or frame_*.png found in {demo_path}")

    key = f"{os.path.abspath(demo_path)}|{target_total_frames}|{len(src_frames)}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    out_dir = Path(cache_root) / f"dsrl_rm_padded_{digest}"

    needed = [out_dir / f"frame_{i:06d}.jpg" for i in range(target_total_frames)]
    if out_dir.is_dir() and all(p.exists() for p in needed):
        return str(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    n_src = len(src_frames)
    n_real = min(n_src, target_total_frames)
    for i in range(n_real):
        dst = out_dir / f"frame_{i:06d}.jpg"
        if not dst.exists():
            import shutil
            shutil.copyfile(src_frames[i], dst)

    last_frame = src_frames[n_real - 1]
    for i in range(n_real, target_total_frames):
        dst = out_dir / f"frame_{i:06d}.jpg"
        if not dst.exists():
            import shutil
            shutil.copyfile(last_frame, dst)

    return str(out_dir)


class AirbotRewardModel:
    """Thin adapter around BinaryProgressRewardModel for DSRL Airbot training.

    Usage:
        rm = AirbotRewardModel(demo_path='...', max_timesteps=200, query_freq=25,
                               num_query_steps=8, rm_repo_path='...')
        rewards = rm.compute_rewards(rollout_frames=[H,W,3 uint8 RGB, ...] * 200,
                                     num_query_steps=8)
        # rewards.shape == (8,); 0.0 = RM hit, -1.0 = miss.
    """

    def __init__(
        self,
        demo_path: str,
        rm_repo_path: str,
        max_timesteps: int = 200,
        query_freq: int = 25,
        num_query_steps: int = 8,
        num_frames: int = 5,
        frame_stride: int = 10,
        demo_clip_stride: int = 5,
        threshold_offset: float = 0.5,
        device: str = 'cuda',
    ):
        _add_rm_repo_to_sys_path(rm_repo_path)

        from reward_model_baseline.MetaWorld.RewardModels.BinaryProgressRewardModel import (
            BinaryProgressRewardModel,
        )
        from reward_model_baseline.MetaWorld.RewardModels.ReferenceAnchoredRewardModel import (
            ReferenceAnchoredRewardModel,
        )

        BinaryProgressRewardModel.reset_global_instance()
        ReferenceAnchoredRewardModel.reset_global_instance()

        self.max_timesteps = max_timesteps
        self.query_freq = query_freq
        self.num_query_steps = num_query_steps
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.demo_clip_stride = demo_clip_stride
        self.threshold_offset = threshold_offset
        self.device = device

        self.clip_span = (num_frames - 1) * frame_stride + 1

        # Demo clip middle-frame should cover env_steps roughly [half_span,
        # max_timesteps]. With max_timesteps=200, clip_span=41, demo_stride=5
        # this gives s_max=180 → 37 demo clips, matching the user spec.
        half_span = (self.clip_span - 1) // 2
        last_demo_start = max(0, max_timesteps - half_span)
        target_demo_total = last_demo_start + self.clip_span
        self.target_demo_total_frames = target_demo_total

        padded_demo_dir = _build_padded_demo_dir(
            demo_path=demo_path,
            target_total_frames=target_demo_total,
        )
        self.padded_demo_dir = padded_demo_dir

        self._rm = BinaryProgressRewardModel.get_global_instance(
            demo_path=padded_demo_dir,
            device=device,
            num_frames=num_frames,
            frame_stride=frame_stride,
            clip_stride=demo_clip_stride,
        )
        self._inner = self._rm.model

        if abs(threshold_offset - 0.5) > 1e-6:
            self._inner.per_clip_thresholds = [
                float(t) + 0.5 - float(threshold_offset)
                for t in self._inner.per_clip_thresholds
            ]

        self.num_demo_clips = len(self._inner.demo_clip_start_indices)
        self.per_clip_thresholds = list(self._inner.per_clip_thresholds)
        self.demo_progress = list(self._inner.demo_progress)

        print(f"[AirbotRewardModel] demo_path={demo_path}")
        print(f"[AirbotRewardModel] padded_demo_dir={padded_demo_dir} "
              f"(target_total_frames={target_demo_total})")
        print(f"[AirbotRewardModel] num_demo_clips={self.num_demo_clips}, "
              f"num_frames={num_frames}, frame_stride={frame_stride}, "
              f"clip_span={self.clip_span}, demo_clip_stride={demo_clip_stride}")
        print(f"[AirbotRewardModel] per_clip_thresholds: "
              f"min={min(self.per_clip_thresholds):.4f}, "
              f"max={max(self.per_clip_thresholds):.4f}, "
              f"mean={float(np.mean(self.per_clip_thresholds)):.4f}")
        print(f"[AirbotRewardModel] demo_progress range: "
              f"[{min(self.demo_progress):.3f}, {max(self.demo_progress):.3f}]")

    @staticmethod
    def _cap_indices(start: int, num_frames: int, frame_stride: int, max_idx: int) -> List[int]:
        """Build clip indices [start, start+stride, ...] capping any out-of-range
        index to `max_idx` (the last valid frame). Mirrors user spec for k=7."""
        return [min(start + i * frame_stride, max_idx) for i in range(num_frames)]

    def compute_rewards(
        self,
        rollout_frames: Sequence[np.ndarray],
        num_query_steps: Optional[int] = None,
    ) -> np.ndarray:
        """Compute per-query-step dense rewards for one rollout.

        Args:
            rollout_frames: List of HxWx3 uint8 RGB frames captured at each env
                step (length should be ~max_timesteps).
            num_query_steps: Number of query steps in the rollout (defaults to
                self.num_query_steps).

        Returns:
            np.ndarray of shape (num_query_steps,), dtype float32: 0.0 on RM
            hit, -1.0 on miss.
        """
        import torch
        from PIL import Image

        if num_query_steps is None:
            num_query_steps = self.num_query_steps

        rewards = -np.ones(num_query_steps, dtype=np.float32)

        if len(rollout_frames) == 0:
            return rewards

        max_idx = len(rollout_frames) - 1

        max_reached_progress = 0.0
        max_demo_progress = max(self.demo_progress) if self.demo_progress else 0.0

        with torch.inference_mode():
            for k in range(num_query_steps):
                if max_demo_progress > 0.0 and max_reached_progress >= max_demo_progress:
                    break

                start = k * self.query_freq
                idxs = self._cap_indices(start, self.num_frames, self.frame_stride, max_idx)

                clip_tensors = []
                for idx in idxs:
                    f = rollout_frames[idx]
                    if isinstance(f, np.ndarray):
                        if f.dtype != np.uint8:
                            if f.max() <= 1.5:
                                f = (f * 255).clip(0, 255).astype(np.uint8)
                            else:
                                f = np.clip(f, 0, 255).astype(np.uint8)
                        pil = Image.fromarray(f, mode='RGB')
                    elif isinstance(f, Image.Image):
                        pil = f.convert('RGB')
                    else:
                        raise TypeError(
                            f"rollout_frames must contain np.ndarray or PIL.Image, got {type(f)}")
                    clip_tensors.append(self._inner._prepare_frame(pil))

                clip = torch.stack(clip_tensors)
                rollout_emb = self._inner._encode_clip(clip)

                demo_batch = self._inner.demo_embs
                rollout_batch = rollout_emb.unsqueeze(0).expand(demo_batch.size(0), -1, -1)
                rollout_batch = rollout_batch.to(self._inner.model.dtype)
                demo_batch_dtype = demo_batch.to(self._inner.model.dtype)

                emb_a_vs_b = self._inner.model.cross_attn(rollout_batch, demo_batch_dtype)
                sims = self._inner.model.comparison_head(emb_a_vs_b)
                sims_np = sims.detach().float().cpu().numpy()

                max_sim = float(np.max(sims_np))
                best_demo_idx = int(np.argmax(sims_np))
                clip_threshold = self._inner.per_clip_thresholds[best_demo_idx]

                if max_sim >= clip_threshold:
                    current_progress = self._inner.demo_progress[best_demo_idx]
                    if current_progress > max_reached_progress:
                        max_reached_progress = current_progress
                        rewards[k] = 0.0

        self.last_max_reached_progress = max_reached_progress
        self.last_max_demo_progress = max_demo_progress
        return rewards

    def close(self) -> None:
        """Release the RM singleton (frees GPU memory)."""
        try:
            from reward_model_baseline.MetaWorld.RewardModels.BinaryProgressRewardModel import (
                BinaryProgressRewardModel,
            )
            from reward_model_baseline.MetaWorld.RewardModels.ReferenceAnchoredRewardModel import (
                ReferenceAnchoredRewardModel,
            )
            BinaryProgressRewardModel.reset_global_instance()
            ReferenceAnchoredRewardModel.reset_global_instance()
        except Exception:
            pass
        self._rm = None
        self._inner = None
