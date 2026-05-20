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
        capture_stride: int = 1,
        device: str = 'cuda',
    ):
        # All stride params are in env-step terms. capture_stride is how often
        # (in env-steps) frames are actually saved to disk / captured into
        # rm_frames during training. capture_stride=1 → dense; capture_stride=5
        # → sparse (one frame per 5 env-steps, file_i corresponds to env_step
        # i*capture_stride). Must evenly divide frame_stride, demo_clip_stride,
        # and query_freq so file/capture-index arithmetic stays integer.
        if frame_stride % capture_stride != 0:
            raise ValueError(
                f"frame_stride ({frame_stride}) must be divisible by "
                f"capture_stride ({capture_stride})")
        if demo_clip_stride % capture_stride != 0:
            raise ValueError(
                f"demo_clip_stride ({demo_clip_stride}) must be divisible by "
                f"capture_stride ({capture_stride})")
        if query_freq % capture_stride != 0:
            raise ValueError(
                f"query_freq ({query_freq}) must be divisible by "
                f"capture_stride ({capture_stride})")

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
        self.query_freq = query_freq                                  # env-step
        self.query_freq_capture = query_freq // capture_stride        # capture-idx
        self.num_query_steps = num_query_steps
        self.num_frames = num_frames
        self.frame_stride = frame_stride                              # env-step
        self.frame_stride_file = frame_stride // capture_stride       # file-idx
        self.demo_clip_stride = demo_clip_stride                      # env-step
        self.demo_clip_stride_file = demo_clip_stride // capture_stride  # file-idx
        self.threshold_offset = threshold_offset
        self.capture_stride = capture_stride
        self.device = device

        self.clip_span = (num_frames - 1) * frame_stride + 1           # env-step
        self.clip_span_file = (num_frames - 1) * self.frame_stride_file + 1

        # Demo clip middle-frame should cover env_steps roughly [half_span,
        # max_timesteps]. target_envstep = last_demo_start + clip_span. For
        # max_timesteps=200, half_span=20, clip_span=41 → target=221 env_steps
        # (env_step indices 0..220). Last 21 are padded copies of frame 199.
        half_span = (self.clip_span - 1) // 2
        last_demo_start_envstep = max(0, max_timesteps - half_span)
        target_envstep = last_demo_start_envstep + self.clip_span
        # Convert to file count for the storage layer: largest needed env_step
        # is (target_envstep - 1); round up to a multiple of capture_stride.
        last_needed_envstep = target_envstep - 1
        last_needed_file = (last_needed_envstep + capture_stride - 1) // capture_stride
        target_demo_total_files = last_needed_file + 1
        self.target_demo_total_files = target_demo_total_files
        self.target_envstep = target_envstep

        padded_demo_dir = _build_padded_demo_dir(
            demo_path=demo_path,
            target_total_frames=target_demo_total_files,
        )
        self.padded_demo_dir = padded_demo_dir

        self._rm = BinaryProgressRewardModel.get_global_instance(
            demo_path=padded_demo_dir,
            device=device,
            num_frames=num_frames,
            frame_stride=self.frame_stride_file,
            clip_stride=self.demo_clip_stride_file,
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
              f"(target_files={target_demo_total_files}, "
              f"target_envstep={target_envstep}, capture_stride={capture_stride})")
        print(f"[AirbotRewardModel] num_demo_clips={self.num_demo_clips}, "
              f"num_frames={num_frames}, frame_stride={frame_stride} env-step "
              f"(={self.frame_stride_file} file), clip_span={self.clip_span} env-step, "
              f"demo_clip_stride={demo_clip_stride} env-step (={self.demo_clip_stride_file} file)")
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
        traj_id: Optional[int] = None,
    ) -> np.ndarray:
        """Compute per-query-step dense rewards for one rollout.

        Args:
            rollout_frames: List of HxWx3 uint8 RGB frames captured at every
                `capture_stride` env-steps. So rollout_frames[i] corresponds to
                env_step i*capture_stride. Length should be roughly
                max_timesteps // capture_stride.
            num_query_steps: Number of query steps in the rollout (defaults to
                self.num_query_steps).
            traj_id: Optional rollout id, used only in the per-clip log header.

        Returns:
            np.ndarray of shape (num_query_steps,), dtype float32. Per-clip
            reward semantics (halved range to keep SAC critic Q-values small):
              * hit   (matched + progress advanced):  -0.5 + delta in [-0.5, +0.5]
                where delta = min(1, delta_progress / (1/num_query_steps))
                and delta_progress = (best_idx - prev_idx) / num_demo_clips.
              * match (matched but no new progress): -0.5
              * miss  (max_sim < threshold):         -0.5
            The is_success override is applied by the caller
            (train_utils_airbot.collect_traj) AFTER this returns: on success
            the caller adds +1.0 to the RM-computed final-step reward (→ in
            [0.5, 1.5]); on failure the final step keeps its raw RM value.
        """
        import torch
        from PIL import Image

        if num_query_steps is None:
            num_query_steps = self.num_query_steps

        rewards = -0.5 * np.ones(num_query_steps, dtype=np.float32)

        if len(rollout_frames) == 0:
            return rewards

        max_idx_capture = len(rollout_frames) - 1

        max_reached_progress = 0.0      # library float (for monotonicity check)
        max_reached_idx = -1            # int demo-clip idx (for display + reward, Formula A)
        max_demo_progress = max(self.demo_progress) if self.demo_progress else 0.0
        N = max(1, self.num_demo_clips)
        expected_delta = 1.0 / max(1, num_query_steps)   # e.g. 0.125 for 8 steps

        # Header for the per-clip table.
        tid = f"{traj_id}" if traj_id is not None else "?"
        print(f"[RM] === rollout {tid}  ({num_query_steps} clips, "
              f"query_freq={self.query_freq}, capture_stride={self.capture_stride}, "
              f"num_demo_clips={self.num_demo_clips}, "
              f"expected_delta_per_step={expected_delta*100:.1f}%) ===")
        print(f"[RM]   k | rollout_frames                | max_sim | best_demo | "
              f"demo_frames                   | prev_prog | curr_prog | reward | status")

        with torch.inference_mode():
            for k in range(num_query_steps):
                if max_demo_progress > 0.0 and max_reached_progress >= max_demo_progress:
                    # All later clips can't advance progress → reward = -0.5
                    # (already the array's initial value). Skip GPU compute AND printing.
                    break

                # Build rollout clip indices in CAPTURE space (rollout_frames idx).
                start_capture = k * self.query_freq_capture
                idxs_capture = self._cap_indices(
                    start_capture, self.num_frames, self.frame_stride_file, max_idx_capture
                )
                idxs_envstep = [i * self.capture_stride for i in idxs_capture]

                clip_tensors = []
                for idx in idxs_capture:
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

                # Classify into hit / match / miss and compute reward.
                #   hit   = matched (sim >= thr) AND advances progress
                #   match = matched but no new progress (best_idx <= max_reached_idx)
                #   miss  = sim < thr
                matched = max_sim >= clip_threshold
                advanced = matched and best_demo_idx > max_reached_idx
                prev_prog_pct = (max(0, max_reached_idx) / N) * 100.0

                if advanced:
                    new_prog_pct = (best_demo_idx / N) * 100.0
                    delta_progress = (new_prog_pct - prev_prog_pct) / 100.0   # back to [0,1]
                    delta = min(1.0, delta_progress / expected_delta)
                    # Halved range: (-1 + 2*delta)/2 = -0.5 + delta ∈ [-0.5, +0.5].
                    rewards[k] = -0.5 + delta
                    # Advance progress trackers
                    max_reached_progress = self._inner.demo_progress[best_demo_idx]
                    max_reached_idx = best_demo_idx
                    status = "hit"
                elif matched:
                    rewards[k] = -0.5
                    status = "match"
                else:
                    rewards[k] = -0.5
                    status = "miss"

                curr_prog_pct = (max(0, max_reached_idx) / N) * 100.0

                # Render row. "match" still shows best_demo + demo_frames (so user
                # can see WHICH demo clip the rollout matched); only "miss" gets
                # the "miss" placeholders.
                rollout_str = self._fmt_frames(idxs_envstep)
                fmt_width = len(rollout_str)
                if status != "miss":
                    demo_start_file = self._inner.demo_clip_start_indices[best_demo_idx]
                    demo_idxs_envstep = [
                        demo_start_file * self.capture_stride + i * self.frame_stride
                        for i in range(self.num_frames)
                    ]
                    demo_str = self._fmt_frames(demo_idxs_envstep)
                    best_str = f"{best_demo_idx:>5d}"
                else:
                    demo_str = f"{'miss':<{fmt_width}}"
                    best_str = f"{'miss':>5}"

                print(f"[RM]  {k:>2d} | {rollout_str} |  {max_sim:.3f}  | "
                      f"{best_str}     | {demo_str} |  {prev_prog_pct:>4.1f}%   | "
                      f"  {curr_prog_pct:>5.1f}%  | {rewards[k]:>+5.2f}  | {status:<5}")

        self.last_max_reached_progress = max_reached_progress
        self.last_max_demo_progress = max_demo_progress
        return rewards

    @staticmethod
    def _fmt_frames(idxs):
        """Format a list of frame indices as a fixed-width string [   0,   10, ...].
        4-digit width so env_steps up to 9999 align nicely (max_timesteps=1000
        → env_steps up to 1020 padded)."""
        return "[" + ", ".join(f"{i:>4d}" for i in idxs) + "]"

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
