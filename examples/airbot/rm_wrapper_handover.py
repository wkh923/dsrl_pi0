"""Handover-task-specific reward model: simple per-step progress reward.

For each query step k in a rollout with `n` query steps:

  is_success = 0 (FAILURE):
      Every step gets -0.5 (regardless of hit / match / miss). Final step
      stays -0.5 too (train_utils' +1.0 success override only fires on
      is_success=1).

  is_success = 1 (SUCCESS):
      matched (max_sim >= per-clip threshold):
          reward = -0.5 + normalized_progress
          where normalized_progress = demo_progress[best_demo_idx] /
          max(demo_progress) ∈ [0, 1]. (RM library raw demo_progress is
          scaled to [0, ~10]; we renormalize so the additive term stays in
          [0, 1] and the reward stays in [-0.5, +0.5].) So matching the last
          demo clip (progress=1.0) yields +0.5; matching a mid clip with
          normalized progress 0.8 yields +0.3, and consecutive steps that
          all match clips at normalized 0.8 each earn +0.3.
      miss (max_sim < threshold):
          reward = -0.5  (i.e. progress treated as 0).

  Final step (any k == n-1) on SUCCESS additionally receives train_utils'
  +1.0 override → final reward = -0.5 + progress + 1.0 (or 0.5 on miss).

Differs from `ClothesRewardModel`:
  * No regime zone, no high-water mark, no advance delta.
  * Failure → flat -1 (vs clothes' progress-based partial credit).
  * Success → uniformly `-0.5 + demo_progress[k]` per step (vs clothes'
    progress-delta on advance only).
"""
import numpy as np

from examples.airbot.rm_wrapper import AirbotRewardModel


class HandoverRewardModel(AirbotRewardModel):
    """Simple progress-position reward model for handover."""

    # The final step still gets train_utils' +1.0 success override.
    handles_success_internally = False

    def compute_rewards(self, rollout_frames, num_query_steps=None,
                        traj_id=None, is_success=False):
        import torch

        if num_query_steps is None:
            num_query_steps = self.num_query_steps
        n = int(num_query_steps)

        rewards = -0.5 * np.ones(n, dtype=np.float32)
        if n == 0 or len(rollout_frames) == 0:
            return rewards

        tid = f"{traj_id}" if traj_id is not None else "?"

        # FAILURE shortcut: every step -0.5, no clip scoring.
        if not is_success:
            print(f"[RM-handover] === rollout {tid}  ({n} clips, "
                  f"is_success=0) — FAILURE: all rewards = -0.5 (no clip scoring) ===")
            print(f"[RM-handover] rewards={[round(float(r), 2) for r in rewards.tolist()]} "
                  f"sum={float(rewards.sum()):.2f} is_success=0")
            return rewards

        # SUCCESS path: score every step, reward = -0.5 + normalized_progress
        # (matched) or -0.5 (miss). The +1.0 success override on the final step
        # is added by train_utils_airbot.collect_traj after this returns.
        # NOTE: RM library scales demo_progress to [0, ~10] (see
        # Reference-Anchored_RM/scripts/reward_model.py:82-85). We renormalize
        # to [0, 1] by dividing by max(demo_progress), so the "-0.5 + progress"
        # reward stays in [-0.5, +0.5] per step.
        max_idx_capture = len(rollout_frames) - 1
        last = n - 1
        max_demo_progress = float(self._inner._max_demo_progress) \
            if self._inner._max_demo_progress > 0 else 1.0

        print(f"[RM-handover] === rollout {tid}  ({n} clips, "
              f"capture_stride={self.capture_stride}, is_success=1) ===")
        print(f"[RM-handover]   k | last | rollout_frames                | max_sim | "
              f"best_demo | demo_env0 | demo_progress | reward | status")

        with torch.inference_mode():
            for k in range(n):
                is_last = (k == last)
                max_sim, best_demo_idx, clip_threshold, idxs_envstep = \
                    self._score_query_clip(rollout_frames, k, max_idx_capture)
                matched = max_sim >= clip_threshold
                best_env0 = self.demo_clip_first_envstep(best_demo_idx)

                if matched:
                    # Normalize to [0, 1] so reward stays in [-0.5, +0.5] per step.
                    progress = float(self._inner.demo_progress[best_demo_idx]) / max_demo_progress
                    rewards[k] = -0.5 + progress
                    status = "match"
                else:
                    progress = 0.0
                    rewards[k] = -0.5
                    status = "miss"

                rollout_str = self._fmt_frames(idxs_envstep)
                best_str = f"{best_demo_idx:>5d}" if matched else f"{'miss':>5}"
                env0_str = f"{best_env0:>5d}" if matched else f"{'-':>5}"
                prog_str = f"{progress:>5.3f}" if matched else f"{'-':>5}"
                tag = "(final)" if is_last else ""
                print(f"[RM-handover]  {k:>2d} | {str(is_last):>5} | {rollout_str} |  "
                      f"{max_sim:.3f}  | {best_str}     | {env0_str}     | "
                      f"{prog_str:>9}     | {rewards[k]:>+5.2f}  | {status}{tag}")

        print(f"[RM-handover] rewards={[round(float(r), 2) for r in rewards.tolist()]} "
              f"sum={float(rewards.sum()):.2f} is_success=1")
        return rewards
