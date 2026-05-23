"""Clothes-folding reward model: progress reward + a success-gated late regime.

Extends the general progress-delta reward (AirbotRewardModel). The late-stage
regime is ENTIRELY conditional on the operator's success label:

  is_success = 0 (FAILURE): regime DISABLED. Every query step uses the
      original progress reward (hit -> -0.5+delta, match/miss -> -0.5) —
      identical to the base `progress` variant.

  is_success = 1 (SUCCESS): steps before the high-water mark first reaches an
      in-zone demo clip use the original progress reward; from the first
      in-zone hit onward, NON-FINAL steps use the regime:
          hit   (advances high-water)  -> +0.50
          match (matched, no advance)  -> +0.25   (ANY clip, in- or below-zone)
          miss  (sim < threshold)      ->  0.00

"In-zone" = the demo clip's first-frame env-step >= REGIME_PROGRESS_FRACTION *
max_timesteps (0.85 * 1000 = env-step 850).

The FINAL query step always uses the original progress reward;
train_utils_airbot.collect_traj adds the usual +1.0 success override
(handles_success_internally stays False).
"""
import numpy as np

from examples.airbot.rm_wrapper import AirbotRewardModel


class ClothesRewardModel(AirbotRewardModel):
    """Progress reward + success-gated late-stage regime for fold_clothes."""

    # The final step still gets train_utils' +1.0 success override.
    handles_success_internally = False

    # Fraction of max_timesteps defining the regime env-step threshold.
    # 0.85 * max_timesteps(1000) = env-step 850.
    REGIME_PROGRESS_FRACTION = 0.85

    def compute_rewards(self, rollout_frames, num_query_steps=None,
                        traj_id=None, is_success=False):
        import torch

        if num_query_steps is None:
            num_query_steps = self.num_query_steps
        n = int(num_query_steps)

        rewards = -0.5 * np.ones(n, dtype=np.float32)
        if n == 0 or len(rollout_frames) == 0:
            return rewards

        max_idx_capture = len(rollout_frames) - 1
        N = max(1, self.num_demo_clips)
        expected_delta = 1.0 / max(1, n)
        regime_envstep = self.REGIME_PROGRESS_FRACTION * self.max_timesteps   # 850 @ 1000
        last = n - 1

        max_reached_idx = -1
        max_reached_progress = 0.0
        reached_regime = False           # latches once high-water clip enters the zone

        tid = f"{traj_id}" if traj_id is not None else "?"
        print(f"[RM-clothes] === rollout {tid}  ({n} clips, "
              f"capture_stride={self.capture_stride}, num_demo_clips={N}, "
              f"regime_envstep>={regime_envstep:.0f}, is_success={int(is_success)}) ===")
        print(f"[RM-clothes]   k | last | rollout_frames                | max_sim | "
              f"best_demo | demo_env0 | in_regime | reward | status")

        with torch.inference_mode():
            for k in range(n):
                is_last = (k == last)
                max_sim, best_demo_idx, clip_threshold, idxs_envstep = \
                    self._score_query_clip(rollout_frames, k, max_idx_capture)
                matched = max_sim >= clip_threshold
                advanced = matched and best_demo_idx > max_reached_idx
                best_env0 = self.demo_clip_first_envstep(best_demo_idx)
                clip_in_zone = matched and best_env0 >= regime_envstep
                prev_idx = max_reached_idx

                # Regime is active for this step if the high-water already
                # entered the zone, OR this step itself crosses into it.
                in_regime_now = reached_regime or (advanced and clip_in_zone)

                if is_last:
                    # Final step: original progress reward; regime never applies.
                    # train_utils adds +1.0 on success.
                    if advanced:
                        delta_progress = (best_demo_idx - max(0, prev_idx)) / N
                        delta = min(1.0, delta_progress / expected_delta)
                        rewards[k] = -0.5 + delta
                        status = "hit(final)"
                    else:
                        rewards[k] = -0.5
                        status = ("match(final)" if matched else "miss(final)")
                elif is_success and in_regime_now:
                    # SUCCESS regime: hit 0.5 / match 0.25 (any clip) / miss 0.
                    if advanced:
                        rewards[k] = 0.5
                        status = "R-hit"
                    elif matched:
                        rewards[k] = 0.25
                        status = "R-match"
                    else:
                        rewards[k] = 0.0
                        status = "R-miss"
                else:
                    # FAILURE (regime disabled), or SUCCESS pre-regime:
                    # original progress reward.
                    if advanced:
                        delta_progress = (best_demo_idx - max(0, prev_idx)) / N
                        delta = min(1.0, delta_progress / expected_delta)
                        rewards[k] = -0.5 + delta
                        status = "hit"
                    else:
                        rewards[k] = -0.5
                        status = ("match" if matched else "miss")

                if advanced:
                    max_reached_idx = best_demo_idx
                    max_reached_progress = self._inner.demo_progress[best_demo_idx]
                # Latch the regime once the high-water clip is in the zone.
                if (max_reached_idx >= 0 and
                        self.demo_clip_first_envstep(max_reached_idx) >= regime_envstep):
                    reached_regime = True

                rollout_str = self._fmt_frames(idxs_envstep)
                best_str = f"{best_demo_idx:>5d}" if matched else f"{'miss':>5}"
                env0_str = f"{best_env0:>5d}" if matched else f"{'-':>5}"
                print(f"[RM-clothes]  {k:>2d} | {str(is_last):>5} | {rollout_str} |  "
                      f"{max_sim:.3f}  | {best_str}     | {env0_str}     | "
                      f"{str(reached_regime):>9} | {rewards[k]:>+5.2f}  | {status:<12}")

        self.last_max_reached_progress = max_reached_progress
        self.last_max_demo_progress = (
            max(self.demo_progress) if self.demo_progress else 0.0)
        print(f"[RM-clothes] rewards sum={float(rewards.sum()):.2f} "
              f"reached_regime={reached_regime} is_success={int(is_success)}")
        return rewards
