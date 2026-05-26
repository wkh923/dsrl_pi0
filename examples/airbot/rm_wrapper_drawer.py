"""Drawer-task-specific reward model: milestone-based rewards.

This variant REPLACES the general progress-delta reward (see rm_wrapper.py's
AirbotRewardModel) with a hard-coded, milestone-based scheme tailored to the
`open_drawer` task (7 query steps, query_freq=25, capture_stride=5,
max_timesteps=175).

Per-query-step reward, for a rollout with `n` query steps (n=7 full, n<7 if the
operator early-stopped with 'q'). `hit` = the rollout clip matched
(max_sim >= per-clip threshold) a demo clip whose first frame is at env-step
>= the step's threshold. Thresholds are RELAXED by ~10 env-steps vs the
nominal chunk start, so a partially-early demo clip still counts as a hit:

  query step (1-idx)   array idx   rollout clip env-steps        demo_env0 hit threshold
       5                   4         [100,110,120,130,140]            >= 80
       6                   5         [125,135,145,155,165]            >= 95
       7                   6         [150,160,170,180,190]            >=105

FAILURE override (is_success=False, i.e. operator pressed 0 at the end):
  * ALL steps -> -1.0, regardless of hit/miss. No clips are scored on the GPU.
  * Rationale: a failed rollout never reaches the drawer-open goal, so giving
    partial credit for matching mid-rollout demo clips encourages the SAC to
    repeat the same "looks-like-progress-but-fails" trajectory.

The rules below apply ONLY when is_success=True (operator pressed 1).

NON-LAST steps (is_success=True):
  * idx 0-3 (steps 1-4):        always -1
  * idx 4 (5th):                0 if hit(demo_env0>=80) else -1
  * idx 5 (6th):                0 if hit(demo_env0>=95) else -1

LAST step (idx n-1, is_success=True):
  * idx 4/5/6, hit  -> +0.5
  * idx 4/5/6, miss ->  0.0
  * idx <=3 (early-stop), success -> 0.0

The success logic is baked in here (handles_success_internally = True), so
train_utils_airbot.collect_traj skips its +1.0 final-step override for this
variant.
"""
import numpy as np

from examples.airbot.rm_wrapper import AirbotRewardModel


class DrawerRewardModel(AirbotRewardModel):
    """Milestone-based reward model for the open_drawer task."""

    # compute_rewards below applies the success label itself.
    handles_success_internally = True

    # query-step index -> demo_env0 hit threshold (inclusive lower bound).
    # idx 4 = 5th step, idx 5 = 6th, idx 6 = 7th (last for full rollout).
    # Thresholds are ~10 env-steps below the nominal chunk start (100/125/150),
    # so a partially-early demo clip still counts.
    STEP_TARGET_ENVSTEP = {4: 80, 5: 95, 6: 105}

    def compute_rewards(self, rollout_frames, num_query_steps=None,
                        traj_id=None, is_success=False):
        import torch

        if num_query_steps is None:
            num_query_steps = self.num_query_steps
        n = int(num_query_steps)

        rewards = -np.ones(n, dtype=np.float32)   # default everything to -1
        if n == 0 or len(rollout_frames) == 0:
            return rewards

        tid = f"{traj_id}" if traj_id is not None else "?"

        # FAILURE override: operator pressed 0 → every step gets -1, no scoring.
        if not is_success:
            print(f"[RM-drawer] === rollout {tid}  ({n} clips, "
                  f"is_success=0) — FAILURE override: all rewards = -1 (no clip scoring) ===")
            print(f"[RM-drawer] rewards={[round(float(r), 2) for r in rewards.tolist()]} "
                  f"sum={float(rewards.sum()):.2f} is_success=0")
            return rewards

        max_idx_capture = len(rollout_frames) - 1
        last = n - 1

        print(f"[RM-drawer] === rollout {tid}  ({n} clips, "
              f"capture_stride={self.capture_stride}, is_success=1) ===")
        print(f"[RM-drawer]   k | last | rollout_frames                | max_sim | "
              f"best_demo_env | thr   | hit | reward")

        with torch.inference_mode():
            for k in range(n):
                is_last = (k == last)
                target_envstep = self.STEP_TARGET_ENVSTEP.get(k)

                if target_envstep is None:
                    # steps 1-4 (idx 0-3): no demo matching — no GPU needed.
                    # is_success=True here (failure path returned above), so
                    # an early-stop on these steps gets 0.0; otherwise -1.0.
                    rewards[k] = 0.0 if is_last else -1.0
                    print(f"[RM-drawer]  {k:>2d} | {str(is_last):>5} | "
                          f"{'(not scored)':<29} |    -    |       -       |   -   |  -  | "
                          f"{rewards[k]:>+5.2f}")
                    continue

                # idx 4/5/6 — score the clip.
                max_sim, best_demo_idx, clip_threshold, idxs_envstep = \
                    self._score_query_clip(rollout_frames, k, max_idx_capture)
                demo_env0 = self.demo_clip_first_envstep(best_demo_idx)
                matched = max_sim >= clip_threshold
                hit = matched and demo_env0 >= target_envstep

                if not is_last:
                    rewards[k] = 0.0 if hit else -1.0
                else:                            # last step + success
                    rewards[k] = 0.5 if hit else 0.0

                print(f"[RM-drawer]  {k:>2d} | {str(is_last):>5} | "
                      f"{self._fmt_frames(idxs_envstep)} |  {max_sim:.3f}  | "
                      f"{demo_env0:>5d} (>= {target_envstep}) | {clip_threshold:>+5.2f} | "
                      f"{('hit' if hit else 'no'):>3} | {rewards[k]:>+5.2f}")

        print(f"[RM-drawer] rewards={[round(float(r), 2) for r in rewards.tolist()]} "
              f"sum={float(rewards.sum()):.2f} is_success={int(is_success)}")
        return rewards
