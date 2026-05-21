"""Eraser-task-specific reward model: milestone-based rewards.

This variant REPLACES the general progress-delta reward (see rm_wrapper.py's
AirbotRewardModel) with a hard-coded, milestone-based scheme tailored to the
`pick_eraser_out_of_box` task (8 query steps, query_freq=25, capture_stride=5).

Per-query-step reward, for a rollout with `n` query steps (n=8 full, n<8 if the
operator early-stopped with 'q'). `hit` = the rollout clip matched
(max_sim >= per-clip threshold) a demo clip whose first frame is at env-step
>= the step's target:

  query step (1-idx)   array idx   target demo clip (first frame env-step)
       6                   5              110   ([110,120,130,140,150])
       7                   6              130   ([130,140,150,160,170])
       8                   7              150   ([150,160,170,180,190])

NON-LAST steps:
  * idx 0-4 (steps 1-5):        always -1
  * idx 5 (6th):                0 if hit(>=110 env-step) else -1
  * idx 6 (7th):                0 if hit(>=130 env-step) else -1

LAST step (idx n-1); success = operator label:
  * idx 5/6/7, success,  hit  -> +0.5
  * idx 5/6/7, success,  miss ->  0.0
  * idx 7 (full-8), fail, hit -> -0.5
  * idx 7 (full-8), fail, miss-> -1.0
  * idx 5/6 (early-stop), fail-> normal non-last rule (0 if hit else -1)
  * idx <=4 (early-stop), success -> 0.0
  * idx <=4 (early-stop), fail    -> -1.0

The success logic is baked in here (handles_success_internally = True), so
train_utils_airbot.collect_traj skips its +1.0 final-step override for this
variant.
"""
import numpy as np

from examples.airbot.rm_wrapper import AirbotRewardModel


class EraserRewardModel(AirbotRewardModel):
    """Milestone-based reward model for the pick_eraser_out_of_box task."""

    # compute_rewards below applies the success label itself.
    handles_success_internally = True

    # query-step index -> target demo-clip FIRST-FRAME env-step for a "hit".
    # idx 5 = 6th step, idx 6 = 7th, idx 7 = 8th.
    STEP_TARGET_ENVSTEP = {5: 110, 6: 130, 7: 150}

    def compute_rewards(self, rollout_frames, num_query_steps=None,
                        traj_id=None, is_success=False):
        import torch

        if num_query_steps is None:
            num_query_steps = self.num_query_steps
        n = int(num_query_steps)

        rewards = -np.ones(n, dtype=np.float32)   # default everything to -1
        if n == 0 or len(rollout_frames) == 0:
            return rewards

        max_idx_capture = len(rollout_frames) - 1
        last = n - 1

        tid = f"{traj_id}" if traj_id is not None else "?"
        print(f"[RM-eraser] === rollout {tid}  ({n} clips, "
              f"capture_stride={self.capture_stride}, is_success={int(is_success)}) ===")
        print(f"[RM-eraser]   k | last | rollout_frames                | max_sim | "
              f"best_demo_env | thr   | hit | reward")

        with torch.inference_mode():
            for k in range(n):
                is_last = (k == last)
                target_envstep = self.STEP_TARGET_ENVSTEP.get(k)

                if target_envstep is None:
                    # steps 1-5 (idx 0-4): no demo matching — no GPU needed.
                    if is_last and is_success:
                        rewards[k] = 0.0          # early-stop before 6th, success
                    else:
                        rewards[k] = -1.0
                    print(f"[RM-eraser]  {k:>2d} | {str(is_last):>5} | "
                          f"{'(not scored)':<29} |    -    |       -       |   -   |  -  | "
                          f"{rewards[k]:>+5.2f}")
                    continue

                # idx 5/6/7 — score the clip.
                max_sim, best_demo_idx, clip_threshold, idxs_envstep = \
                    self._score_query_clip(rollout_frames, k, max_idx_capture)
                demo_env0 = self.demo_clip_first_envstep(best_demo_idx)
                matched = max_sim >= clip_threshold
                hit = matched and demo_env0 >= target_envstep

                if not is_last:
                    rewards[k] = 0.0 if hit else -1.0
                else:
                    if is_success:
                        rewards[k] = 0.5 if hit else 0.0
                    elif k == 7:                  # full-8 final step, failed
                        rewards[k] = -0.5 if hit else -1.0
                    else:                         # idx 5/6 early-stop, failed
                        rewards[k] = 0.0 if hit else -1.0

                print(f"[RM-eraser]  {k:>2d} | {str(is_last):>5} | "
                      f"{self._fmt_frames(idxs_envstep)} |  {max_sim:.3f}  | "
                      f"{demo_env0:>5d} (>= {target_envstep}) | {clip_threshold:>+5.2f} | "
                      f"{('hit' if hit else 'no'):>3} | {rewards[k]:>+5.2f}")

        print(f"[RM-eraser] rewards={[round(float(r), 2) for r in rewards.tolist()]} "
              f"sum={float(rewards.sum()):.2f} is_success={int(is_success)}")
        return rewards
