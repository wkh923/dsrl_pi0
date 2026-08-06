"""Dense-reward integration: Robometer adapter for DSRL Airbot.

Robometer (https://robometer.github.io, MIT) is a VLM-based reward model that
scores task progress from a rollout video + task-language description, with no
reference demo clip — unlike the Reference-Anchored RM (RARM) variants in
`rm_wrapper*.py`, which compare rollout clips against a demo.

This is the Robometer baseline for the RARM comparison. It follows TWO references:

  1. Scoring call pattern — Reward-Model-MVP @ 42ad7c9 (branch yiduo/drq_v2_libero),
     `reward_model_baseline/Libero/RewardModels/RobometerRewardModel.py`, the SAC
     "deferred" path (`compute_reward_from_images`): ONE forward pass per finished
     episode over all frames, which returns M subsampled progress values
     (M <= config max_frames = 8), then `np.interp` back up to per-frame
     resolution. The DrQ-v2 path in that same file (`get_progress` every
     ROBOMETER_CALL_INTERVAL=40 env-steps) is deliberately NOT used: DSRL scores
     once per finished rollout, and one 4B-VLM pass per rollout is far cheaper
     than one per query step on a single 4090.

  2. Reward scale — this repo's `rm_wrapper.AirbotRewardModel`:
        reward[k] = -0.5 + clip(delta_progress / expected_delta, lo, 1),
                    expected_delta = 1 / num_query_steps
     The `/expected_delta` normalization is required because raw per-query-step
     progress deltas are ~1/num_query_steps (~0.025 for 40 steps), which without
     scaling would pin every reward at ~-0.475 and give SAC no gradient. The -0.5
     offset is the same constant the sim subtracts as STEP_COST
     (Libero/constants/env_constants.py). The +1.0 final-step success override
     stays with the caller (train_utils_airbot.collect_traj) —
     hence `handles_success_internally = False`.

REGRESSION HANDLING (`monotonic`) — the one place RARM and Robometer genuinely
differ, so it is a switch rather than a hardcoded choice:

  * monotonic=False (default, faithful to Robometer + the sim): progress is
    compared against the PREVIOUS query step and the delta keeps its sign, so a
    rollout that undoes its own progress (eraser dropped back in the box, folded
    clothes shaken out, drawer sliding shut) is penalized below the floor:
    lo=-1 → reward in [-1.5, +0.5]. This mirrors the sim's robometer branch,
    `r = (current_progress - previous_progress) - STEP_COST`, which applies no
    max() and no clamp (drqv2_runner.py:827).
  * monotonic=True (RARM-comparable): progress is compared against a high-water
    mark that only ever rises, so stalls and regressions both floor at -0.5 and
    reward stays in [-0.5, +0.5], identical to
    `rm_wrapper.AirbotRewardModel`. Use this when you want Q-values on exactly
    the RARM scale at the cost of discarding Robometer's regression signal.

  Note the sim applies monotonic filtering ONLY to the RARM-family path
  (`image_progress[i] = max(image_progress[i], image_progress[i-1])`,
  drqv2_runner.py:249) and deliberately not to robometer — hence the default.

Setup: the `robometer` package is vendored inside Reward-Model-MVP. Install it
editably once (mirrors the LIBERO wrapper, which relies on site-packages rather
than mutating sys.path, avoiding cross-simulator import bleed):

    pip install -e <Reward-Model-MVP>/reward_model_baseline/MetaWorld/baseline/robometer

Weights: pass a local snapshot dir via --robometer_model_path (default is the
Robometer-4B snapshot under <Reward-Model-MVP>/checkpoints/), or a HF hub id.
"""
from typing import Optional, Sequence

import numpy as np


class RobometerRewardModel:
    """Scores DSRL rollouts against the task instruction via Robometer.

    Returns one float per query step: [-1.5, +0.5] by default (regressions
    penalized), or [-0.5, +0.5] with monotonic=True (RARM-comparable). See the
    module docstring for the formula and the regression-handling rationale.
    """

    # The caller applies the +1.0 is_success override on the final step.
    handles_success_internally = False

    def __init__(
        self,
        instruction: str,
        robometer_model_path: str,
        server_url: str = '',
        max_timesteps: int = 300,
        query_freq: int = 25,
        num_query_steps: int = 12,
        capture_stride: int = 1,
        relative_rewards: bool = False,
        max_input_frames: int = 8,
        success_threshold: float = 0.65,
        device: str = 'cuda',
        **_unused,
    ):
        if query_freq % capture_stride != 0:
            raise ValueError(
                f"query_freq ({query_freq}) must be divisible by "
                f"capture_stride ({capture_stride})")

        # server_url set -> HTTP client mode (the only mode that works from
        # dsrl_pi0, which cannot import robometer). Empty -> in-process, which
        # requires running inside the `robometer` env (smoke test only).
        self.server_url = (server_url or '').rstrip('/')

        self.instruction = instruction
        self.max_timesteps = max_timesteps
        self.query_freq = query_freq                            # env-step
        self.query_freq_capture = query_freq // capture_stride   # capture-idx
        self.num_query_steps = num_query_steps
        self.capture_stride = capture_stride
        # robometer's own default (example_libero_robometer_wrapper.py:223):
        # absolute progress as the reward. True switches to their
        # use_relative_rewards branch (consecutive difference).
        self.relative_rewards = bool(relative_rewards)
        self.max_input_frames = int(max_input_frames)
        # Their success-detection threshold (same file, :227).
        self.success_threshold = float(success_threshold)
        self.last_max_reached_progress = 0.0

        if self.server_url:
            import requests
            try:
                h = requests.get(f'{self.server_url}/health', timeout=10).json()
            except Exception as e:
                raise RuntimeError(
                    f"Robometer sidecar unreachable at {self.server_url}: {e}\n"
                    "Start it first, in the robometer env:\n"
                    "    conda activate robometer && python "
                    "examples/airbot/robometer_server.py") from e
            if not h.get('ok'):
                raise RuntimeError(f"Robometer sidecar not ready: {h}")
            print(f"[Robometer] using sidecar {self.server_url} (device={h.get('device')})")
            return

        import torch
        from robometer.utils.save import load_model_from_hf
        from robometer.utils.setup_utils import setup_batch_collator
        self._device = torch.device(
            device if torch.cuda.is_available() else 'cpu')

        print(f"[Robometer] loading {robometer_model_path} on {self._device} ...")
        exp_config, tokenizer, processor, reward_model = load_model_from_hf(
            model_path=robometer_model_path, device=self._device)
        reward_model.eval()
        self._reward_model = reward_model
        self._tokenizer = tokenizer
        self._batch_collator = setup_batch_collator(
            processor, tokenizer, exp_config, is_eval=True)

        # Progress head mode must come from the checkpoint, not be assumed:
        # the shipped Robometer-4B config is discrete with 10 bins, and decoding
        # it as continuous yields garbage progress values.
        loss_config = getattr(exp_config, 'loss', None)
        self._is_discrete = (
            getattr(loss_config, 'progress_loss_type', 'l2').lower() == 'discrete'
            if loss_config else False
        )
        self._num_bins = (
            getattr(loss_config, 'progress_discrete_bins', None)
            or getattr(exp_config.model, 'progress_discrete_bins', 10)
        )
        self._max_frames = int(
            getattr(getattr(exp_config, 'data', None), 'max_frames', 16))

        print(f"[Robometer] ready: task={instruction!r} "
              f"discrete={self._is_discrete} num_bins={self._num_bins} "
              f"max_frames={self._max_frames}")

    def compute_rewards(
        self,
        rollout_frames: Sequence[np.ndarray],
        num_query_steps: Optional[int] = None,
        traj_id: Optional[int] = None,
        is_success: bool = False,
    ) -> np.ndarray:
        """Per-query-step dense rewards for one rollout. Signature mirrors
        `rm_wrapper.AirbotRewardModel.compute_rewards`.

        Args:
            rollout_frames: HxWx3 uint8 RGB frames captured every
                `capture_stride` env-steps (rollout_frames[i] <-> env_step
                i*capture_stride).
            num_query_steps: defaults to self.num_query_steps.
            traj_id: rollout id, for the log line only.
            is_success: operator label. Unused here — the caller applies the
                +1.0 final-step override (handles_success_internally=False).

        Returns:
            (num_query_steps,) float32. -0.5 = no progress this step; up to
            +0.5 for a full expected step of progress. Regressions reach down to
            -1.5 unless monotonic=True, which floors them at -0.5.
        """
        if num_query_steps is None:
            num_query_steps = self.num_query_steps

        # -0.5 floor, matching RARM's "no progress" baseline.
        rewards = -0.5 * np.ones(num_query_steps, dtype=np.float32)

        frames_list = [f for f in rollout_frames if f is not None]
        # Robometer needs >= 2 frames to form a progress sample.
        if len(frames_list) < 2:
            self.last_max_reached_progress = 0.0
            return rewards

        # N_orig is the captured timeline; query-step indices are defined on it.
        N = int(len(frames_list))
        # Subsample to at most max_input_frames before the forward pass. Measured
        # on a 4090: 8 frames -> 9.35 GiB peak / 0.3s, 40 -> 13.58 GiB / 2.8s,
        # 60 -> OOM (no flash-attn, so sdpa attention over vision tokens is
        # quadratic in frame count). Coarser spacing also scores BETTER: on a
        # successful fold_clothes demo the reward sum was +3.27 at 8 frames vs
        # -6.11 at 40, because at fine spacing real per-step progress (~0.016)
        # is below the model's jitter and most steps read as regression. 8 also
        # matches the checkpoint's trained data.max_frames.
        if self.max_input_frames > 0 and N > self.max_input_frames:
            sel = np.unique(np.linspace(0, N - 1, self.max_input_frames).astype(int))
        else:
            sel = np.arange(N)
        frames = np.stack([np.asarray(frames_list[i]) for i in sel], axis=0)

        if self.server_url:
            progress = self._progress_via_server(frames, traj_id)
        else:
            progress = self._progress_in_process(frames, traj_id)
        M = len(progress)
        if M == 0:
            self.last_max_reached_progress = 0.0
            return rewards

        return self._shape(progress, sel, N, num_query_steps, traj_id, M)

    def _progress_via_server(self, frames, traj_id):
        """POST the (already subsampled) frames as JPEGs; get raw progress back."""
        import base64
        import io
        import requests
        from PIL import Image

        jpegs = []
        for f in frames:
            buf = io.BytesIO()
            Image.fromarray(f.astype(np.uint8)).save(buf, format='JPEG', quality=92)
            jpegs.append(base64.b64encode(buf.getvalue()).decode())
        r = requests.post(
            f'{self.server_url}/progress',
            json={'frames': jpegs, 'instruction': self.instruction,
                  'traj_id': str(traj_id if traj_id is not None else 0)},
            timeout=300).json()
        if 'error' in r:
            raise RuntimeError(f"sidecar: {r['error']}")
        return np.clip(np.asarray(r['progress'], dtype=np.float32), 0.0, 1.0)

    def _progress_in_process(self, frames, traj_id):
        import torch
        from robometer.data.dataset_types import ProgressSample, Trajectory
        from robometer.evals.eval_server import compute_batch_outputs

        traj = Trajectory(
            frames=frames,
            frames_shape=tuple(frames.shape),
            task=self.instruction,
            id=str(traj_id if traj_id is not None else 0),
            metadata={'subsequence_length': int(frames.shape[0])},
            video_embeddings=None,
        )
        sample = ProgressSample(trajectory=traj, sample_type='progress')
        batch = self._batch_collator([sample])
        progress_inputs = batch['progress_inputs']
        for key, value in progress_inputs.items():
            if hasattr(value, 'to'):
                progress_inputs[key] = value.to(self._device)

        with torch.inference_mode():
            results = compute_batch_outputs(
                self._reward_model, self._tokenizer, progress_inputs,
                sample_type='progress', is_discrete_mode=self._is_discrete,
                num_bins=self._num_bins)

        # progress_pred is per-BATCH-element, and each element holds M
        # subsampled progress values (M <= max_frames), NOT one per input
        # frame — so interpolate back onto the N-frame axis before sampling
        # at query-step boundaries.
        progress_pred = results.get('progress_pred', [[]])
        return np.clip(np.asarray(progress_pred[0], dtype=np.float32), 0.0, 1.0)

    def _shape(self, progress, sel, N, num_query_steps, traj_id, M):
        """Progress -> per-query-step rewards, following robometer's own protocol
        (robometer/evals/eval_utils.py::extract_rewards_from_output and
        scripts/example_libero_robometer_wrapper.py):

            reward = the LAST progress value of the subsequence, clamped [0, 1]

        i.e. the reward IS the absolute progress at that point, not a delta and
        not a pace score. Their wrapper exposes `use_relative_rewards` for the
        delta form but defaults it to False, so absolute is the reference
        behavior. Deliberately NO expected_delta pacing, no -0.5 offset, no
        smoothing and no high-water mark — those were RARM-specific heuristics
        and are not part of robometer's formulation.

        One property to be aware of: rewards are all >= 0, so the undiscounted
        sum grows with rollout length. Comparing sums across rollouts of
        different length is not meaningful; compare per-step values, or the
        final progress.
        """
        if M == 1:
            progress_per_frame = np.full(N, float(progress[0]), dtype=np.float32)
        else:
            # x-coords are the actual fed-frame indices, so subsampling does not
            # shift progress in time.
            xp = (sel.astype(float) if len(sel) == M
                  else np.linspace(0, N - 1, M))
            progress_per_frame = np.interp(
                np.arange(N, dtype=float), xp, progress).astype(np.float32)

        rewards = np.zeros(num_query_steps, dtype=np.float32)
        tid = f"{traj_id}" if traj_id is not None else "?"
        print(f"[Robometer] === rollout {tid} ({num_query_steps} query steps, "
              f"query_freq={self.query_freq}, capture_stride={self.capture_stride}, "
              f"frames={N}->{len(sel)} fed, model_progress_pts={M}, "
              f"relative={self.relative_rewards}) ===")
        print("[Robometer]   k | frame_idx | progress | reward")

        prev = 0.0
        for k in range(num_query_steps):
            # Progress at the END of query step k == "last value of the
            # subsequence ending there", matching extract_rewards_from_output.
            idx = min((k + 1) * self.query_freq_capture - 1, N - 1)
            curr = float(np.clip(progress_per_frame[idx], 0.0, 1.0))
            if self.relative_rewards:
                rewards[k] = curr - prev
                prev = curr
            else:
                rewards[k] = curr
            print(f"[Robometer] {k:>3} | {idx:>9} | {curr:>8.4f} | {rewards[k]:>+6.3f}")

        self.last_max_reached_progress = float(
            np.clip(progress_per_frame.max(), 0.0, 1.0))
        return rewards
