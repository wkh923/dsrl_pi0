"""Smoke-test Robometer scoring on a saved reference demo.

Must run in the `robometer` conda env (NOT dsrl_pi0 — see robometer_wrapper.py).
A successful demo should show progress rising roughly monotonically from ~0 to
~1; flat, noisy, or inverted progress means the wiring or the model config is
wrong and training would learn from garbage.

Usage:
    conda activate robometer
    python examples/airbot/smoke_test_robometer.py \
        --demo data/rm_demos/pick_eraser_out_of_box/demo_seed0 \
        --instruction "pick eraser out of box" \
        --num_query_steps 8 --capture_stride 5 --query_freq 25
"""
import argparse
import pathlib
import time

import numpy as np

MODEL = '/home/jpy/RM/Airbot-VLA-RL/Reward-Model-MVP/checkpoints/Robometer-4B'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--demo', required=True, help='dir of frame_*.jpg')
    ap.add_argument('--instruction', required=True)
    ap.add_argument('--num_query_steps', type=int, required=True)
    ap.add_argument('--query_freq', type=int, default=25)
    ap.add_argument('--capture_stride', type=int, default=5)
    ap.add_argument('--model_path', default=MODEL)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    ap.add_argument('--progress_scale', type=float, default=0.65)
    ap.add_argument('--smooth_window', type=int, default=5)
    ap.add_argument('--monotonic', action='store_true')
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from PIL import Image
    from examples.airbot.robometer_wrapper import RobometerRewardModel

    d = pathlib.Path(args.demo)
    files = sorted(d.glob('frame_*.jpg')) or sorted(d.glob('frame_*.png'))
    if not files:
        raise SystemExit(f'no frame_*.jpg/png in {d}')
    frames = [np.asarray(Image.open(f).convert('RGB')) for f in files]
    print(f'[smoke] {len(files)} frames from {d}, shape={frames[0].shape}')

    t0 = time.time()
    # Exercise the SAME class training uses, so what is validated here is what runs.
    rm = RobometerRewardModel(
        instruction=args.instruction,
        robometer_model_path=args.model_path,
        query_freq=args.query_freq,
        num_query_steps=args.num_query_steps,
        capture_stride=args.capture_stride,
        monotonic=args.monotonic,
        progress_scale=args.progress_scale,
        smooth_window=args.smooth_window,
        device=args.device,
    )
    print(f'[smoke] wrapper ready in {time.time() - t0:.1f}s')

    t0 = time.time()
    rewards = rm.compute_rewards(frames, num_query_steps=args.num_query_steps,
                                 traj_id=0, is_success=True)
    infer_s = time.time() - t0

    print(f'\n[smoke] inference {infer_s:.1f}s')
    print(f'[smoke] rewards ({args.num_query_steps}): {np.round(rewards, 2).tolist()}')
    print(f'[smoke] sum={rewards.sum():+.2f}  '
          f'with_success_bonus={rewards.sum() + 1.0:+.2f}  '
          f'range=[{rewards.min():+.2f}, {rewards.max():+.2f}]')
    print(f'[smoke] last_max_reached_progress={rm.last_max_reached_progress:.3f}')
    print('\n[smoke] NOTE: absolute sign is NOT the acceptance criterion. What matters')
    print('[smoke]       is that successful rollouts score HIGHER than failed ones.')
    print('[smoke]       Run this on a known failure and compare the sums.')


if __name__ == '__main__':
    main()
