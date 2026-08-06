"""Robometer progress-oracle sidecar. Runs in the `robometer` conda env.

Why a sidecar: robometer needs python 3.10 + transformers>=4.57 + torch 2.8,
while dsrl_pi0 is python 3.11 with transformers==4.48.1 hard-pinned by openpi.
They cannot share an env, so the reward model runs as its own process and the
DSRL side reaches it over HTTP (see robometer_wrapper.RobometerRewardModel).

Scope is deliberately narrow: this serves RAW PROGRESS only. All reward shaping
(progress_scale, smoothing, signed-vs-monotonic delta) lives client-side in
robometer_wrapper, so reward tuning never requires restarting this server — the
model takes ~76s to load onto the GPU.

Usage:
    conda activate robometer
    python examples/airbot/robometer_server.py            # GPU, port 8765
    python examples/airbot/robometer_server.py --device cpu --port 9000
"""
import argparse
import base64
import io
import time

import numpy as np

MODEL = '/home/jpy/RM/Airbot-VLA-RL/Reward-Model-MVP/checkpoints/Robometer-4B'

_STATE = {}


def _load(model_path: str, device: str):
    import torch
    from robometer.utils.save import load_model_from_hf
    from robometer.utils.setup_utils import setup_batch_collator

    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f'[server] loading {model_path} on {dev} ...', flush=True)
    t0 = time.time()
    exp_config, tok, proc, rm = load_model_from_hf(model_path=model_path, device=dev)
    rm.eval()
    loss_cfg = getattr(exp_config, 'loss', None)
    _STATE.update(
        device=dev, model=rm, tokenizer=tok,
        collator=setup_batch_collator(proc, tok, exp_config, is_eval=True),
        is_discrete=(getattr(loss_cfg, 'progress_loss_type', 'l2').lower() == 'discrete'
                     if loss_cfg else False),
        num_bins=(getattr(loss_cfg, 'progress_discrete_bins', None)
                  or getattr(exp_config.model, 'progress_discrete_bins', 10)),
    )
    print(f'[server] ready in {time.time() - t0:.1f}s '
          f'(discrete={_STATE["is_discrete"]} bins={_STATE["num_bins"]})', flush=True)


def _progress(frames: np.ndarray, instruction: str, traj_id: str) -> list:
    import torch
    from robometer.data.dataset_types import ProgressSample, Trajectory
    from robometer.evals.eval_server import compute_batch_outputs

    traj = Trajectory(frames=frames, frames_shape=tuple(frames.shape),
                      task=instruction, id=traj_id,
                      metadata={'subsequence_length': int(frames.shape[0])},
                      video_embeddings=None)
    batch = _STATE['collator']([ProgressSample(trajectory=traj, sample_type='progress')])
    inp = batch['progress_inputs']
    for k, v in inp.items():
        if hasattr(v, 'to'):
            inp[k] = v.to(_STATE['device'])
    with torch.inference_mode():
        res = compute_batch_outputs(
            _STATE['model'], _STATE['tokenizer'], inp, sample_type='progress',
            is_discrete_mode=_STATE['is_discrete'], num_bins=_STATE['num_bins'])
    p = np.clip(np.asarray(res['progress_pred'][0], dtype=np.float32), 0.0, 1.0)
    return [float(x) for x in p]


def build_app():
    from fastapi import FastAPI, Request
    from PIL import Image

    app = FastAPI()

    @app.get('/health')
    async def health():
        return {'ok': 'model' in _STATE, 'device': str(_STATE.get('device'))}

    @app.post('/progress')
    async def progress(req: Request):
        body = await req.json()
        jpegs = body['frames']            # list of base64 JPEG strings
        instruction = body['instruction']
        traj_id = str(body.get('traj_id', 0))
        frames = np.stack([
            np.asarray(Image.open(io.BytesIO(base64.b64decode(j))).convert('RGB'))
            for j in jpegs], axis=0)
        t0 = time.time()
        try:
            p = _progress(frames, instruction, traj_id)
        except Exception as e:  # never take the training run down with us
            print(f'[server] ERROR {type(e).__name__}: {e}', flush=True)
            return {'error': f'{type(e).__name__}: {e}'}
        dt = time.time() - t0
        print(f'[server] traj={traj_id} frames={frames.shape[0]} -> '
              f'{len(p)} progress pts in {dt:.2f}s', flush=True)
        return {'progress': p, 'infer_s': dt}

    return app


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', default=MODEL)
    ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8765)
    a = ap.parse_args()

    import uvicorn
    _load(a.model_path, a.device)
    uvicorn.run(build_app(), host=a.host, port=a.port, log_level='warning')
