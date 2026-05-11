"""Persistence helpers for crash-resilient DSRL Airbot training.

Saves/loads the training state needed to resume after a crash:
  * replay buffer (only the [:size] inserted slots, not the whole pre-allocated capacity)
  * counters: i, total_num_traj, total_env_steps
  * agent JAX RNG (PRNGKey)

SAC network parameters are saved/restored via the existing
flax.training.checkpoints mechanism (agent.save_checkpoint /
agent.restore_checkpoint), not in this module.

All writes are atomic (write to .tmp, os.replace → rename) so a crash
mid-write leaves the previous file intact.
"""
from __future__ import annotations

import os
import pickle
from typing import Any

import numpy as np


STATE_META_FILE = "state_meta.pkl"
BUFFER_FILE = "replay_buffer.pkl"


def _atomic_save(path: str, obj: Any) -> None:
    """Pickle ``obj`` to ``path`` atomically (write to .tmp + os.replace)."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _slice_data(data_dict: dict, n: int) -> dict:
    """Recursively slice every ndarray in a (possibly nested) dict to ``[:n]``."""
    out = {}
    for k, v in data_dict.items():
        if isinstance(v, dict):
            out[k] = _slice_data(v, n)
        else:
            out[k] = np.asarray(v[:n])
    return out


def _copy_into(target: dict, source: dict, n: int) -> None:
    """Recursively copy ``source[k][:n]`` into ``target[k][:n]`` (in-place)."""
    for k, v in source.items():
        if isinstance(v, dict):
            _copy_into(target[k], v, n)
        else:
            target[k][:n] = v


def save_buffer(outputdir: str, replay_buffer) -> None:
    """Pickle only the ``[:size]`` inserted slots of the ``ReplayBuffer``.

    Avoids dumping the entire pre-allocated capacity (e.g. 33333 slots ×
    128×128×9 bytes ≈ 5 GB) when only a few hundred slots are valid.
    """
    n = replay_buffer.size
    payload = {
        "capacity": replay_buffer.capacity,
        "size": n,
        "_start": replay_buffer._start,
        "_traj_counter": replay_buffer._traj_counter,
        "traj_bounds": dict(replay_buffer.traj_bounds),
        "data": _slice_data(replay_buffer.data, n),
    }
    _atomic_save(os.path.join(outputdir, BUFFER_FILE), payload)


def load_buffer(outputdir: str, replay_buffer) -> None:
    """Restore saved ``[:size]`` data into an empty ``ReplayBuffer``.

    The caller is expected to have constructed an empty ``ReplayBuffer``
    with the same observation/action spaces and capacity. This function
    copies the saved slices in-place into ``replay_buffer.data`` and
    restores the counters.
    """
    path = os.path.join(outputdir, BUFFER_FILE)
    with open(path, "rb") as f:
        payload = pickle.load(f)
    n = payload["size"]
    if n == 0:
        return
    if payload["capacity"] != replay_buffer.capacity:
        raise ValueError(
            f"Buffer capacity mismatch: saved {payload['capacity']}, "
            f"current {replay_buffer.capacity}"
        )
    _copy_into(replay_buffer.data, payload["data"], n)
    replay_buffer.size = n
    replay_buffer._start = payload["_start"]
    replay_buffer._traj_counter = payload["_traj_counter"]
    replay_buffer.traj_bounds = dict(payload["traj_bounds"])


def save_meta(outputdir: str, **kwargs) -> None:
    """Pickle the metadata dict (counters, RNG, etc.) atomically."""
    _atomic_save(os.path.join(outputdir, STATE_META_FILE), kwargs)


def load_meta(outputdir: str) -> dict:
    """Load the previously-saved metadata dict."""
    with open(os.path.join(outputdir, STATE_META_FILE), "rb") as f:
        return pickle.load(f)


def has_state(outputdir: str) -> bool:
    """True if a complete saved state exists in ``outputdir``."""
    return all(
        os.path.exists(os.path.join(outputdir, f))
        for f in (STATE_META_FILE, BUFFER_FILE)
    )
