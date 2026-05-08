# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DSRL (Diffusion Steering via Reinforcement Learning) steers the pre-trained π₀ generalist robot policy using latent-space RL. A pixel-based SAC agent learns to predict noise that guides π₀'s diffusion process, enabling task-specific adaptation without fine-tuning π₀ itself.

Paper: "Steering Your Diffusion Policy with Latent Space Reinforcement Learning" (CoRL 2025)

## Setup

```bash
conda create -n dsrl_pi0 python=3.11.11 && conda activate dsrl_pi0
pip install -e . && pip install -r requirements.txt && pip install "jax[cuda12]==0.5.0"
pip install -e openpi && pip install -e openpi/packages/openpi-client
pip install -e LIBERO && pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

The `jaxrl2` package is installed in editable mode via `setup.py`. `openpi/` and `LIBERO/` are git submodules.

## Running Training

```bash
bash examples/scripts/run_libero.sh       # Libero simulation
bash examples/scripts/run_aloha.sh        # Aloha simulation
bash examples/scripts/run_real.sh         # Real Franka robot (requires DROID setup + remote π₀ server)
bash examples/scripts/run_airbot.sh       # Real Airbot robot (requires VLA-RL SFT checkpoint)
bash examples/scripts/run_airbot_dual.sh  # Real Airbot dual-arm
```

Direct invocation:
```bash
python3 examples/launch_train_sim.py --algorithm pixel_sac --env libero --batch_size 256 \
  --max_steps 500000 --multi_grad_step 20 --resize_image 64 --query_freq 20
```

Key environment variables set by the shell scripts:
- `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl` — headless rendering
- `OPENPI_DATA_HOME=./openpi` — π₀ data/checkpoint path
- `EXP=./logs/<project>` — output directory for logs and checkpoints
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` — JAX memory management

## Architecture

### `jaxrl2/` — Core RL Library (JAX/Flax)

**Agent** (`agents/pixel_sac/`):
- `pixel_sac_learner.py` — `PixelSACLearner` class and `_update_jit()`: the JAX-compiled training step running actor, critic, and temperature updates with data augmentation (random crop + optional color jitter).
- `actor_updater.py` / `critic_updater.py` / `temperature_updater.py` — Individual SAC component updates.
- `temperature.py` — Learnable entropy temperature (auto-tuned).

**Networks** (`networks/`):
- `encoders/networks.py` — `PixelMultiplexer`: combines a CNN encoder with a downstream policy/critic network through an optional latent bottleneck. This is the wrapper used for both actor and critic.
- `encoders/` — Encoder options: `small` (simple CNN, default), `impala`/`impala_small`, `resnet_*_v1`, `resnet_*_v2`.
- `learned_std_normal_policy.py` — Gaussian policy with tanh squashing and learned std.
- `values.py` — `StateActionEnsemble`: ensemble of Q-functions (default 10 for sim, 2 for real).

**Data** (`data/`):
- `replay_buffer.py` — Online replay buffer with trajectory tracking.
- `augmentations.py` — Random crop and color jitter (JAX vmap-compatible).

### `examples/` — Training Entry Points and Loop

- `launch_train_sim.py` / `launch_train_real.py` / `launch_train_airbot.py` — CLI entry points (argparse). Define default hyperparameters and call `main()`.
- `train_sim.py` / `train_real.py` / `train_airbot.py` — `main()`: environment setup, agent creation, starts training loop.
- `train_utils_sim.py` / `train_utils_real.py` / `train_utils_airbot.py` — Core training logic. Key function: `trajwise_alternating_training_loop()`:
  1. **Rollout**: collect trajectory. SAC predicts noise to steer π₀'s diffusion at `query_freq` step intervals.
  2. **Reward assignment**: sparse rewards based on task success/failure at query steps only.
  3. **SGD phase**: train SAC on replay buffer with `multi_grad_step` (UTD ratio) gradient steps per env step.
  4. **Evaluation**: periodic rollouts measuring success rate, logged to W&B.

### `examples/airbot/` — Airbot Robot Interface and Transforms (self-contained)

- `play_operator.py` — `Robot` class: captures observations, sends joint actions, manages robot lifecycle. Uses `airbot_py` (gRPC) and `pyrealsense2`.
- `robot_config.py` — `RobotConfig` pydantic model (robot type, ports, camera names/indices, robot groups).
- `airbot_policy.py` — `AirbotInputs`/`AirbotOutputs` transforms for pi0. Maps airbot observation keys to pi0's 3 fixed camera slots (missing cameras zero-padded with False mask).
- `airbot_data_config.py` — Config creation for loading airbot SFT checkpoints with correct transforms (delta actions, normalization). Uses VLA-RL's task `config.py` format.

### `airbot_data_collection/` — Vendored Robot/Camera Abstraction Layer

Copied from VLA-RL's `airbot-data/data-collection` package. Provides the hardware abstraction used by `play_operator.py`:
- `airbot/robots/airbot_play.py` — `AIRBOTPlay` system: wraps `airbot_py.arm.AIRBOTArm` (gRPC) for joint state reading, joint position commands, mode switching.
- `airbot/sensors/cameras/realsense.py` — `RealSense` sensor: wraps `pyrealsense2` for camera frame capture.
- `basis.py` — Base classes (`Sensor`, `System`, `ConfigurableBasis`) and config enums.
- `common/utils/transformations.py` — Christoph Gohlke's transformation library (quaternions, Euler angles, matrices).

External hardware dependencies (not vendored): `airbot_py` (gRPC SDK wheel), `pyrealsense2` (Intel RealSense SDK).

### External Submodules

- `openpi/` — Pre-trained π₀ policy with DSRL modifications (noise injection, VLM feature extraction). SAC steers this model's diffusion process.
- `LIBERO/` — LIBERO benchmark environments for simulation.

## Key Hyperparameters by Environment

| Parameter | Libero | Aloha | Real (Franka) | Airbot |
|---|---|---|---|---|
| `multi_grad_step` (UTD) | 20 | 20 | 30 | 30 |
| `query_freq` | 20 | 50 | 10 | 25 |
| `action_magnitude` | 1.0 | 2.0 | 2.5 | 2.5 |
| `num_qs` | 10 | 10 | 2 | 2 |
| `resize_image` | 64 | 64 | 128 | 128 |
| `hidden_dims` | 128 | 128 | 1024 | 1024 |
| `discount` | 0.999 | 0.999 | 0.99 | 0.99 |

## Tech Stack

- **JAX 0.5.0 / Flax 0.10.2 / Optax 0.2.4** — Core ML framework
- **Distrax** — Probability distributions
- **W&B** — Experiment tracking (`--wandb_project`)
- **MuJoCo** — Physics simulation (v3.3.1 for Libero, v2.3.7 for Aloha)

## Notes

- No test suite or linting configuration exists in this repository.
- Mujoco version differs between environments — the shell scripts install the correct version.
