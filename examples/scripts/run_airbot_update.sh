#!/bin/bash
# DSRL training on Airbot robot — VARIANT with extended random-noise warmup phase
#
# Difference from run_airbot.sh:
#   * --num_random_rollouts 20  → first 20 rollouts use random Gaussian noise
#                                 (instead of just rollout 1). SAC is still
#                                 trained during these rollouts (240 grad
#                                 steps per rollout) but does not yet steer
#                                 pi0. The 5000-step warmup is moved from
#                                 "after rollout 1" to "after rollout 20"
#                                 (decision: option B).
#   * --prefix dsrl_pi0_airbot_warm20  → distinct W&B / output dir naming.
#
# Same W&B project as run_airbot.sh so you can compare runs side-by-side.
#
# Prerequisites:
# 1. SFT checkpoint from VLA-RL
# 2. Task config.py from VLA-RL
# 3. airbot_py SDK installed
# 4. airbot_data_collection package installed (from VLA-RL/airbot-data/data-collection)
# 5. dsrl_pi0's openpi installed (pip install -e openpi)

proj_name=DSRL_pi0_Airbot
device_id=0

export EXP=./logs/$proj_name
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=.:./Airbot

# ============================================================
# Fill in your configuration below
# ============================================================

# Path to your VLA-RL task config.py (defines TASK_NAME, CAMERA_TOPICS, DELTA_ACTION_MASK, etc.)
PI0_CONFIG_PATH="/home/jpy/RM/Airbot-VLA-RL/VLA/airbot-pi0/openpi/data/pick_eraser_out_of_box/config.py"

# Path to your SFT checkpoint directory
PI0_CHECKPOINT_DIR="/home/jpy/RM/Airbot-VLA-RL/VLA/airbot-pi0/checkpoints/pick_eraser_out_of_box/pick_eraser/30000"

# Robot ports (single-arm: one port, dual-arm: two ports)
ROBOT_PORTS="50051 50053"  # dual-arm

# Camera device indices (matching your physical setup; mapping mirrors
# Airbot-VLA-RL/Airbot/spacemouse/test_inference_with_spacemouse.py dual-arm).
# base_0_rgb=243222074218, left_wrist_0_rgb=243522071794, right_wrist_0_rgb=243222071389
CAMERA_INDEX="243222074218 243522071794 243222071389"

# Camera names (must match what's defined in your config.py's CAMERA_TOPICS)
CAMERA_NAMES="base_0_rgb left_wrist_0_rgb right_wrist_0_rgb"

# Task instruction
INSTRUCTION="pick eraser out of box"

# Auto-reset between episodes. 7 floats per arm: [j1..j6, gripper].
# Single-arm: 7 floats; dual-arm: 14 floats. Comment out the corresponding
# `--reset_action` line below to disable auto-reset (operator resets by hand).
# Defaults below mirror Airbot-VLA-RL/airbot/deploy/airbot_inference_sync.py.
RESET_ACTION="-0.001618 -1.036 0.842 -1.615 0.634 1.695 0.0 -0.001618 -1.036 0.842 -1.615 0.634 1.695 0.0"

# Number of initial rollouts that use random Gaussian noise (rather than SAC
# output) to drive pi0. Each rollout still triggers 240 SAC gradient steps,
# so SAC critic accumulates ~9560 updates (5000 warmup + 19*240) on a richer
# buffer (~160 transitions) before its own outputs start steering pi0.
NUM_RANDOM_ROLLOUTS=20

# ============================================================

python3 examples/launch_train_airbot.py \
--algorithm pixel_sac \
--env airbot \
--prefix dsrl_pi0_airbot_warm20 \
--wandb_project ${proj_name} \
--batch_size 256 \
--discount 0.99 \
--seed 42 \
--max_steps 500000 \
--eval_interval 2000 \
--log_interval 100 \
--checkpoint_interval 5000 \
--multi_grad_step 30 \
--resize_image 128 \
--action_magnitude 2.5 \
--query_freq 25 \
--hidden_dims 1024 \
--num_qs 2 \
--num_random_rollouts ${NUM_RANDOM_ROLLOUTS} \
--pi0_mode local \
--pi0_config_path "${PI0_CONFIG_PATH}" \
--pi0_checkpoint_dir "${PI0_CHECKPOINT_DIR}" \
--robot_type play \
--robot_ports ${ROBOT_PORTS} \
--camera_names ${CAMERA_NAMES} \
--camera_index ${CAMERA_INDEX} \
--max_timesteps 200 \
--control_rate 20 \
--reset_action ${RESET_ACTION} \
--instruction "${INSTRUCTION}"
