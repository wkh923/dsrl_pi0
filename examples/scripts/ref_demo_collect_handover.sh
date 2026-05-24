#!/bin/bash
# DIAGNOSTIC TEST: Evaluate pi0 SFT with REPEATED noise (mimicking DSRL i=0).
# Same robot/camera/checkpoint config as eval_airbot.sh; the only difference
# is that pi0 mode now feeds pi0 a single (1, 32) random Gaussian repeated
# 50 times — the exact noise structure used by run_airbot.sh i=0. Use this
# to confirm whether the "arm goes weird" behavior is caused by repeated
# noise alone (rather than SAC, prompt, image preprocessing, etc.).
#
# Usage:
#   bash examples/scripts/eval_airbot_test.sh          # defaults to pi0 mode (repeated noise)
#   bash examples/scripts/eval_airbot_test.sh pi0      # same as above
#   bash examples/scripts/eval_airbot_test.sh dsrl /path/to/sac_checkpoint

MODE=${1:-pi0}
SAC_CHECKPOINT_DIR=${2:-""}

device_id=0

export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=.:./Airbot

# ============================================================
# Shared configuration (keep in sync with run_airbot.sh)
# ============================================================

# Task tag — change when switching tasks. Used to namespace the demo-candidate
# save dir (data/rm_demos/<TASK_TAG>/candidates/demo_<timestamp>/).
TASK_TAG=handover

# PI0_CONFIG_PATH="/home/jpy/RM/airbot/airbot-VLA-RL/airbot-pi0/openpi/data/pick_and_place/config.py"
# PI0_CHECKPOINT_DIR="/home/jpy/RM/airbot/airbot-VLA-RL/airbot-pi0/openpi/checkpoints/1-1_pick_and_place/pnp_100/19999"

# RESET_ACTION="-0.001618136651813984 -1.0361113548278809 0.8421794176101685 -1.6158959865570068 0.6345375776290894 1.6957406997680664 0.0"

# # Path to your VLA-RL task config.py (defines TASK_NAME, CAMERA_TOPICS, DELTA_ACTION_MASK, etc.)
# PI0_CONFIG_PATH="/home/jpy/RM/Airbot-VLA-RL/VLA/airbot-pi0/openpi/data/pick_eraser_out_of_box/config.py"

# # Path to your SFT checkpoint directory
# PI0_CHECKPOINT_DIR="/home/jpy/RM/Airbot-VLA-RL/VLA/airbot-pi0/checkpoints/pick_eraser_out_of_box/pick_eraser/30000"

PI0_CONFIG_PATH="/home/jpy/RM/Airbot-VLA-RL/VLA/airbot-pi0/openpi/data/handover/config.py"
PI0_CHECKPOINT_DIR="/home/jpy/RM/Airbot-VLA-RL/VLA/airbot-pi0/checkpoints/handover/39999"


# Robot ports (single-arm: one port, dual-arm: two ports)
ROBOT_PORTS="50051 50053"  # dual-arm

# Camera device indices (matching your physical setup; mapping mirrors
# Airbot-VLA-RL/Airbot/spacemouse/test_inference_with_spacemouse.py dual-arm).
# base_0_rgb=243222074218, left_wrist_0_rgb=243522071794, right_wrist_0_rgb=243222071389
CAMERA_INDEX="243222074218 243522071794 243222071389"

# Camera names (must match what's defined in your config.py's CAMERA_TOPICS)
CAMERA_NAMES="base_0_rgb left_wrist_0_rgb right_wrist_0_rgb"

# Task instruction
INSTRUCTION="please hand over the object"
# INSTRUCTION="please fold the clothes"
RESET_ACTION="-0.001618 -1.036 0.842 -1.615 0.634 1.695 0.0 -0.001618 -1.036 0.842 -1.615 0.634 1.695 0.0"

# Demo-candidate save dir. On every rollout the user labels SUCCESS ("1"), the
# frames captured at multiples of 5 env-steps are dumped here as
#   <SAVE_DEMO_DIR>/demo_<YYYYMMDD_HHMMSS>/frame_<env_step:06d>.jpg + meta.txt
# Pick the best candidate later, then copy/symlink it to
#   data/rm_demos/<TASK_TAG>/demo_seed0/
# for the training scripts to consume.
SAVE_DEMO_DIR="./data/rm_demos/${TASK_TAG}/candidates"


# ============================================================

DSRL_ARGS=""
if [ "$MODE" = "dsrl" ]; then
    if [ -z "$SAC_CHECKPOINT_DIR" ]; then
        echo "Error: dsrl mode requires SAC checkpoint path as second argument"
        echo "Usage: bash examples/scripts/eval_airbot.sh dsrl /path/to/sac_checkpoint"
        exit 1
    fi
    DSRL_ARGS="--sac_checkpoint_dir ${SAC_CHECKPOINT_DIR}"
fi

echo "=== Evaluating ${MODE} policy ==="

python3 examples/eval_airbot_test.py \
  --mode ${MODE} \
  --pi0_mode local \
  --pi0_config_path "${PI0_CONFIG_PATH}" \
  --pi0_checkpoint_dir "${PI0_CHECKPOINT_DIR}" \
  --robot_ports ${ROBOT_PORTS} \
  --camera_names ${CAMERA_NAMES} \
  --camera_index ${CAMERA_INDEX} \
  --instruction "${INSTRUCTION}" \
  --reset_action ${RESET_ACTION} \
  --query_freq 25 \
  --max_timesteps 300 \
  --control_rate 20 \
  --num_episodes 10 \
  --output_dir ./logs/eval_airbot_test_${MODE} \
  --save_demo_dir "${SAVE_DEMO_DIR}" \
  --save_demo_capture_stride 5 \
  ${DSRL_ARGS}
