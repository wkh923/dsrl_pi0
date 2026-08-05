#!/bin/bash
# Evaluate pi0 SFT or DSRL policy on Airbot robot
#
# Usage:
#   bash examples/scripts/eval_airbot.sh          # defaults to pi0 mode
#   bash examples/scripts/eval_airbot.sh pi0      # base SFT policy
#   bash examples/scripts/eval_airbot.sh dsrl /path/to/sac_checkpoint

MODE=${1:-pi0}
SAC_CHECKPOINT_DIR=${2:-""}

device_id=0

export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=.:./Airbot

# ============================================================
# Shared configuration (keep in sync with run_airbot.sh)
# ============================================================

# PI0_CONFIG_PATH="/home/jpy/RM/airbot/airbot-VLA-RL/airbot-pi0/openpi/data/pick_and_place/config.py"
# PI0_CHECKPOINT_DIR="/home/jpy/RM/airbot/airbot-VLA-RL/airbot-pi0/openpi/checkpoints/1-1_pick_and_place/pnp_100/19999"

# RESET_ACTION="-0.001618136651813984 -1.0361113548278809 0.8421794176101685 -1.6158959865570068 0.6345375776290894 1.6957406997680664 0.0"

PI0_CONFIG_PATH="/home/jpy/RM/Airbot-VLA-RL/VLA/airbot-pi0/openpi/data/handover/config.py"
PI0_CHECKPOINT_DIR="/home/jpy/RM/Airbot-VLA-RL/VLA/airbot-pi0/checkpoints/handover/74999"

# Robot ports (single-arm: one port, dual-arm: two ports)
ROBOT_PORTS="50051 50053"  # dual-arm

# Camera device indices (matching your physical setup; mapping mirrors
# Airbot-VLA-RL/Airbot/spacemouse/test_inference_with_spacemouse.py dual-arm).
# base_0_rgb=243222074218, left_wrist_0_rgb=243522071794, right_wrist_0_rgb=243222071389
CAMERA_INDEX="243222074218 243522071794 243222071389"

# Camera names (must match what's defined in your config.py's CAMERA_TOPICS)
CAMERA_NAMES="base_0_rgb left_wrist_0_rgb right_wrist_0_rgb"

# Task instruction
# INSTRUCTION="pick eraser out of box"
INSTRUCTION="please hand over the object"
RESET_ACTION="-0.001618 -1.036 0.842 -1.615 0.634 1.695 0.0 -0.001618 -1.036 0.842 -1.615 0.634 1.695 0.0"


# ============================================================

DSRL_ARGS=""
if [ "$MODE" = "dsrl" ]; then
    if [ -z "$SAC_CHECKPOINT_DIR" ]; then
        echo "Error: dsrl mode requires SAC checkpoint path as second argument"
        echo "Usage: bash examples/scripts/eval_airbot.sh dsrl /path/to/sac_checkpoint"
        exit 1
    fi
    # Walk up from the checkpoint dir to find the nearest variant.json (the
    # training output dir). Handles both top-level checkpoints (parent has
    # variant.json) and milestones/checkpointN (parent's parent has it).
    VARIANT_DIR="$(dirname "${SAC_CHECKPOINT_DIR}")"
    while [ "${VARIANT_DIR}" != "/" ] && [ ! -f "${VARIANT_DIR}/variant.json" ]; do
        VARIANT_DIR="$(dirname "${VARIANT_DIR}")"
    done
    if [ ! -f "${VARIANT_DIR}/variant.json" ]; then
        echo "Error: could not locate variant.json above ${SAC_CHECKPOINT_DIR}"
        exit 1
    fi
    echo "Located training variant: ${VARIANT_DIR}/variant.json"
    DSRL_ARGS="--sac_checkpoint_dir ${SAC_CHECKPOINT_DIR} --training_variant_path ${VARIANT_DIR}/variant.json"
fi

echo "=== Evaluating ${MODE} policy ==="

python3 examples/eval_airbot.py \
  --mode ${MODE} \
  --pi0_mode local \
  --pi0_config_path "${PI0_CONFIG_PATH}" \
  --pi0_checkpoint_dir "${PI0_CHECKPOINT_DIR}" \
  --robot_ports ${ROBOT_PORTS} \
  --camera_names ${CAMERA_NAMES} \
  --camera_index ${CAMERA_INDEX} \
  --instruction "${INSTRUCTION}" \
  --reset_action ${RESET_ACTION} \
  --reset_release_grippers \
  --query_freq 25 \
  --max_timesteps 300 \
  --control_rate 20 \
  --num_episodes 30 \
  --output_dir ./logs/eval_airbot_${MODE} \
  ${DSRL_ARGS}
