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
export PYTHONPATH=.

# ============================================================
# Shared configuration (keep in sync with run_airbot.sh)
# ============================================================

PI0_CONFIG_PATH="/home/jpy/RM/airbot/airbot-VLA-RL/airbot-pi0/openpi/data/pick_and_place/config.py"
PI0_CHECKPOINT_DIR="/home/jpy/RM/airbot/airbot-VLA-RL/airbot-pi0/openpi/checkpoints/1-1_pick_and_place/pnp_100/19999"

ROBOT_PORTS="50051"
CAMERA_INDEX="243322074422 243522071794"
CAMERA_NAMES="base_0_rgb left_wrist_0_rgb"
INSTRUCTION="pick and place"

RESET_ACTION="-0.001618136651813984 -1.0361113548278809 0.8421794176101685 -1.6158959865570068 0.6345375776290894 1.6957406997680664 0.0"

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
  --query_freq 25 \
  --max_timesteps 200 \
  --control_rate 20 \
  --num_episodes 10 \
  --output_dir ./logs/eval_airbot_${MODE} \
  ${DSRL_ARGS}
