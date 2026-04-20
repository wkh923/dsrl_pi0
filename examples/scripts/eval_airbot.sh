#!/bin/bash
# Evaluate pi0 or DSRL policy on Airbot robot
#
# Usage:
#   bash examples/scripts/eval_airbot.sh pi0     # test base pi0 policy
#   bash examples/scripts/eval_airbot.sh dsrl    # test pi0 + DSRL

MODE=${1:-pi0}  # default to pi0-only

export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# ============================================================
# Fill in your configuration below
# ============================================================

PI0_CONFIG_PATH=""       # e.g., "./checkpoints/put_cup/config.py"
PI0_CHECKPOINT_DIR=""    # e.g., "./checkpoints/put_cup/9000"

# Only needed for dsrl mode
SAC_CHECKPOINT_DIR=""    # e.g., "./logs/DSRL_pi0_Airbot/dsrl_pi0_airbot_s42/checkpoint_50000"

INSTRUCTION="put the cup on the plate"
ROBOT_PORTS="50051"
CAMERA_NAMES="base_0_rgb left_wrist_0_rgb"
CAMERA_INDEX="2 4"

# ============================================================

COMMON_ARGS=(
    --pi0_mode local
    --pi0_config_path "${PI0_CONFIG_PATH}"
    --pi0_checkpoint_dir "${PI0_CHECKPOINT_DIR}"
    --instruction "${INSTRUCTION}"
    --robot_ports ${ROBOT_PORTS}
    --camera_names ${CAMERA_NAMES}
    --camera_index ${CAMERA_INDEX}
    --max_timesteps 200
    --control_rate 20
    --query_freq 10
    --num_episodes 10
    --output_dir "./logs/eval_airbot_${MODE}"
)

if [ "$MODE" = "pi0" ]; then
    echo "=== Evaluating base pi0 policy ==="
    python3 examples/eval_airbot.py --mode pi0 "${COMMON_ARGS[@]}"

elif [ "$MODE" = "dsrl" ]; then
    if [ -z "$SAC_CHECKPOINT_DIR" ]; then
        echo "Error: SAC_CHECKPOINT_DIR must be set for dsrl mode"
        exit 1
    fi
    echo "=== Evaluating DSRL (pi0 + SAC) policy ==="
    # SAC architecture is loaded from variant.json saved alongside the
    # checkpoint during training, so no arch flags are needed here.
    python3 examples/eval_airbot.py --mode dsrl \
        --sac_checkpoint_dir "${SAC_CHECKPOINT_DIR}" \
        "${COMMON_ARGS[@]}"
else
    echo "Usage: $0 [pi0|dsrl]"
    exit 1
fi
