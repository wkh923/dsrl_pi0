#!/bin/bash
# Crash-resilient DSRL Airbot training WITH Robometer dense rewards.
#
# Difference vs run_airbot_resilient_rm_handover.sh:
#   * --rm_variant robometer + --robometer_repo_path + --robometer_model_path
#     replace --rm_demo_path / --rm_camera / --rm_threshold_offset / --rm_repo_path
#     → Robometer-4B scores task progress straight from (rollout video,
#     INSTRUCTION), no reference demo clip needed. See
#     examples/airbot/robometer_wrapper.py for the reward contract (per-query-step
#     progress-delta, clipped to [-0.5, 0.5]; +1.0 success override on the final
#     step still applied by train_utils_airbot.collect_traj).
#   * RESUME_DIR uses a "_robometer_" suffix so this run lives in its own
#     persistent dir, distinct from both the sparse baseline and the RM-MVP run.
#
# One-time setup (NOT done by this script):
#   git clone https://github.com/robometer/robometer /home/jpy/RM/Airbot-VLA-RL/robometer
#   pip install -e /home/jpy/RM/Airbot-VLA-RL/robometer
#   Check free disk space first (`df -h /home/jpy`) — Robometer-4B weights are
#   several GB and download to HF_HOME on first run.
#
# Behavior:
#   - Same outer bash while-true restart loop as run_airbot_resilient_handover.sh.
#   - python exit 0 / 130 (Ctrl+C) → break; else sleep 10 → retry.

proj_name=DSRL_pi0_Airbot
device_id=0
seed=42

# Task tag — appears in RESUME_DIR. Keep stable per task so state persists
# across wrapper restarts; change it when switching tasks.
TASK_TAG=handover

export EXP=./logs/$proj_name
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=.:./Airbot

# Persistent dir (no timestamp; survives multiple python restarts).
RESUME_DIR="${EXP}/persistent_${TASK_TAG}_warm20_robometer_s${seed}"
mkdir -p "$RESUME_DIR"

# ============================================================
# Fill in your configuration below (keep in sync with run_airbot_resilient_handover.sh)
# ============================================================

PI0_CONFIG_PATH="/home/jpy/RM/Airbot-VLA-RL/VLA/airbot-pi0/openpi/data/handover/config.py"
PI0_CHECKPOINT_DIR="/home/jpy/RM/Airbot-VLA-RL/VLA/airbot-pi0/checkpoints/handover/74999"

ROBOT_PORTS="50051 50053"  # dual-arm
CAMERA_INDEX="243222074218 243522071794 243222071389"
CAMERA_NAMES="base_0_rgb left_wrist_0_rgb right_wrist_0_rgb"
INSTRUCTION="please hand over the object"
RESET_ACTION="-0.001618 -1.036 0.842 -1.615 0.634 1.695 0.0 -0.001618 -1.036 0.842 -1.615 0.634 1.695 0.0"
NUM_RANDOM_ROLLOUTS=10

# ============================================================
# Robometer dense reward configuration
# ============================================================
ROBOMETER_REPO_PATH="/home/jpy/RM/Airbot-VLA-RL/robometer"
ROBOMETER_MODEL_PATH="robometer/Robometer-4B"

# ============================================================

attempt=1
while true; do
    echo ""
    echo "[$(date)] === Attempt $attempt — DSRL+Robometer training starting ==="
    echo "[$(date)]     RESUME_DIR=${RESUME_DIR}"
    echo ""

    python3 examples/launch_train_airbot.py \
    --algorithm pixel_sac \
    --env airbot \
    --prefix dsrl_pi0_airbot_warm20_robometer \
    --wandb_project ${proj_name} \
    --batch_size 256 \
    --discount 0.99 \
    --seed $seed \
    --max_steps 500000 \
    --eval_interval 2000 \
    --log_interval 100 \
    --checkpoint_interval 5000 \
    --multi_grad_step 30 \
    --resize_image 128 \
    --action_magnitude 1.5 \
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
    --max_timesteps 300 \
    --control_rate 20 \
    --reset_action ${RESET_ACTION} \
    --reset_release_grippers \
    --instruction "${INSTRUCTION}" \
    --resume_dir "${RESUME_DIR}" \
    --use_rm \
    --rm_variant robometer \
    --rm_capture_stride 5 \
    --robometer_repo_path "${ROBOMETER_REPO_PATH}" \
    --robometer_model_path "${ROBOMETER_MODEL_PATH}"
    code=$?

    if [ $code -eq 0 ]; then
        echo "[$(date)] === clean exit (max_steps reached or all done) ==="
        break
    fi
    if [ $code -eq 130 ]; then
        echo "[$(date)] === user interrupt (Ctrl+C) — stopping ==="
        break
    fi

    echo ""
    echo "[$(date)] === Python crashed with exit code $code ==="
    echo "[$(date)]     Sleeping 10 seconds, then retrying (attempt $((attempt+1)))"
    echo ""
    sleep 10
    attempt=$((attempt+1))
done
