#!/bin/bash
# Crash-resilient DSRL Airbot training WITH BinaryProgressRewardModel dense rewards.
#
# Difference vs run_airbot_resilient.sh:
#   * --use_rm + --rm_demo_path + --rm_camera + --rm_threshold_offset + --rm_repo_path
#     → the 8 per-rollout transition rewards are derived from the Reward-Model-MVP
#       BinaryProgressRewardModel (DINOv3 ViT-B16 + cross-attn comparator). 0.0 on
#       hit, -1.0 on miss; user success label still forces reward[-1]=0.
#   * RESUME_DIR includes TASK_TAG + "_rm_" suffix so this run lives in its own
#     persistent dir, distinct from the baseline (run_airbot_resilient.sh).
#
# Demo prep:
#   Drop a successful rollout's frames into
#     /home/jpy/dsrl_pi0/data/rm_demos/${TASK_TAG}/demo_seed0/frame_000.jpg ...
#   Frames must be the base_0_rgb camera output, named frame_000.jpg ...
#   frame_<N-1>.jpg (no trailing frame_200.jpg if rollout is 200 steps).
#   The wrapper will auto-pad to the required length under /tmp.
#
# Behavior:
#   - Same outer bash while-true restart loop as run_airbot_resilient.sh.
#   - python exit 0 / 130 (Ctrl+C) → break; else sleep 10 → retry.
#   - RM dense reward signal replaces sparse user-1/0; user 1/0 still recorded
#     as rollout/user_success W&B metric for monitoring.

proj_name=DSRL_pi0_Airbot
device_id=0
seed=42

# Task tag — appears in RESUME_DIR and RM demo path. Keep stable per task so
# state persists across wrapper restarts; change it when switching tasks.
TASK_TAG=pick_eraser_out_of_box

export EXP=./logs/$proj_name
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=.:./Airbot

# Persistent dir (no timestamp; survives multiple python restarts).
# TASK_TAG keeps state isolated per task; "_rm_" tag keeps RM run separate from
# the baseline (run_airbot_resilient.sh) so you can run both on the same task
# without state collision.
RESUME_DIR="${EXP}/persistent_${TASK_TAG}_warm20_rm_s${seed}"
mkdir -p "$RESUME_DIR"

# ============================================================
# Fill in your configuration below (must match run_airbot_resilient.sh)
# ============================================================

PI0_CONFIG_PATH="/home/jpy/RM/Airbot-VLA-RL/VLA/airbot-pi0/openpi/data/${TASK_TAG}/config.py"
PI0_CHECKPOINT_DIR="/home/jpy/RM/Airbot-VLA-RL/VLA/airbot-pi0/checkpoints/${TASK_TAG}/pick_eraser/30000"
ROBOT_PORTS="50051 50053"  # dual-arm
CAMERA_INDEX="243222074218 243522071794 243222071389"
CAMERA_NAMES="base_0_rgb left_wrist_0_rgb right_wrist_0_rgb"
INSTRUCTION="pick eraser out of box"
RESET_ACTION="-0.001618 -1.036 0.842 -1.615 0.634 1.695 0.0 -0.001618 -1.036 0.842 -1.615 0.634 1.695 0.0"
NUM_RANDOM_ROLLOUTS=10

# ============================================================
# Reward-Model-MVP dense reward configuration
# ============================================================
RM_DEMO_PATH="/home/jpy/dsrl_pi0/data/rm_demos/${TASK_TAG}/demo_seed0"
RM_CAMERA="base_0_rgb"
RM_THRESHOLD_OFFSET=0.3   # per_clip_threshold = max_self_sim - this; SMALLER offset
                          # → HIGHER threshold → STRICTER match (rollout clip needs
                          # higher similarity to count as match/hit). 0.3: e.g. a
                          # demo clip with max_self_sim 0.8 needs rollout sim ≥ 0.5.
RM_REPO_PATH="/home/jpy/RM/Airbot-VLA-RL/Reward-Model-MVP"

# ============================================================

attempt=1
while true; do
    echo ""
    echo "[$(date)] === Attempt $attempt — DSRL+RM training starting ==="
    echo "[$(date)]     RESUME_DIR=${RESUME_DIR}"
    echo "[$(date)]     RM_DEMO_PATH=${RM_DEMO_PATH}"
    echo ""

    python3 examples/launch_train_airbot.py \
    --algorithm pixel_sac \
    --env airbot \
    --prefix dsrl_pi0_airbot_warm20_rm \
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
    --max_timesteps 200 \
    --control_rate 20 \
    --reset_action ${RESET_ACTION} \
    --instruction "${INSTRUCTION}" \
    --resume_dir "${RESUME_DIR}" \
    --use_rm \
    --rm_demo_path "${RM_DEMO_PATH}" \
    --rm_camera "${RM_CAMERA}" \
    --rm_threshold_offset ${RM_THRESHOLD_OFFSET} \
    --rm_repo_path "${RM_REPO_PATH}" \
    --rm_capture_stride 5 \
    --rm_variant eraser
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
