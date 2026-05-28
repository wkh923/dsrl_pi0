#!/bin/bash
# Crash-resilient DSRL Airbot training WITH HandoverRewardModel dense rewards.
#
# Difference vs run_airbot_resilient_handover.sh:
#   * --use_rm + --rm_demo_path + --rm_camera + --rm_threshold_offset + --rm_repo_path
#     + --rm_variant handover  → progress reward + success-gated late regime
#     (0.70 * max_timesteps = env-step 210 latch). On is_success=1, the last
#     ~4 query steps earn boosted regime rewards (hit +0.5 / match +0.25 /
#     miss 0); the final step always uses progress reward + train_utils'
#     +1.0 success override.
#   * RESUME_DIR includes TASK_TAG + "_rm_" suffix so this run lives in its own
#     persistent dir, distinct from the baseline (run_airbot_resilient_handover.sh).
#
# Demo prep:
#   Drop a successful rollout's frames into
#     /home/jpy/dsrl_pi0/data/rm_demos/handover/demo_seed0/frame_000000.jpg ...
#   The wrapper will auto-pad to the required length under /tmp.
#
# Behavior:
#   - Same outer bash while-true restart loop as run_airbot_resilient_handover.sh.
#   - python exit 0 / 130 (Ctrl+C) → break; else sleep 10 → retry.
#   - RM dense reward signal replaces sparse user-1/0; user 1/0 still recorded
#     as rollout/user_success W&B metric for monitoring.

proj_name=DSRL_pi0_Airbot
device_id=0
seed=42

# Task tag — appears in RESUME_DIR and RM demo path. Keep stable per task so
# state persists across wrapper restarts; change it when switching tasks.
TASK_TAG=handover

export EXP=./logs/$proj_name
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=.:./Airbot

# Persistent dir (no timestamp; survives multiple python restarts).
# TASK_TAG keeps state isolated per task; "_rm_" tag keeps RM run separate from
# the baseline (run_airbot_resilient_handover.sh) so you can run both on the
# same task without state collision.
RESUME_DIR="${EXP}/persistent_${TASK_TAG}_warm20_rm_s${seed}"
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
    --max_timesteps 300 \
    --control_rate 20 \
    --reset_action ${RESET_ACTION} \
    --reset_release_grippers \
    --instruction "${INSTRUCTION}" \
    --resume_dir "${RESUME_DIR}" \
    --use_rm \
    --rm_demo_path "${RM_DEMO_PATH}" \
    --rm_camera "${RM_CAMERA}" \
    --rm_threshold_offset ${RM_THRESHOLD_OFFSET} \
    --rm_repo_path "${RM_REPO_PATH}" \
    --rm_capture_stride 5 \
    --rm_variant handover
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
