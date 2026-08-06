#!/bin/bash
# Crash-resilient DSRL Airbot training WITH Robometer dense rewards (fold_clothes).
#
# Difference vs run_airbot_resilient_rm_clothes.sh:
#   * --rm_variant robometer + --robometer_model_path replace --rm_demo_path /
#     --rm_threshold_offset / --rm_repo_path → Robometer-4B scores task progress
#     straight from (rollout frames, INSTRUCTION); no reference demo needed.
#     Reward per query step = -0.5 + clip(progress_delta / (1/num_query_steps),
#     -1, 1), on the same -0.5-centered scale as the RARM variants. Unlike RARM
#     (monotonic high-water mark) the delta keeps its SIGN by default, so undoing
#     progress is penalized down to -1.5 — faithful to the sim robometer branch,
#     which applies no max()/clamp. Set MONOTONIC_FLAG below for RARM's exact
#     [-0.5, +0.5] range. See examples/airbot/robometer_wrapper.py.
#   * RESUME_DIR uses a "_robometer_" suffix so this run lives in its own
#     persistent dir, distinct from the sparse baseline and the RARM run.
#
# NOTE on reward resolution: Robometer's config caps a single forward pass at
# max_frames=8 progress points. fold_clothes has 1000/25 = 40 query steps, so 8
# points are interpolated across 40 steps — the coarsest of the four tasks
# (eraser=8 and drawer=7 are ~1:1). If credit looks too smeared here, the
# alternative is the sim's growing-window get_progress path (one VLM pass per
# query step: finer, ~40x the inference cost).
#
# One-time setup (NOT done by this script):
#   The robometer package is vendored inside Reward-Model-MVP on branch
#   yiduo/drq_v2_libero. Note ~/RM/.../Reward-Model-MVP is NOT its own git repo
#   (it sits inside the Airbot-VLA-RL repo), so fetch via the real clone under
#   ~/RLinf, then mirror the subdir into the RM deployment copy that DSRL uses:
#     RMVP=reward_model_baseline/MetaWorld/baseline/robometer
#     cd /home/jpy/RLinf/Reward-Model-MVP && git fetch origin
#     git restore --source=origin/yiduo/drq_v2_libero --worktree -- "$RMVP"
#     cp -r "$RMVP" /home/jpy/RM/Airbot-VLA-RL/Reward-Model-MVP/"$(dirname $RMVP)"/
#     conda activate dsrl_pi0
#     pip install -e /home/jpy/RM/Airbot-VLA-RL/Reward-Model-MVP/"$RMVP"
#   Robometer-4B weights (8.3G) are already downloaded to ROBOMETER_MODEL_PATH.
#
# Behavior:
#   - Same outer bash while-true restart loop as run_airbot_resilient_clothes.sh.
#   - python exit 0 / 130 (Ctrl+C) → break; else sleep 10 → retry.

proj_name=DSRL_pi0_Airbot
device_id=0
seed=42

# Task tag — appears in RESUME_DIR. Keep stable per task so state persists
# across wrapper restarts; change it when switching tasks.
TASK_TAG=fold_clothes

export EXP=./logs/$proj_name
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# pi0/SAC (JAX) and Robometer-4B (PyTorch) share one 4090. Torch's expandable
# segments cut fragmentation when JAX grows its arena alongside it.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# If you still hit OOM, cap JAX's share so torch keeps room for the 4B weights:
#   export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export PYTHONPATH=.:./Airbot

# Persistent dir (no timestamp; survives multiple python restarts).
RESUME_DIR="${EXP}/persistent_${TASK_TAG}_warm20_robometer_s${seed}"
mkdir -p "$RESUME_DIR"

# ============================================================
# Fill in your configuration below (must match run_airbot_resilient_clothes.sh)
# ============================================================

PI0_CONFIG_PATH="/home/jpy/RM/Airbot-VLA-RL/VLA/airbot-pi0/openpi/data/fold_clothes/config.py"
PI0_CHECKPOINT_DIR="/home/jpy/RM/Airbot-VLA-RL/VLA/airbot-pi0/checkpoints/fold_clothes/fold_clothes/74999"

ROBOT_PORTS="50051 50053"  # dual-arm
CAMERA_INDEX="243222074218 243522071794 243222071389"
CAMERA_NAMES="base_0_rgb left_wrist_0_rgb right_wrist_0_rgb"
INSTRUCTION="please fold the clothes"
RESET_ACTION="-0.001618 -1.036 0.842 -1.615 0.634 1.695 0.0 -0.001618 -1.036 0.842 -1.615 0.634 1.695 0.0"
NUM_RANDOM_ROLLOUTS=10

# ============================================================
# Robometer dense reward configuration
# ============================================================
ROBOMETER_MODEL_PATH="/home/jpy/RM/Airbot-VLA-RL/Reward-Model-MVP/checkpoints/Robometer-4B"
ROBOMETER_SERVER_URL="http://127.0.0.1:8765"   # start robometer_server.py first
RM_CAMERA="base_0_rgb"   # collect_traj captures rm_frames from this camera

# ============================================================

attempt=1
while true; do
    echo ""
    echo "[$(date)] === Attempt $attempt — DSRL+Robometer training starting ==="
    echo "[$(date)]     RESUME_DIR=${RESUME_DIR}"
    echo "[$(date)]     ROBOMETER_MODEL_PATH=${ROBOMETER_MODEL_PATH}"
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
    --max_timesteps 1000 \
    --control_rate 20 \
    --reset_action ${RESET_ACTION} \
    --instruction "${INSTRUCTION}" \
    --resume_dir "${RESUME_DIR}" \
    --use_rm \
    --rm_variant robometer \
    --rm_camera "${RM_CAMERA}" \
    --rm_capture_stride 5 \
    --robometer_model_path "${ROBOMETER_MODEL_PATH}" \
    --robometer_server_url "${ROBOMETER_SERVER_URL}"
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
