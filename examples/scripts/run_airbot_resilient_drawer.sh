#!/bin/bash
# Crash-resilient wrapper for DSRL Airbot training.
#
# Difference from run_airbot_update.sh:
#   * --resume_dir <fixed path, no timestamp>  → 跨重启共享同一个目录；
#                                                python 启动时检测并 load 上次状态续训。
#   * 外层 while-true bash 循环：python 退出码 0/130 时跳出，
#                                其他退出码 sleep 10 秒后重启。
#
# 这样一来，训练中遇到的 motor timeout / OOM / segfault / KeyError
# 都会触发自动重启，不会丢已经收集的 (s, a, r, s') 数据和 SAC 训练状态。
#
# 操作员重启后还是要按 Enter 开始新一条 rollout（保留现状）。
# 完全不想跑了：ctrl+c wrapper bash（python 已死/被杀的情况下也行）。

proj_name=DSRL_pi0_Airbot
device_id=0
seed=42

# Task tag — appears in RESUME_DIR so state stays isolated per task.
# Change it when switching tasks (e.g. pick_eraser_out_of_box → pour_water);
# otherwise the new task will resume from the old task's buffer.
TASK_TAG=open_drawer

export EXP=./logs/$proj_name
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=.:./Airbot

# 持久化 dir（不带 timestamp） — 跨多次重启共享。TASK_TAG 隔离不同任务。
RESUME_DIR="${EXP}/persistent_${TASK_TAG}_warm20_s${seed}"
mkdir -p "$RESUME_DIR"

# ============================================================
# Fill in your configuration below (与 run_airbot_update.sh 保持同步)
# ============================================================


PI0_CONFIG_PATH="/home/jpy/RM/Airbot-VLA-RL/VLA/airbot-pi0/openpi/data/pull_drawer/config.py"
PI0_CHECKPOINT_DIR="/home/jpy/RM/Airbot-VLA-RL/VLA/airbot-pi0/checkpoints/open_drawer/please open the drawer/20000"

ROBOT_PORTS="50051 50053"  # dual-arm
CAMERA_INDEX="243222074218 243522071794 243222071389"
CAMERA_NAMES="base_0_rgb left_wrist_0_rgb right_wrist_0_rgb"
INSTRUCTION="pull the drawer open"
RESET_ACTION="-0.001618 -1.036 0.842 -1.615 0.634 1.695 0.0 -0.001618 -1.036 0.842 -1.615 0.634 1.695 0.0"
NUM_RANDOM_ROLLOUTS=10

# ============================================================

attempt=1
while true; do
    echo ""
    echo "[$(date)] === Attempt $attempt — DSRL training starting ==="
    echo "[$(date)]     RESUME_DIR=${RESUME_DIR}"
    echo ""

    python3 examples/launch_train_airbot.py \
    --algorithm pixel_sac \
    --env airbot \
    --prefix dsrl_pi0_airbot_warm20 \
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
    --max_timesteps 175 \
    --control_rate 20 \
    --reset_action ${RESET_ACTION} \
    --instruction "${INSTRUCTION}" \
    --resume_dir "${RESUME_DIR}"
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
