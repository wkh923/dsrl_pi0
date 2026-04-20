#!/bin/bash
# DSRL training on Airbot robot (single-arm example)
#
# Prerequisites:
# 1. SFT checkpoint from VLA-RL (e.g., checkpoints/put_cup/9000)
# 2. Task config.py from VLA-RL (e.g., data/put_cup/config.py)
# 3. airbot_py SDK installed
# 4. airbot_data_collection package installed (from VLA-RL/airbot-data/data-collection)
# 5. dsrl_pi0's openpi installed (pip install -e openpi)

proj_name=DSRL_pi0_Airbot
device_id=0

export EXP=./logs/$proj_name
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# ============================================================
# Fill in your configuration below
# ============================================================

# Path to your VLA-RL task config.py (defines TASK_NAME, CAMERA_TOPICS, DELTA_ACTION_MASK, etc.)
PI0_CONFIG_PATH=""  # e.g., "./VLA-RL/airbot-pi0/openpi/data/put_cup/config.py"

# Path to your SFT checkpoint directory
PI0_CHECKPOINT_DIR=""  # e.g., "./VLA-RL/airbot-pi0/openpi/checkpoints/put_cup/9000"

# Robot ports (single-arm: one port, dual-arm: two ports)
ROBOT_PORTS="50051"  # For dual-arm: "50051 50053"

# Camera device indices (matching your physical setup)
CAMERA_INDEX="2 4"

# Camera names (must match what's defined in your config.py's CAMERA_TOPICS)
CAMERA_NAMES="base_0_rgb left_wrist_0_rgb"

# Task instruction
INSTRUCTION="put the cup on the plate"

# ============================================================

python3 examples/launch_train_airbot.py \
--algorithm pixel_sac \
--env airbot \
--prefix dsrl_pi0_airbot \
--wandb_project ${proj_name} \
--batch_size 256 \
--discount 0.99 \
--seed 0 \
--max_steps 500000 \
--eval_interval 2000 \
--log_interval 100 \
--checkpoint_interval 5000 \
--multi_grad_step 30 \
--resize_image 128 \
--action_magnitude 2.5 \
--query_freq 10 \
--hidden_dims 1024 \
--num_qs 2 \
--pi0_mode local \
--pi0_config_path "${PI0_CONFIG_PATH}" \
--pi0_checkpoint_dir "${PI0_CHECKPOINT_DIR}" \
--robot_type play \
--robot_ports ${ROBOT_PORTS} \
--camera_names ${CAMERA_NAMES} \
--camera_index ${CAMERA_INDEX} \
--max_timesteps 200 \
--control_rate 20 \
--instruction "${INSTRUCTION}"
