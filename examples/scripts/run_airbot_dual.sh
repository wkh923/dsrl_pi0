#!/bin/bash
# DSRL training on Airbot robot (dual-arm / ptk example)

proj_name=DSRL_pi0_Airbot_Dual
device_id=0

export EXP=./logs/$proj_name
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# ============================================================
# Fill in your configuration below
# ============================================================

# Path to your VLA-RL task config.py (ptk dual-arm config)
PI0_CONFIG_PATH=""  # e.g., "./VLA-RL/airbot-pi0/openpi/data/ptk_task/config.py"

# Path to your SFT checkpoint directory
PI0_CHECKPOINT_DIR=""  # e.g., "./VLA-RL/airbot-pi0/openpi/checkpoints/ptk_task/9000"

# Task instruction
INSTRUCTION="pick up the object"

# ============================================================

python3 examples/launch_train_airbot.py \
--algorithm pixel_sac \
--env airbot \
--prefix dsrl_pi0_airbot_dual \
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
--robot_ports 50051 50053 \
--camera_names base_0_rgb left_wrist_0_rgb right_wrist_0_rgb \
--camera_index 2 4 6 \
--max_timesteps 200 \
--control_rate 20 \
--instruction "${INSTRUCTION}"
