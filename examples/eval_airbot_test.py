"""Evaluation script for playing pi0 (with or without DSRL) on Airbot robots.

Supports two modes:
  --mode pi0       : Run the base pi0 SFT policy (no SAC noise steering)
  --mode dsrl      : Run pi0 steered by a trained SAC checkpoint

Usage:
  python3 examples/eval_airbot.py --mode pi0 --pi0_mode local \
      --pi0_config_path /path/to/config.py --pi0_checkpoint_dir /path/to/ckpt \
      --robot_ports 50051 --camera_names base_0_rgb left_wrist_0_rgb --camera_index 2 4

  python3 examples/eval_airbot.py --mode dsrl --sac_checkpoint_dir /path/to/sac_ckpt \
      --pi0_mode local --pi0_config_path /path/to/config.py --pi0_checkpoint_dir /path/to/ckpt \
      --robot_ports 50051 --camera_names base_0_rgb left_wrist_0_rgb --camera_index 2 4
"""
import argparse
import json
import os
import sys
import time
import select
import tty
import termios
import logging

import jax
import numpy as np
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip
import tensorflow as tf
from jax.experimental.compilation_cache import compilation_cache

from openpi_client import image_tools

home_dir = os.environ['HOME']
compilation_cache.initialize_cache(os.path.join(home_dir, 'jax_compilation_cache'))


def obs_to_pi0_input(curr_obs, airbot_config, instruction):
    request_data = {
        "observation/state": curr_obs["qpos"],
        "prompt": instruction,
    }
    for cam_name in airbot_config['camera_names']:
        if cam_name in curr_obs['images']:
            request_data[f"observation/{cam_name}"] = image_tools.resize_with_pad(
                curr_obs['images'][cam_name], 224, 224
            )
    return request_data


def process_images_for_sac(resize_image, curr_obs, airbot_config):
    img_list = []
    for cam_name in airbot_config['camera_names']:
        if cam_name in curr_obs['images']:
            img = curr_obs['images'][cam_name]
            img = image_tools.resize_with_pad(img, resize_image, resize_image)
            img_list.append(img)
    if len(img_list) == 0:
        raise ValueError("No camera images found in observation")
    img_all = np.concatenate(img_list, axis=2)
    return img_all[np.newaxis, ..., np.newaxis]


def extract_observation(robot, obs_raw, airbot_config):
    qpos = np.array(robot.get_qpos(obs_raw), dtype=np.float32)
    images = {}
    for cam_name in airbot_config['camera_names']:
        img_data = obs_raw.get(f"{cam_name}/color/image_raw")
        if img_data is not None:
            img = img_data["data"]
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            images[cam_name] = img
    return {"qpos": qpos, "images": images}


def run_episode(args, robot, agent_dp, airbot_config, agent=None, rng=None, episode_id=0):
    """Run a single episode.

    Args:
        agent: SAC agent (None for pi0-only mode)
        rng: JAX RNG key
    Returns:
        dict with is_success, env_steps, image_list
    """
    query_frequency = args.query_freq
    instruction = args.instruction
    max_timesteps = airbot_config['max_timesteps']
    step_time = 1.0 / airbot_config.get('control_rate', 20)
    action_horizon = airbot_config['action_horizon']
    mode = args.mode

    image_list = []
    is_success = False

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())

        print(f"\n--- Episode {episode_id + 1} ({mode} mode) ---")
        print("Press Enter to start (or 'q' to quit)...")
        while True:
            if select.select([sys.stdin], [], [], 0.1) == ([sys.stdin], [], []):
                char_input = sys.stdin.read(1)
                if char_input.lower() == 'q':
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                    return None  # signal to quit
                elif char_input in ('\r', '\n'):
                    break

        last_step_time = time.time()

        for t in tqdm(range(max_timesteps), desc=f"Episode {episode_id + 1}"):
            # Check for early stop
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                char_input = sys.stdin.read(1)
                if char_input.lower() == 'q':
                    print("Early stop.")
                    break

            t0 = time.time()
            t_cam = 0.0
            t_infer = 0.0

            # Read cameras every 5 steps (for video) + always at query steps (for inference)
            if t % 5 == 0 or t % query_frequency == 0:
                obs_raw = robot.capture_observation()
                t1 = time.time()
                t_cam = t1 - t0
                curr_obs = extract_observation(robot, obs_raw, airbot_config)

                first_cam = airbot_config['camera_names'][0]
                if first_cam in curr_obs['images']:
                    image_list.append(curr_obs['images'][first_cam])

            # Inference only at query steps
            if t % query_frequency == 0:
                request_data = obs_to_pi0_input(curr_obs, airbot_config, instruction)
                t2 = time.time()
                if mode == 'pi0':
                    # DIAGNOSTIC TEST VARIANT — feed pi0 the SAME repeated-noise
                    # structure that DSRL train (i=0) and DSRL eval use, instead
                    # of letting pi0 sample its own (50, 32) independent noise.
                    # If this still breaks the robot the way `run_airbot.sh` does,
                    # we've isolated the bug to the repeated-noise OOD input —
                    # nothing else changes between this and `eval_airbot.py pi0`.
                    rng, key = jax.random.split(rng)
                    seed = jax.random.normal(key, (1, 1, 32))                 # one (1, 32) Gaussian
                    repeat = jax.numpy.repeat(seed[:, -1:, :], action_horizon - 1, axis=1)
                    noise = jax.numpy.concatenate([seed, repeat], axis=1)     # (1, 50, 32) all identical
                    action = agent_dp.infer(request_data, noise=np.asarray(noise))["actions"]
                elif mode == 'dsrl':
                    rng, key = jax.random.split(rng)

                    img_all = process_images_for_sac(args.resize_image, curr_obs, airbot_config)
                    img_rep_pi0, _ = agent_dp.get_prefix_rep(request_data)
                    img_rep_pi0 = img_rep_pi0[:, -1, :]
                    qpos = np.concatenate([curr_obs["qpos"], img_rep_pi0.flatten()])

                    obs_dict = {
                        'pixels': img_all,
                        'state': qpos[np.newaxis, ..., np.newaxis],
                    }

                    actions_noise = agent.sample_actions(obs_dict)
                    actions_noise = np.reshape(actions_noise, agent.action_chunk_shape)
                    noise = np.repeat(
                        actions_noise[-1:, :], action_horizon - actions_noise.shape[0], axis=0
                    )
                    noise = jax.numpy.concatenate([actions_noise, noise], axis=0)[None]
                    action = agent_dp.infer(request_data, noise=np.asarray(noise))["actions"]
                t_infer = time.time() - t2

            t3 = time.time()
            print(f"[t={t:3d}] cam:{t_cam*1000:.0f}ms  infer:{t_infer*1000:.0f}ms  total:{(t3-t0)*1000:.0f}ms")

            action_t = action[t % query_frequency]

            if mode == 'dsrl':
                action_t = np.clip(action_t, -1, 1)
                state_dim = airbot_config['state_dim']
                num_arms = state_dim // 7
                for arm_idx in range(num_arms):
                    gripper_idx = (arm_idx + 1) * 7 - 1
                    if gripper_idx < len(action_t):
                        action_t[gripper_idx] = 1.0 if action_t[gripper_idx] > 0.5 else 0.0
                robot.send_action(action_t[:state_dim])
            else:
                robot.send_action(action_t)

            now = time.time()
            dt = now - last_step_time
            if dt < step_time:
                time.sleep(step_time - dt)
                last_step_time = time.time()
            else:
                last_step_time = now

        # Human labels success/failure
        print("Mark as (1) Success or (0) Failure:")
        while True:
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                char_input = sys.stdin.read(1)
                if char_input == '1':
                    print("SUCCESS")
                    is_success = True
                    break
                elif char_input == '0':
                    print("FAILURE")
                    is_success = False
                    break
            time.sleep(0.01)

        print("Press Enter after resetting the environment...")
        while True:
            if select.select([sys.stdin], [], [], 0.1) == ([sys.stdin], [], []):
                char_input = sys.stdin.read(1)
                if char_input in ('\r', '\n'):
                    break

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    return {
        'is_success': is_success,
        'env_steps': t + 1,
        'image_list': image_list,
    }


def main(args):
    tf.config.set_visible_devices([], "GPU")

    # ---- Load pi0 policy ----
    if args.pi0_mode == 'local':
        from openpi.policies.policy_config import create_trained_policy
        from examples.airbot.airbot_data_config import get_task_config, get_config
        task_config = get_task_config(args.pi0_config_path)
        config = get_config(task_config)
        agent_dp = create_trained_policy(config, args.pi0_checkpoint_dir)
        print(f"Loaded pi0 locally from {args.pi0_checkpoint_dir}")
        if not args.instruction:
            args.instruction = task_config.task_name
    elif args.pi0_mode == 'remote':
        from openpi_client import websocket_client_policy as _wcp
        agent_dp = _wcp.WebsocketClientPolicy(
            host=os.environ['PI0_REMOTE_HOST'],
            port=int(os.environ['PI0_REMOTE_PORT'])
        )
        logging.info("Connected to remote pi0 server")
    else:
        raise ValueError(f"Unknown pi0_mode: {args.pi0_mode}")

    action_horizon = getattr(agent_dp, 'action_horizon', 50)

    # ---- Set up robot ----
    from examples.airbot.play_operator import Robot
    from examples.airbot.robot_config import RobotConfig
    from airbot_data_collection.basis import SystemMode

    robot_config = RobotConfig(
        robot_type=args.robot_type,
        robot_ports=[int(p) for p in args.robot_ports],
        camera_names=args.camera_names,
        camera_index=args.camera_index,
    )
    if args.robot_groups:
        robot_config.robot_groups = args.robot_groups

    robot = Robot(robot_config)
    robot.switch_mode(SystemMode.SAMPLING)
    print(f"Initialized Airbot robot: type={args.robot_type}, ports={args.robot_ports}")

    num_arms = len(args.robot_ports)
    state_dim = 7 * num_arms

    airbot_config = {
        'camera_names': args.camera_names,
        'state_dim': state_dim,
        'max_timesteps': args.max_timesteps,
        'control_rate': args.control_rate,
        'action_horizon': action_horizon,
    }

    # ---- Load SAC agent (dsrl mode only) ----
    agent = None
    if args.mode == 'dsrl':
        if not args.sac_checkpoint_dir:
            raise ValueError("--sac_checkpoint_dir is required for dsrl mode")

        from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
        from jaxrl2.utils.general_utils import add_batch_dim
        from gym.spaces import Dict, Box

        # Load the training variant so the SAC architecture matches exactly.
        # Prefer an explicit --training_variant_path; otherwise look for
        # variant.json in the parent of the checkpoint directory.
        variant_path = args.training_variant_path or os.path.join(
            os.path.dirname(os.path.abspath(args.sac_checkpoint_dir)), 'variant.json'
        )
        if not os.path.isfile(variant_path):
            raise FileNotFoundError(
                f"variant.json not found at {variant_path}. Pass "
                "--training_variant_path to point at the training output dir."
            )
        with open(variant_path) as f:
            train_variant = json.load(f)
        train_kwargs = train_variant['train_kwargs']
        print(f"Loaded training variant from {variant_path}")

        # Reconcile CLI overrides with the training variant.
        resize_image = train_variant.get('resize_image', args.resize_image)
        add_states = int(train_variant.get('add_states', args.add_states))

        num_cameras = len(args.camera_names)
        image_shape = (resize_image, resize_image, 3 * num_cameras, 1)
        obs_dict = {'pixels': Box(low=0, high=255, shape=image_shape, dtype=np.uint8)}
        if add_states:
            obs_dict['state'] = Box(low=-np.inf, high=np.inf, shape=(state_dim + 2048, 1), dtype=np.float32)
        observation_space = Dict(obs_dict)
        action_space = Box(low=-1, high=1, shape=(1, 32,), dtype=np.float32)

        sample_obs = add_batch_dim(observation_space.sample())
        sample_action = add_batch_dim(action_space.sample())

        # Ensure num_cameras stays in sync with the actual cameras at eval time.
        train_kwargs = dict(train_kwargs)
        train_kwargs.pop('decay_steps', None)
        train_kwargs.pop('cosine_decay', None)
        train_kwargs['num_cameras'] = num_cameras

        agent = PixelSACLearner(args.seed, sample_obs, sample_action, **train_kwargs)
        agent.restore_checkpoint(args.sac_checkpoint_dir)
        # Propagate the actual resize_image downstream so run_episode reads it.
        args.resize_image = resize_image
        print(f"Loaded SAC checkpoint from {args.sac_checkpoint_dir}")

    # ---- Run evaluation episodes ----
    rng = jax.random.PRNGKey(args.seed)
    output_dir = args.output_dir
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    successes = []
    for ep in range(args.num_episodes):
        if args.reset_action:
            print(f"Resetting arm to specified pose...")
            robot.reset_to_pose(args.reset_action)
            print("Reset done.")
        rng, ep_rng = jax.random.split(rng)
        result = run_episode(args, robot, agent_dp, airbot_config, agent=agent, rng=ep_rng, episode_id=ep)

        if result is None:
            print("Quit requested. Stopping evaluation.")
            break

        successes.append(result['is_success'])
        print(f"Episode {ep + 1}: {'SUCCESS' if result['is_success'] else 'FAILURE'} "
              f"({result['env_steps']} steps)")

        # Save video
        if output_dir and len(result['image_list']) > 0:
            video_path = os.path.join(output_dir, f'eval_{args.mode}_ep{ep}.mp4')
            video = np.stack(result['image_list'])
            ImageSequenceClip(list(video), fps=args.control_rate).write_videofile(
                video_path, codec="libx264", logger=None
            )

    # Summary
    if successes:
        rate = sum(successes) / len(successes)
        print(f"\n{'='*40}")
        print(f"Mode: {args.mode}")
        print(f"Episodes: {len(successes)}")
        print(f"Success rate: {rate:.1%} ({sum(successes)}/{len(successes)})")
        print(f"{'='*40}")

    robot.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate pi0 / DSRL policy on Airbot')

    # Evaluation mode
    parser.add_argument('--mode', required=True, choices=['pi0', 'dsrl'],
                        help='pi0: base policy only; dsrl: pi0 + trained SAC noise steering')
    parser.add_argument('--num_episodes', default=10, type=int, help='Number of evaluation episodes')
    parser.add_argument('--output_dir', default='./logs/eval_airbot', help='Directory to save videos')
    parser.add_argument('--seed', default=42, type=int)

    # Pi0 policy
    parser.add_argument('--pi0_mode', default='local', choices=['local', 'remote'])
    parser.add_argument('--pi0_config_path', default='', help='Path to airbot task config.py')
    parser.add_argument('--pi0_checkpoint_dir', default='', help='Path to SFT checkpoint dir')
    parser.add_argument('--instruction', default='', help='Language instruction for the task')

    # SAC checkpoint (dsrl mode only)
    parser.add_argument('--sac_checkpoint_dir', default='', help='Path to trained SAC checkpoint')
    parser.add_argument('--training_variant_path', default='',
                        help='Path to variant.json saved during training. '
                             'If omitted, looks for it in the parent of --sac_checkpoint_dir.')

    # SAC fallback defaults (only used when variant.json cannot be found).
    parser.add_argument('--resize_image', default=128, type=int)
    parser.add_argument('--add_states', default=1, type=int)
    parser.add_argument('--query_freq', default=25, type=int)

    # Robot config
    parser.add_argument('--robot_type', default='play')
    parser.add_argument('--robot_ports', nargs='+', default=[50051], type=int)
    parser.add_argument('--robot_groups', nargs='+', default=None)
    parser.add_argument('--camera_names', nargs='+', default=['base_0_rgb', 'left_wrist_0_rgb'])
    parser.add_argument('--camera_index', nargs='+', default=['243322074422', '243522071794'])
    parser.add_argument('--max_timesteps', default=200, type=int)
    parser.add_argument('--control_rate', default=20, type=int)
    parser.add_argument('--reset_action', nargs='+', type=float, default=None,
                        help='Joint positions to reset the arm to before each episode (e.g. 7 values for single-arm)')

    args = parser.parse_args()
    main(args)
