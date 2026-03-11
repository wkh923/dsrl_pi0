"""Training utilities for DSRL on Airbot robots.

Based on train_utils_real.py (DROID/Franka), adapted for Airbot hardware
using the VLA-RL robot interface (play_operator / System).
"""
import os
import time
from tqdm import tqdm
import numpy as np
import jax
import sys
import select
import tty
import termios
from openpi_client import image_tools
from moviepy.editor import ImageSequenceClip
import PIL


def obs_to_pi0_input(curr_obs, airbot_config, instruction):
    """Format airbot observation into the dict expected by the airbot pi0 policy.

    The AirbotInputs transform expects keys like:
        observation/base_0_rgb, observation/left_wrist_0_rgb, observation/right_wrist_0_rgb,
        observation/state, prompt
    """
    request_data = {
        "observation/state": curr_obs["qpos"],
        "prompt": instruction,
    }
    # Add camera images with the standard airbot observation keys
    for cam_name in airbot_config['camera_names']:
        if cam_name in curr_obs['images']:
            request_data[f"observation/{cam_name}"] = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(curr_obs['images'][cam_name], 224, 224)
            )
    return request_data


def process_images_for_sac(variant, curr_obs, airbot_config):
    """Resize and concatenate camera images for the SAC agent's pixel observation.

    Returns array of shape (1, H, W, 3*num_cameras, 1).
    """
    img_list = []
    for cam_name in airbot_config['camera_names']:
        if cam_name in curr_obs['images']:
            img = curr_obs['images'][cam_name]
            img = np.array(PIL.Image.fromarray(img).resize(
                (variant.resize_image, variant.resize_image)
            ))
            img_list.append(img)

    if len(img_list) == 0:
        raise ValueError("No camera images found in observation")

    # Concatenate along channel dimension: (H, W, 3*num_cameras)
    img_all = np.concatenate(img_list, axis=2)
    # Add batch and trailing dim: (1, H, W, 3*num_cameras, 1)
    return img_all[np.newaxis, ..., np.newaxis]


def extract_observation(robot, obs_raw, airbot_config):
    """Extract structured observation from raw airbot robot observation.

    Returns dict with 'qpos' (float32 array) and 'images' (dict of uint8 arrays).
    """
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


def trajwise_alternating_training_loop(variant, agent, robot, online_replay_buffer, replay_buffer, wandb_logger,
                                       shard_fn=None, agent_dp=None, airbot_config=None):
    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
    if shard_fn is not None:
        replay_buffer_iterator = map(shard_fn, replay_buffer_iterator)

    i = 0
    total_env_steps = 0
    total_num_traj = 0
    wandb_logger.log({'num_online_samples': 0}, step=i)
    wandb_logger.log({'num_online_trajs': 0}, step=i)
    wandb_logger.log({'env_steps': 0}, step=i)

    with tqdm(total=variant.max_steps, initial=0) as pbar:
        while i <= variant.max_steps:
            traj = collect_traj(variant, agent, robot, i, agent_dp, wandb_logger, total_num_traj, airbot_config)
            total_num_traj += 1
            add_online_data_to_buffer(variant, traj, online_replay_buffer)
            total_env_steps += traj['env_steps']
            print('online buffer timesteps length:', len(online_replay_buffer))
            print('online buffer num traj:', total_num_traj)
            print('total env steps:', total_env_steps)

            if i == 0:
                num_gradsteps = 5000
            else:
                num_gradsteps = len(traj["rewards"]) * variant.multi_grad_step
            print(f'num_gradsteps: {num_gradsteps}')

            if total_num_traj >= variant.num_initial_traj_collect:
                for _ in range(num_gradsteps):
                    batch = next(replay_buffer_iterator)
                    update_info = agent.update(batch)

                    pbar.update()
                    i += 1

                    if i % variant.log_interval == 0:
                        update_info = {k: jax.device_get(v) for k, v in update_info.items()}
                        for k, v in update_info.items():
                            if v.ndim == 0:
                                wandb_logger.log({f'training/{k}': v}, step=i)
                            elif v.ndim <= 2:
                                wandb_logger.log_histogram(f'training/{k}', v, i)
                        wandb_logger.log({
                            'replay_buffer_size': len(online_replay_buffer),
                            'is_success (exploration)': int(traj['is_success']),
                        }, i)

                    if i % variant.eval_interval == 0:
                        wandb_logger.log({'num_online_samples': len(online_replay_buffer)}, step=i)
                        wandb_logger.log({'num_online_trajs': total_num_traj}, step=i)
                        wandb_logger.log({'env_steps': total_env_steps}, step=i)
                        if hasattr(agent, 'perform_eval'):
                            agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, None)

                    if variant.checkpoint_interval != -1:
                        if i % variant.checkpoint_interval == 0:
                            agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)


def add_online_data_to_buffer(variant, traj, online_replay_buffer):
    discount_horizon = variant.query_freq
    actions = np.array(traj['actions'])  # (T, chunk_size, action_dim)
    episode_len = len(actions)
    rewards = np.array(traj['rewards'])
    masks = np.array(traj['masks'])

    for t in range(episode_len):
        obs = traj['observations'][t]
        next_obs = traj['observations'][t + 1]
        # remove batch dimension
        obs = {k: v[0] for k, v in obs.items()}
        next_obs = {k: v[0] for k, v in next_obs.items()}
        if not variant.add_states:
            obs.pop('state', None)
            next_obs.pop('state', None)

        insert_dict = dict(
            observations=obs,
            next_observations=next_obs,
            actions=actions[t],
            next_actions=actions[t + 1] if t < episode_len - 1 else actions[t],
            rewards=rewards[t],
            masks=masks[t],
            discount=variant.discount ** discount_horizon
        )
        online_replay_buffer.insert(insert_dict)
    online_replay_buffer.increment_traj_counter()


def collect_traj(variant, agent, robot, i, agent_dp=None, wandb_logger=None, traj_id=None, airbot_config=None):
    query_frequency = variant.query_freq
    instruction = variant.instruction
    max_timesteps = airbot_config['max_timesteps']
    step_time = 1.0 / airbot_config.get('control_rate', 20)  # default 20 Hz
    action_horizon = airbot_config['action_horizon']

    agent._rng, rng = jax.random.split(agent._rng)

    rewards = []
    action_list = []
    obs_list = []
    image_list = []
    is_success = False

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())

        # Wait for user to start episode
        print("Press Enter to start episode (or 'q' to quit)...")
        while True:
            if select.select([sys.stdin], [], [], 0.1) == ([sys.stdin], [], []):
                char_input = sys.stdin.read(1)
                if char_input.lower() == 'q':
                    print("Quitting...")
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                    return {
                        'observations': [], 'actions': [], 'rewards': np.array([]),
                        'masks': np.array([]), 'is_success': False, 'env_steps': 0,
                    }
                elif char_input in ('\r', '\n'):
                    break

        last_step_time = time.time()

        for t in tqdm(range(max_timesteps)):
            # Check for early stop
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                char_input = sys.stdin.read(1)
                if char_input.lower() == 'q':
                    print("'q' pressed, stopping episode.")
                    break

            # Get observation from robot
            obs_raw = robot.capture_observation()
            curr_obs = extract_observation(robot, obs_raw, airbot_config)
            # Save first camera image for video logging
            first_cam = airbot_config['camera_names'][0]
            if first_cam in curr_obs['images']:
                image_list.append(curr_obs['images'][first_cam])

            request_data = obs_to_pi0_input(curr_obs, airbot_config, instruction)

            if t % query_frequency == 0:
                rng, key = jax.random.split(rng)

                img_all = process_images_for_sac(variant, curr_obs, airbot_config)

                # Extract VLM features from pi0 backbone and concat with qpos as state
                img_rep_pi0, _ = agent_dp.get_prefix_rep(request_data)
                img_rep_pi0 = img_rep_pi0[:, -1, :]  # (1, 2048)
                qpos = np.concatenate([curr_obs["qpos"], img_rep_pi0.flatten()])

                obs_dict = {
                    'pixels': img_all,
                    'state': qpos[np.newaxis, ..., np.newaxis],
                }

                if i == 0:
                    # Initial data collection: sample from standard Gaussian noise
                    noise = jax.random.normal(key, (1, *agent.action_chunk_shape))
                    noise_repeat = jax.numpy.repeat(
                        noise[:, -1:, :], action_horizon - noise.shape[1], axis=1
                    )
                    noise = jax.numpy.concatenate([noise, noise_repeat], axis=1)
                    actions_noise = noise[0, :agent.action_chunk_shape[0], :]
                else:
                    # SAC agent predicts noise for the diffusion model
                    actions_noise = agent.sample_actions(obs_dict)
                    actions_noise = np.reshape(actions_noise, agent.action_chunk_shape)
                    noise = np.repeat(
                        actions_noise[-1:, :], action_horizon - actions_noise.shape[0], axis=0
                    )
                    noise = jax.numpy.concatenate([actions_noise, noise], axis=0)[None]

                action_list.append(actions_noise)
                obs_list.append(obs_dict)
                action = agent_dp.infer(request_data, noise=np.asarray(noise))["actions"]

            action_t = action[t % query_frequency]

            # Clip actions to valid range
            action_t = np.clip(action_t, -1, 1)

            # Binarize gripper action(s)
            state_dim = airbot_config['state_dim']
            num_arms = state_dim // 7
            for arm_idx in range(num_arms):
                gripper_idx = (arm_idx + 1) * 7 - 1
                if gripper_idx < len(action_t):
                    action_t[gripper_idx] = 1.0 if action_t[gripper_idx] > 0.5 else 0.0

            # Send action to robot (only the first state_dim dimensions)
            robot.send_action(action_t[:state_dim])

            # Maintain control rate
            now = time.time()
            dt = now - last_step_time
            if dt < step_time:
                time.sleep(step_time - dt)
                last_step_time = time.time()
            else:
                last_step_time = now

        # Human labels success/failure
        print("Trial finished. Mark as (1) Success or (0) Failure:")
        while True:
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                char_input = sys.stdin.read(1)
                if char_input == '1':
                    print("Trial marked as SUCCESS.")
                    is_success = True
                    break
                elif char_input == '0':
                    print("Trial marked as FAILURE.")
                    is_success = False
                    break
            time.sleep(0.01)

        # Add last observation
        obs_raw = robot.capture_observation()
        curr_obs = extract_observation(robot, obs_raw, airbot_config)
        first_cam = airbot_config['camera_names'][0]
        if first_cam in curr_obs['images']:
            image_list.append(curr_obs['images'][first_cam])
        request_data = obs_to_pi0_input(curr_obs, airbot_config, instruction)
        img_all = process_images_for_sac(variant, curr_obs, airbot_config)
        img_rep_pi0, _ = agent_dp.get_prefix_rep(request_data)
        img_rep_pi0 = img_rep_pi0[:, -1, :]
        qpos = np.concatenate([curr_obs["qpos"], img_rep_pi0.flatten()])
        obs_dict = {
            'pixels': img_all,
            'state': qpos[np.newaxis, ..., np.newaxis],
        }
        obs_list.append(obs_dict)
        print('Rollout Done')

    finally:
        # Assign sparse rewards
        if is_success:
            query_steps = len(action_list)
            rewards = np.concatenate([-np.ones(query_steps - 1), [0]])
            masks = np.concatenate([np.ones(query_steps - 1), [0]])
        else:
            query_steps = len(action_list)
            rewards = -np.ones(query_steps)
            masks = np.ones(query_steps)

        if wandb_logger is not None:
            wandb_logger.log({'is_success': int(is_success)}, step=i)
            wandb_logger.log({'total_num_traj': traj_id}, step=i)

        # Save rollout video
        if len(image_list) > 0:
            video_path = os.path.join(variant.outputdir, f'video_{traj_id}.mp4')
            video = np.stack(image_list)
            ImageSequenceClip(list(video), fps=airbot_config.get('control_rate', 20)).write_videofile(
                video_path, codec="libx264"
            )

        print("Episode Done! Press Enter after resetting the environment...")
        # Wait for user confirmation before continuing
        while True:
            if select.select([sys.stdin], [], [], 0.1) == ([sys.stdin], [], []):
                char_input = sys.stdin.read(1)
                if char_input in ('\r', '\n'):
                    break
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    traj = {
        'observations': obs_list,
        'actions': action_list,
        'rewards': rewards,
        'masks': masks,
        'is_success': is_success,
        'env_steps': t + 1,
    }

    return traj
