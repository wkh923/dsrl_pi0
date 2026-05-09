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
            request_data[f"observation/{cam_name}"] = image_tools.resize_with_pad(
                curr_obs['images'][cam_name], 224, 224
            )
    return request_data


def process_images_for_sac(variant, curr_obs, airbot_config):
    """Resize and concatenate camera images for the SAC agent's pixel observation.

    Uses aspect-preserving letterbox resize to match the pi0 input pipeline.
    Returns array of shape (1, H, W, 3*num_cameras, 1).
    """
    img_list = []
    for cam_name in airbot_config['camera_names']:
        if cam_name in curr_obs['images']:
            img = curr_obs['images'][cam_name]
            img = image_tools.resize_with_pad(
                img, variant.resize_image, variant.resize_image
            )
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
            if traj.get('aborted'):
                # User quit at the start prompt — skip without polluting the buffer.
                print("Trajectory aborted; skipping update.")
                continue
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
                        # NOTE: agent.perform_eval -> make_value_reward_visulization -> make_visual
                        # asserts images.shape[-1] == 3 (single-camera RGB). With multi-camera
                        # setups (Airbot dual-arm has 3 cams concat'd to 9 channels) the assert
                        # fires and the whole training loop crashes. Pure visualization — no
                        # effect on SAC weights / replay buffer / checkpoint / rollout. Disabled
                        # until make_visual is patched to handle multi-camera obs.
                        # if hasattr(agent, 'perform_eval'):
                        #     agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, None)

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
    t = -1

    # Auto-reset both arms to a known starting pose before each episode, so
    # the operator only has to put workbench objects back, not drive the arms.
    # Empty list (default unless --reset_action is passed) skips the reset.
    reset_action = getattr(variant, 'reset_action', None) or []
    if reset_action:
        wait_time = getattr(variant, 'reset_wait_time', 3.0)
        print(f"Auto-resetting arms to home pose (wait {wait_time}s)...")
        robot.reset_to_pose(reset_action, wait_time=wait_time)
        print("Reset complete.")

    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    # Wait for user to start episode — handled outside the rollout try/finally
    # so that pressing 'q' here returns immediately instead of running the
    # post-episode cleanup (video save + reset prompt).
    try:
        print("Reset workbench as needed, then press Enter to start episode (or 'q' to quit)...")
        while True:
            if select.select([sys.stdin], [], [], 0.1) == ([sys.stdin], [], []):
                char_input = sys.stdin.read(1)
                if char_input.lower() == 'q':
                    print("Quitting (no trajectory collected).")
                    return {'aborted': True, 'env_steps': 0}
                elif char_input in ('\r', '\n'):
                    break
    except BaseException:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        raise

    try:
        last_step_time = time.time()

        for t in tqdm(range(max_timesteps)):
            t_step_start = time.time()  # DEBUG TIMING
            # Check for early stop
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                char_input = sys.stdin.read(1)
                if char_input.lower() == 'q':
                    print("'q' pressed, stopping episode.")
                    break

            # Get observation from robot
            _t0 = time.time()
            obs_raw = robot.capture_observation()
            t_cam = time.time() - _t0
            curr_obs = extract_observation(robot, obs_raw, airbot_config)
            # Save first camera image for video logging
            first_cam = airbot_config['camera_names'][0]
            if first_cam in curr_obs['images']:
                image_list.append(curr_obs['images'][first_cam])

            _t0 = time.time()
            request_data = obs_to_pi0_input(curr_obs, airbot_config, instruction)
            t_pi0in = time.time() - _t0

            # Per-segment timers (set 0 on non-query steps)
            t_img_sac = t_prefix = t_sac = t_infer = 0.0

            if t % query_frequency == 0:
                rng, key = jax.random.split(rng)

                _t0 = time.time()
                img_all = process_images_for_sac(variant, curr_obs, airbot_config)
                t_img_sac = time.time() - _t0

                # Extract VLM features from pi0 backbone and concat with qpos as state
                _t0 = time.time()
                img_rep_pi0, _ = agent_dp.get_prefix_rep(request_data)
                img_rep_pi0 = img_rep_pi0[:, -1, :]  # (1, 2048)
                img_rep_pi0.block_until_ready()      # force JAX sync for accurate timing
                t_prefix = time.time() - _t0
                qpos = np.concatenate([curr_obs["qpos"], np.asarray(img_rep_pi0).flatten()])

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
                    _t0 = time.time()
                    actions_noise = agent.sample_actions(obs_dict)
                    _na = np.asarray(actions_noise)   # blocks JAX → measures real wall time
                    t_sac = time.time() - _t0
                    # --- DEBUG: SAC noise stats — verify distribution sanity ---
                    print(f"  SAC noise: shape={_na.shape} mean={float(_na.mean()):+.3f} "
                          f"std={float(_na.std()):.3f} min={float(_na.min()):+.3f} max={float(_na.max()):+.3f}")
                    # --- end DEBUG ---
                    actions_noise = np.reshape(actions_noise, agent.action_chunk_shape)
                    noise = np.repeat(
                        actions_noise[-1:, :], action_horizon - actions_noise.shape[0], axis=0
                    )
                    noise = jax.numpy.concatenate([actions_noise, noise], axis=0)[None]

                action_list.append(actions_noise)
                obs_list.append(obs_dict)
                _t0 = time.time()
                action = agent_dp.infer(request_data, noise=np.asarray(noise))["actions"]
                action = np.asarray(action)           # blocks JAX → measures real wall time
                t_infer = time.time() - _t0

                # --- DEBUG: inspect pi0 output ---
                # action shape: (action_horizon, action_dim) = (50, 32). Only first
                # state_dim values are real; rest is pi0's 32-dim padding.
                _state_dim = airbot_config['state_dim']
                _first = action[0, :_state_dim]
                _all_real = action[:, :_state_dim]
                print(f"\n[pi0 out @ traj_i={i}, t={t}]")
                print(f"  shape={action.shape}  state_dim={_state_dim}")
                print(f"  qpos    (current)        = {curr_obs['qpos']}")
                print(f"  action[0,:state_dim]     = {_first}")
                print(f"  chunk min/max/mean       = {_all_real.min():.4f} / "
                      f"{_all_real.max():.4f} / {_all_real.mean():.4f}")
                print(f"  delta_to_first (action-qpos) = {_first - np.asarray(curr_obs['qpos'])}")
                # --- end DEBUG ---

            action_t = action[t % query_frequency]

            # Send action to robot. pi0 outputs absolute joint angles (radians)
            # via AbsoluteActions output transform — clipping to [-1,1] would
            # mangle them (e.g. j4=-1.615 rad would become -1.0). Match
            # eval_airbot.py's pi0-mode path: send raw, slice to state_dim.
            state_dim = airbot_config['state_dim']
            _t0 = time.time()
            robot.send_action(action_t[:state_dim])
            t_send = time.time() - _t0

            # --- DEBUG TIMING: full per-step breakdown at query, slow warning otherwise ---
            t_step = time.time() - t_step_start
            if t % query_frequency == 0:
                print(f"  TIMING t={t}: cam={t_cam*1000:.0f} pi0in={t_pi0in*1000:.0f} "
                      f"img_sac={t_img_sac*1000:.0f} prefix={t_prefix*1000:.0f} "
                      f"sac={t_sac*1000:.0f} infer={t_infer*1000:.0f} send={t_send*1000:.0f} "
                      f"TOTAL={t_step*1000:.0f}ms (target {int(step_time*1000)}ms)")
            elif t_step > 0.080:
                print(f"  SLOW t={t}: cam={t_cam*1000:.0f} pi0in={t_pi0in*1000:.0f} "
                      f"send={t_send*1000:.0f} TOTAL={t_step*1000:.0f}ms")
            # --- end DEBUG TIMING ---

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
