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
                                       shard_fn=None, agent_dp=None, airbot_config=None,
                                       initial_i=0, initial_total_num_traj=0, initial_total_env_steps=0,
                                       rm=None):
    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
    if shard_fn is not None:
        replay_buffer_iterator = map(shard_fn, replay_buffer_iterator)

    # Counters resume from the values loaded by main() (all 0 for a fresh run).
    i = initial_i
    total_env_steps = initial_total_env_steps
    total_num_traj = initial_total_num_traj
    wandb_logger.log({'num_online_samples': len(online_replay_buffer)}, step=i)
    wandb_logger.log({'num_online_trajs': total_num_traj}, step=i)
    wandb_logger.log({'env_steps': total_env_steps}, step=i)

    with tqdm(total=variant.max_steps, initial=i) as pbar:
        while i <= variant.max_steps:
            traj = collect_traj(variant, agent, robot, i, agent_dp, wandb_logger, total_num_traj, airbot_config, rm=rm)
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

            # Warmup (5000 grad steps on the bootstrap data) is triggered at the END
            # of the random-noise phase, i.e. immediately after rollout
            # `num_random_rollouts`. Default num_random_rollouts=1 reproduces the
            # original DSRL behavior (warmup right after rollout 1). Setting it
            # to e.g. 20 means the buffer accumulates ~160 transitions across 20
            # random-noise rollouts before warmup, then SAC takes over.
            if total_num_traj == variant.num_random_rollouts:
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

            # === Crash-resilient state save (only when --resume_dir is in use) ===
            # Save replay buffer + counters + agent RNG + SAC ckpt at the end of
            # every rollout's SAC update phase. On crash + auto-restart by
            # run_airbot_resilient.sh, training resumes from the last saved state.
            # Saves are atomic (tmp + os.replace) so a partial write doesn't
            # corrupt the previous good state.
            if getattr(variant, 'resume_dir', ''):
                from examples.airbot.state_persist import save_buffer, save_meta
                save_buffer(variant.outputdir, online_replay_buffer)
                save_meta(
                    variant.outputdir,
                    i=i,
                    total_num_traj=total_num_traj,
                    total_env_steps=total_env_steps,
                    agent_rng=agent._rng,
                )
                # Also force a SAC checkpoint every rollout (default cadence is
                # every checkpoint_interval=5000 SAC steps ≈ 21 rollouts; that
                # would lose ~21 rollouts of SAC training on crash). One extra
                # ckpt write per rollout costs ~5 MB / ~1 second.
                agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)

            # === Milestone SAC checkpoint every 5 rollouts ===
            # Saved under outputdir/milestones/ with keep_every_n_steps=1 so
            # EVERY milestone survives flax pruning (the rolling resume ckpt
            # above prunes to multiples of checkpoint_interval). Dirs are named
            # by rollout count — milestones/checkpoint5, checkpoint10, ... —
            # giving a permanent, predictable policy-snapshot history for eval.
            # Does not affect resume: latest_checkpoint() reads outputdir/, not
            # outputdir/milestones/.
            if total_num_traj % 5 == 0:
                milestones_dir = os.path.join(variant.outputdir, 'milestones')
                agent.save_checkpoint(milestones_dir, total_num_traj, keep_every_n_steps=1)


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


def collect_traj(variant, agent, robot, i, agent_dp=None, wandb_logger=None, traj_id=None, airbot_config=None, rm=None):
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
    # Transient per-episode frame buffer for RM dense rewards. One frame per
    # rm_capture_stride env-steps from the RM camera; rm_frames[i] corresponds
    # to env_step i*rm_capture_stride. Discarded after RM scoring — never
    # enters the replay buffer. Only collected when --use_rm is set.
    rm_frames = []
    rm_camera = getattr(variant, 'rm_camera', None) if rm is not None else None
    rm_capture_stride = int(getattr(rm, 'capture_stride', 1)) if rm is not None else 1
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
            # Save first camera image for video logging. RealSense default
            # profile delivers BGR8 but downstream code treats as RGB; swap
            # channels so saved videos look correct AND so RM consumes true
            # RGB (matches DINOv3's training distribution). pi0's input
            # pipeline still reads curr_obs['images'] unmodified, so SFT
            # inference behavior is unchanged.
            first_cam = airbot_config['camera_names'][0]
            if first_cam in curr_obs['images']:
                image_list.append(curr_obs['images'][first_cam][..., ::-1].copy())
            # RM frame collection at env_steps that are multiples of
            # rm_capture_stride (e.g. every 5 env-steps for sparse mode).
            # rm_frames[len(rm_frames)] will correspond to env_step
            # len(rm_frames)*rm_capture_stride at the moment of append.
            # Transient — discarded after RM scoring; not in replay buffer.
            # Same BGR->RGB swap as image_list above, kept consistent with the
            # demo saver in eval_airbot_test.py so demo and rollout land in
            # the same color space when scored by RM.
            if (rm is not None and rm_camera is not None
                    and rm_camera in curr_obs['images']
                    and t % rm_capture_stride == 0):
                rm_frames.append(curr_obs['images'][rm_camera][..., ::-1].copy())

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

                if traj_id < variant.num_random_rollouts:
                    # Random-noise phase: sample standard Gaussian (like the
                    # original i==0 branch). With num_random_rollouts=20, the
                    # first 20 rollouts all collect data using random noise so
                    # SAC has a richer buffer (~160 transitions) to learn from
                    # before its own noise output starts steering pi0.
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

        # Add last observation. Same BGR->RGB swap as the in-loop append
        # (line ~304) so the final video frame matches the rest — RealSense
        # default profile delivers BGR8.
        obs_raw = robot.capture_observation()
        curr_obs = extract_observation(robot, obs_raw, airbot_config)
        first_cam = airbot_config['camera_names'][0]
        if first_cam in curr_obs['images']:
            image_list.append(curr_obs['images'][first_cam][..., ::-1].copy())
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
        # Assign sparse rewards (default / RM-disabled / RM-failure fallback)
        query_steps = len(action_list)
        if is_success:
            rewards = np.concatenate([-np.ones(query_steps - 1), [0]])
            masks = np.concatenate([np.ones(query_steps - 1), [0]])
        else:
            rewards = -np.ones(query_steps)
            masks = np.ones(query_steps)

        rm_used = False
        if rm is not None and query_steps > 0 and len(rm_frames) > 0:
            try:
                rm_rewards = rm.compute_rewards(
                    rm_frames, num_query_steps=query_steps, traj_id=traj_id)
                rewards = rm_rewards.astype(np.float32)
                # User-label override on the FINAL query step only:
                #   pressed "1" (success) → rewards[-1] += 1.0  (completion bonus
                #       ON TOP of the halved RM step reward → final ∈ [0.5, 1.5])
                #   pressed "0" (failure) → keep RM-computed value (the halved
                #       step reward, ∈ [-0.5, +0.5])
                # Non-final steps are never overridden. Per-clip RM rewards are
                # in the halved range [-0.5, +0.5]; episode sum ∈ [-4.0, +5.0].
                if is_success:
                    rewards[-1] = float(rewards[-1]) + 1.0
                rm_used = True
                rewards_pretty = [round(float(r), 2) for r in rewards.tolist()]
                print(f"[RM] rewards={rewards_pretty} sum={float(rewards.sum()):.2f} "
                      f"final={float(rewards[-1]):+.2f} is_success={int(is_success)}")
            except Exception as e:
                print(f"[RM] compute_rewards failed: {e}; falling back to sparse reward")

        if wandb_logger is not None:
            wandb_logger.log({'is_success': int(is_success)}, step=i)
            wandb_logger.log({'total_num_traj': traj_id}, step=i)
            wandb_logger.log({'rollout/user_success': float(is_success)}, step=i)
            if rm_used:
                # Number of clips that advanced progress (= "hit" status):
                # halved per-clip rewards are -0.5 for match/miss and > -0.5 for
                # hits, so count rewards strictly above -0.5. The final clip's
                # +1.0 success bonus also clears this, so it's "advance OR
                # final-step success" — a useful proxy for RM hit rate.
                hit_count = int((rewards > -0.5 + 1e-6).sum())
                wandb_logger.log({
                    'rollout/rm_hit_count': hit_count,
                    'rollout/rm_mean_reward': float(rewards.mean()),
                    'rollout/rm_sum_reward': float(rewards.sum()),
                    'rollout/rm_final_reward': float(rewards[-1]),
                    'rollout/rm_max_reached_progress': float(
                        getattr(rm, 'last_max_reached_progress', 0.0)),
                }, step=i)

        # Save rollout frames — one every 5 env-steps, same format as the
        # reference demo (frame_000000.jpg, frame_000005.jpg, ...). image_list
        # is captured every env-step so image_list[i] is env_step i; saving
        # image_list[::5] yields env_steps 0, 5, 10, ... A good rollout's
        # folder can be used directly as an RM reference demo.
        if len(image_list) > 0:
            from PIL import Image
            frames_dir = os.path.join(variant.outputdir, f'rollout_{traj_id}')
            os.makedirs(frames_dir, exist_ok=True)
            for envstep in range(0, len(image_list), 5):
                arr = np.asarray(image_list[envstep])
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                Image.fromarray(arr, mode='RGB').save(
                    os.path.join(frames_dir, f'frame_{envstep:06d}.jpg'), quality=95)

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
