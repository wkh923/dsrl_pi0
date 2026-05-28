"""Main training script for DSRL on Airbot robots.

Based on train_real.py (DROID/Franka), adapted for Airbot hardware.
Supports both local and remote pi0 policy loading.
"""
import json
import os
import logging

import jax
from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
from jaxrl2.utils.general_utils import add_batch_dim
import numpy as np

import gymnasium as gym
from gym.spaces import Dict, Box

from jaxrl2.data import ReplayBuffer
from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name
import tempfile
from functools import partial
from examples.train_utils_airbot import trajwise_alternating_training_loop
import tensorflow as tf
from jax.experimental.compilation_cache import compilation_cache

home_dir = os.environ['HOME']
compilation_cache.initialize_cache(os.path.join(home_dir, 'jax_compilation_cache'))


def shard_batch(batch, sharding):
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(
            x, sharding.reshape(sharding.shape[0], *((1,) * (x.ndim - 1)))
        ),
        batch,
    )


class DummyEnv(gym.ObservationWrapper):
    """Dummy environment to define observation/action spaces for SAC agent initialization."""

    def __init__(self, variant, airbot_config):
        self.variant = variant
        num_cameras = len(airbot_config['camera_names'])
        self.image_shape = (variant.resize_image, variant.resize_image, 3 * num_cameras, 1)
        obs_dict = {}
        obs_dict['pixels'] = Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)
        if variant.add_states:
            # state_dim = robot proprioceptive dims + VLM feature dims (2048)
            state_dim = airbot_config['state_dim'] + 2048
            obs_dict['state'] = Box(low=-np.inf, high=np.inf, shape=(state_dim, 1), dtype=np.float32)
        self.observation_space = Dict(obs_dict)
        # SAC action space: (1, 32) noise vectors — same as other DSRL setups
        # 32 is pi0's internal action dimension
        self.action_space = Box(low=-1, high=1, shape=(1, 32,), dtype=np.float32)


def main(variant):
    devices = jax.local_devices()
    num_devices = len(devices)
    assert variant.batch_size % num_devices == 0
    logging.info(f'num devices: {num_devices}')
    logging.info(f'batch size: {variant.batch_size}')
    sharding = jax.sharding.PositionalSharding(devices)
    shard_fn = partial(shard_batch, sharding=sharding)

    # Prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    kwargs = variant['train_kwargs']
    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = variant.max_steps

    # Override num_cameras from actual camera_names to keep them in sync
    kwargs['num_cameras'] = len(variant.camera_names)

    if not variant.prefix:
        import uuid
        variant.prefix = str(uuid.uuid4().fields[-1])[:5]

    if variant.suffix:
        expname = create_exp_name(variant.prefix, seed=variant.seed) + f"_{variant.suffix}"
    else:
        expname = create_exp_name(variant.prefix, seed=variant.seed)

    # outputdir logic depends on whether --resume_dir is set:
    #   * --resume_dir non-empty: use it as outputdir (no timestamp suffix).
    #     If state files exist there, we'll auto-load buffer + counters + RNG +
    #     latest SAC ckpt below. Used by run_airbot_resilient.sh for
    #     crash-resilient training across multiple python restarts.
    #   * --resume_dir empty: original behavior — fresh timestamped dir.
    # orbax checkpoint backend (used by save_checkpoint) requires an absolute path.
    from examples.airbot.state_persist import has_state, load_meta, load_buffer

    if variant.resume_dir:
        outputdir = os.path.abspath(variant.resume_dir)
        os.makedirs(outputdir, exist_ok=True)
        is_resuming = has_state(outputdir)
        if is_resuming:
            print(f'Resuming training from existing state at {outputdir}')
        else:
            print(f'No prior state at {outputdir}, starting fresh in persistent dir')
    else:
        outputdir = os.path.abspath(os.path.join(os.environ.get('EXP', './logs/dsrl_airbot'), expname))
        os.makedirs(outputdir, exist_ok=True)
        is_resuming = False
    variant.outputdir = outputdir
    print('writing to output dir', outputdir)

    # Persist the training variant so eval_airbot.py can reconstruct the SAC
    # architecture without re-specifying every hyperparameter on the CLI.
    with open(os.path.join(outputdir, 'variant.json'), 'w') as f:
        json.dump(dict(variant), f, indent=2, default=str)

    group_name = variant.prefix + '_' + variant.launch_group_id
    wandb_output_dir = tempfile.mkdtemp()
    wandb_logger = WandBLogger(
        variant.prefix != '', variant, variant.wandb_project,
        experiment_id=expname, output_dir=wandb_output_dir, group_name=group_name
    )

    # ---- Load pi0 policy ----
    if variant.pi0_mode == 'local':
        # Local loading: use airbot config + checkpoint
        from openpi.policies.policy_config import create_trained_policy
        from examples.airbot.airbot_data_config import get_task_config, get_config
        task_config = get_task_config(variant.pi0_config_path)
        config = get_config(task_config)
        agent_dp = create_trained_policy(config, variant.pi0_checkpoint_dir)
        print(f"Loaded pi0 locally from {variant.pi0_checkpoint_dir}")
        if not variant.instruction:
            variant.instruction = task_config.task_name
    elif variant.pi0_mode == 'remote':
        # Remote loading: connect to websocket server
        from openpi_client import websocket_client_policy as _websocket_client_policy
        agent_dp = _websocket_client_policy.WebsocketClientPolicy(
            host=os.environ['PI0_REMOTE_HOST'],
            port=int(os.environ['PI0_REMOTE_PORT'])
        )
        logging.info(f"Connected to remote pi0 server")
    else:
        raise ValueError(f"Unknown pi0_mode: {variant.pi0_mode}. Use 'local' or 'remote'.")

    # Get action_horizon from the policy (for local) or use pi0 default (for remote)
    action_horizon = getattr(agent_dp, 'action_horizon', 50)

    # ---- Set up Airbot robot ----
    from examples.airbot.play_operator import Robot
    from examples.airbot.robot_config import RobotConfig
    from airbot_data_collection.basis import SystemMode

    robot_config_obj = RobotConfig(
        robot_type=variant.robot_type,
        robot_ports=[int(p) for p in variant.robot_ports],
        camera_names=variant.camera_names,
        camera_index=variant.camera_index,
    )
    if variant.robot_groups:
        robot_config_obj.robot_groups = variant.robot_groups

    robot = Robot(robot_config_obj)
    # Move arms into servo-control mode. Without this, send_action either no-ops
    # or KeyErrors inside AIRBOTPlay._comp_act[...].
    robot.switch_mode(SystemMode.SAMPLING)
    print(f"Initialized Airbot robot: type={variant.robot_type}, ports={variant.robot_ports}")

    # Airbot config for training utilities
    # state_dim: 7 per arm (6 joints + 1 gripper)
    num_arms = len(variant.robot_ports)
    state_dim = 7 * num_arms

    airbot_config = {
        'camera_names': variant.camera_names,
        'state_dim': state_dim,
        'max_timesteps': variant.max_timesteps,
        'control_rate': variant.control_rate,
        'action_horizon': action_horizon,
    }

    # ---- Initialize SAC agent ----
    dummy_env = DummyEnv(variant, airbot_config)
    sample_obs = add_batch_dim(dummy_env.observation_space.sample())
    sample_action = add_batch_dim(dummy_env.action_space.sample())
    print('sample obs shapes', [(k, v.shape) for k, v in sample_obs.items()])
    print('sample action shape', sample_action.shape)

    agent = PixelSACLearner(variant.seed, sample_obs, sample_action, **kwargs)

    if variant.restore_path:
        print(f'restoring from {variant.restore_path}')
        agent.restore_checkpoint(variant.restore_path)

    # ---- Create replay buffer ----
    online_buffer_size = 2 * variant.max_steps // variant.multi_grad_step
    online_replay_buffer = ReplayBuffer(dummy_env.observation_space, dummy_env.action_space, int(online_buffer_size))
    replay_buffer = online_replay_buffer
    replay_buffer.seed(variant.seed)

    # ---- Resume training state (if --resume_dir is set and state exists) ----
    initial_i = 0
    initial_total_num_traj = 0
    initial_total_env_steps = 0
    if is_resuming:
        # 1. Load buffer (in-place into the empty buffer just created above)
        load_buffer(outputdir, online_replay_buffer)

        # 2. Load metadata (counters + agent JAX RNG)
        meta = load_meta(outputdir)
        initial_i = meta['i']
        initial_total_num_traj = meta['total_num_traj']
        initial_total_env_steps = meta['total_env_steps']
        agent._rng = meta['agent_rng']

        # 3. Restore SAC checkpoint (uses flax.training.checkpoints under the hood)
        from flax.training.checkpoints import latest_checkpoint
        latest = latest_checkpoint(outputdir, prefix='checkpoint')
        if latest is not None:
            agent.restore_checkpoint(latest)
            print(f'Restored SAC checkpoint: {latest}')
        else:
            print('Warning: state files exist but no SAC checkpoint found; '
                  'SAC params remain freshly-initialized.')

        print(f'Resumed: i={initial_i}, total_num_traj={initial_total_num_traj}, '
              f'total_env_steps={initial_total_env_steps}, '
              f'buffer_size={len(online_replay_buffer)}')

    # ---- Initialize Reward Model (RM) for dense rewards (optional) ----
    rm = None
    if getattr(variant, 'use_rm', False):
        if not variant.rm_demo_path:
            raise ValueError(
                "--use_rm is set but --rm_demo_path is empty. Provide a folder "
                "of frame_*.jpg from one successful rollout.")
        if variant.rm_camera not in variant.camera_names:
            raise ValueError(
                f"--rm_camera={variant.rm_camera!r} not in --camera_names={variant.camera_names}")
        rm_variant = getattr(variant, 'rm_variant', 'progress')
        if rm_variant == 'eraser':
            from examples.airbot.rm_wrapper_eraser import EraserRewardModel as RMClass
        elif rm_variant == 'clothes':
            from examples.airbot.rm_wrapper_clothes import ClothesRewardModel as RMClass
        elif rm_variant == 'drawer':
            from examples.airbot.rm_wrapper_drawer import DrawerRewardModel as RMClass
        elif rm_variant == 'handover':
            from examples.airbot.rm_wrapper_handover import HandoverRewardModel as RMClass
        else:
            from examples.airbot.rm_wrapper import AirbotRewardModel as RMClass
        print(f"[RM] using reward variant: {rm_variant} ({RMClass.__name__})")
        rm = RMClass(
            demo_path=variant.rm_demo_path,
            rm_repo_path=variant.rm_repo_path,
            max_timesteps=variant.max_timesteps,
            query_freq=variant.query_freq,
            num_query_steps=variant.max_timesteps // variant.query_freq,
            threshold_offset=variant.rm_threshold_offset,
            capture_stride=getattr(variant, 'rm_capture_stride', 1),
        )

    # ---- Start DSRL training ----
    try:
        trajwise_alternating_training_loop(
            variant, agent, robot, online_replay_buffer, replay_buffer, wandb_logger,
            shard_fn=shard_fn, agent_dp=agent_dp, airbot_config=airbot_config,
            initial_i=initial_i,
            initial_total_num_traj=initial_total_num_traj,
            initial_total_env_steps=initial_total_env_steps,
            rm=rm,
        )
    finally:
        # Release gRPC / RealSense handles even on Ctrl+C or exception.
        try:
            robot.shutdown()
        except Exception as e:
            logging.warning(f"robot.shutdown() failed: {e}")
        if rm is not None:
            try:
                rm.close()
            except Exception as e:
                logging.warning(f"rm.close() failed: {e}")
