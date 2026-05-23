"""CLI launcher for DSRL training on Airbot robots."""
import argparse
import sys
from examples.train_airbot import main
from jaxrl2.utils.launch_util import parse_training_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DSRL training on Airbot robots')

    # General training args
    parser.add_argument('--seed', default=42, help='Random seed.', type=int)
    parser.add_argument('--launch_group_id', default='', help='group id used to group runs on wandb.')
    parser.add_argument('--env', default='airbot', help='name of environment')
    parser.add_argument('--log_interval', default=100, help='Logging interval.', type=int)
    parser.add_argument('--eval_interval', default=2000, help='Eval interval.', type=int)
    parser.add_argument('--checkpoint_interval', default=5000, help='checkpoint interval.', type=int)
    parser.add_argument('--batch_size', default=256, help='Mini batch size.', type=int)
    parser.add_argument('--max_steps', default=500000, help='Number of training steps.', type=int)
    parser.add_argument('--add_states', default=1, help='whether to add low-dim states to the observations', type=int)
    parser.add_argument('--wandb_project', default='DSRL_pi0_Airbot', help='wandb project')
    parser.add_argument('--num_initial_traj_collect', default=1, help='number of trajectories to collect before starting online updates', type=int)
    parser.add_argument('--algorithm', default='pixel_sac', help='type of algorithm')
    parser.add_argument('--prefix', default='', help='prefix to use for wandb')
    parser.add_argument('--suffix', default='', help='suffix to use for wandb')
    parser.add_argument('--multi_grad_step', default=30, help='Number of gradient steps per env step (UTD)', type=int)
    parser.add_argument('--resize_image', default=128, help='the size of image for SAC agent', type=int)
    parser.add_argument('--query_freq', default=25, help='how often to query pi0 (in env steps)', type=int)
    parser.add_argument('--num_random_rollouts', default=1, type=int,
                        help='Number of initial rollouts that use random Gaussian noise (instead of SAC output) '
                             'as the steering noise for pi0. SAC is still trained during these rollouts. '
                             'The 5000-step warmup happens at the END of the random-noise phase '
                             '(i.e. after rollout num_random_rollouts). Default 1 = original DSRL behavior.')
    parser.add_argument('--instruction', default='', help='language instruction for the task')
    parser.add_argument('--restore_path', default='', help='path to restore SAC checkpoint from')

    # Pi0 policy loading
    parser.add_argument('--pi0_mode', default='local', choices=['local', 'remote'],
                        help='How to load pi0: local (from checkpoint) or remote (websocket)')
    parser.add_argument('--pi0_config_path', default='', help='Path to airbot task config.py (for local mode)')
    parser.add_argument('--pi0_checkpoint_dir', default='', help='Path to SFT checkpoint dir (for local mode)')

    # Airbot robot configuration
    parser.add_argument('--robot_type', default='play', help='Airbot robot type')
    parser.add_argument('--robot_ports', nargs='+', default=[50051], type=int,
                        help='gRPC ports for robot arms')
    parser.add_argument('--robot_groups', nargs='+', default=None,
                        help='Robot group names (e.g., left right)')
    parser.add_argument('--camera_names', nargs='+',
                        default=['base_0_rgb', 'left_wrist_0_rgb'],
                        help='Camera names matching airbot config')
    parser.add_argument('--camera_index', nargs='+', default=['243322074422', '243522071794'],
                        help='Camera serial numbers or device indices')
    parser.add_argument('--max_timesteps', default=200, help='Max timesteps per episode', type=int)
    parser.add_argument('--control_rate', default=20, help='Robot control rate in Hz', type=int)
    parser.add_argument('--resume_dir', default='', type=str,
                        help='If non-empty, use this directory as outputdir AND auto-load '
                             'training state (replay buffer + counters + agent RNG + '
                             'latest SAC checkpoint) from it. Empty = original behavior '
                             '(new timestamped exp dir, no resume / no state save). Used '
                             'by run_airbot_resilient.sh for crash-resilient training.')
    parser.add_argument('--reset_action', nargs='+', default=[], type=float,
                        help='Joint pose to auto-reset to between episodes. '
                             '7 floats per arm: [j1..j6, gripper]. '
                             'Single-arm: 7 floats; dual-arm: 14 floats. '
                             'Empty = skip auto-reset (user resets manually).')
    parser.add_argument('--reset_wait_time', default=3.0, type=float,
                        help='Seconds to wait after sending reset_action (planning mode is blocking but server-side completion may lag).')

    # Reward Model (RM) — BinaryProgressRewardModel dense rewards
    parser.add_argument('--use_rm', action='store_true',
                        help='Enable dense rewards from BinaryProgressRewardModel '
                             '(Reward-Model-MVP). When set, the 8 per-rollout '
                             'transition rewards are derived from RM hits '
                             '(0.0 hit / -1.0 miss). If --use_rm is not set, '
                             'the original sparse user-label reward is kept.')
    parser.add_argument('--rm_demo_path', default='', type=str,
                        help='Folder containing one successful rollout as '
                             'frame_000.jpg, frame_001.jpg, ... (or .png). '
                             'Required when --use_rm is set.')
    parser.add_argument('--rm_camera', default='base_0_rgb', type=str,
                        help='Which camera to use as RM input (must be one of '
                             '--camera_names).')
    parser.add_argument('--rm_threshold_offset', default=0.5, type=float,
                        help='per_clip_threshold = max_self_sim - threshold_offset. '
                             '0.5 is the RM library default. Increase to make hits '
                             'rarer (stricter), decrease to make hits more common.')
    parser.add_argument('--rm_repo_path',
                        default='/home/jpy/RM/Airbot-VLA-RL/Reward-Model-MVP',
                        type=str,
                        help='Path to the Reward-Model-MVP repo root. The '
                             '<repo>/reward_model_baseline/MetaWorld dir is '
                             'added to sys.path so RewardModels.* is importable.')
    parser.add_argument('--rm_capture_stride', default=1, type=int,
                        help='Env-step interval at which RM rollout frames are '
                             'captured AND at which demo frames are stored. '
                             '1 = dense (every env-step); 5 = sparse (every 5 '
                             'env-steps, matching the saver in eval_airbot_test.py). '
                             'Must evenly divide frame_stride (10), '
                             'demo_clip_stride (5), and query_freq.')
    parser.add_argument('--rm_variant', default='progress',
                        choices=['progress', 'eraser', 'clothes'],
                        help="Which RM reward scheme to use. 'progress' = the "
                             'general continuous progress-delta reward '
                             '(rm_wrapper.AirbotRewardModel). \'eraser\' = the '
                             'milestone-based reward hard-coded for the '
                             'pick_eraser_out_of_box task '
                             '(rm_wrapper_eraser.EraserRewardModel). \'clothes\' '
                             '= progress reward + a 90%-regime milestone reward '
                             'for fold_clothes (rm_wrapper_clothes.ClothesRewardModel).')

    # SAC hyperparameters tuned for real robot (following train_real.py / run_real.sh)
    train_args_dict = dict(
        actor_lr=1e-4,
        critic_lr=3e-4,
        temp_lr=3e-4,
        hidden_dims=(1024, 1024, 1024),
        cnn_features=(32, 32, 32, 32),
        cnn_strides=(3, 2, 2, 2),
        cnn_padding='VALID',
        latent_dim=50,
        discount=0.99,
        tau=0.005,
        critic_reduction='min',
        dropout_rate=0.0,
        aug_next=1,
        use_bottleneck=True,
        encoder_type='small',
        encoder_norm='group',
        use_spatial_softmax=True,
        softmax_temperature=-1,
        target_entropy=0.0,
        num_qs=2,
        action_magnitude=2.5,
        num_cameras=2,
    )

    variant, args = parse_training_args(train_args_dict, parser)
    print(variant)
    main(variant)
    sys.exit()
