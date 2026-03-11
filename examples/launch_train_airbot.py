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
    parser.add_argument('--query_freq', default=10, help='how often to query pi0 (in env steps)', type=int)
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
    parser.add_argument('--camera_index', nargs='+', default=[2, 4], type=int,
                        help='Camera device indices')
    parser.add_argument('--max_timesteps', default=200, help='Max timesteps per episode', type=int)
    parser.add_argument('--control_rate', default=20, help='Robot control rate in Hz', type=int)

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
