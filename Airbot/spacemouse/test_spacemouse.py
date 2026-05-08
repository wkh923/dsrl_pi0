#!/usr/bin/env python3
"""
Test script for SpaceMouse functionality.

Usage (run from repo root):
    # Single SpaceMouse
    python Airbot/spacemouse/test_spacemouse.py

    # Dual SpaceMouse (bimanual control)
    python Airbot/spacemouse/test_spacemouse.py --dual

    # Dual + auto-calibration
    python Airbot/spacemouse/test_spacemouse.py --dual --auto-calibrate

    # Mock (no hardware)
    python Airbot/spacemouse/test_spacemouse.py --mock

This script tests:
1. SpaceMouse connection
2. Toggle-based intervention (left button)
3. Gripper toggle (right button)
4. 6DOF motion reading
5. Dual SpaceMouse for bimanual control (--dual flag)
"""

import argparse
import sys
import time
from pathlib import Path

# Add Airbot/ to sys.path so `from spacemouse import ...` finds the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spacemouse import (
    AirbotSpaceMouse,
    SpaceMouseState,
    create_spacemouse,
    DualAirbotSpaceMouse,
    create_dual_spacemouse,
)


def test_single_spacemouse(use_mock: bool = False):
    """Test single SpaceMouse (original behavior)."""
    print("=" * 60)
    print("Single SpaceMouse Test (num_arms=1)")
    print("=" * 60)
    print()
    print("Controls:")
    print("  - Left button: Toggle intervention mode")
    print("  - Right button: Toggle gripper (while intervening)")
    print("  - Move SpaceMouse to see 6DOF readings")
    print()
    print("Press Ctrl+C to exit")
    print("=" * 60)

    # Create SpaceMouse
    spacemouse = create_spacemouse(use_mock=use_mock)

    print(f"\nSpaceMouse type: {type(spacemouse).__name__}")
    print()

    try:
        iteration = 0
        while True:
            # Get action and state
            action, state, state_changed = spacemouse.get_action()

            # Print state changes
            if state_changed:
                print(f"\n{'='*40}")
                print(f"STATE CHANGED: {state.name}")
                print(f"{'='*40}\n")

            # Print status every 10 iterations
            if iteration % 10 == 0:
                pos = action[:3]
                rot = action[3:6]
                gripper = action[6]

                print(
                    f"\rState: {state.name:12s} | "
                    f"Pos: [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}] | "
                    f"Rot: [{rot[0]:+.3f}, {rot[1]:+.3f}, {rot[2]:+.3f}] | "
                    f"Gripper: {gripper:.2f}",
                    end="",
                    flush=True,
                )

            iteration += 1
            time.sleep(0.01)  # 100Hz update

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    finally:
        spacemouse.close()
        print("SpaceMouse closed")


def test_dual_spacemouse(use_mock: bool = False, auto_calibrate: bool = False):
    """Test dual SpaceMouse for bimanual control (num_arms=2)."""
    print("=" * 70)
    print("Dual SpaceMouse Test (num_arms=2)")
    print("=" * 70)
    print()
    print("Controls for EACH SpaceMouse:")
    print("  - Left button: Toggle intervention for THIS arm only")
    print("  - Right button: Toggle gripper for THIS arm only (while intervening)")
    print()
    print("Expected behavior:")
    print("  - Left SpaceMouse left button -> only LEFT arm enters/exits intervention")
    print("  - Right SpaceMouse left button -> only RIGHT arm enters/exits intervention")
    print("  - Each arm can be controlled independently")
    print()
    print("Action output: 14D [left_7D, right_7D]")
    print()
    print("Press Ctrl+C to exit")
    print("=" * 70)

    # Create dual SpaceMouse
    spacemouse = create_dual_spacemouse(
        num_arms=2,
        use_mock=use_mock,
        auto_calibrate=auto_calibrate,
    )

    print(f"\nDualSpaceMouse type: {type(spacemouse).__name__}")
    print(f"Number of arms: {spacemouse.num_arms}")
    print()

    try:
        iteration = 0
        while True:
            # Get action and states
            action, arm_states, any_state_changed = spacemouse.get_action()

            # Print state changes
            if any_state_changed:
                print(f"\n{'='*50}")
                states_str = ", ".join(
                    f"{DualAirbotSpaceMouse.ARM_NAMES[i]}={arm_states[i].name}"
                    for i in range(len(arm_states))
                )
                print(f"STATE CHANGED: {states_str}")
                print(f"{'='*50}\n")

            # Print status every 10 iterations
            if iteration % 10 == 0:
                # Left arm (indices 0-6)
                left_pos = action[:3]
                left_rot = action[3:6]
                left_gripper = action[6]
                left_state = arm_states[0].name[:4]  # Truncate for display

                # Right arm (indices 7-13)
                right_pos = action[7:10]
                right_rot = action[10:13]
                right_gripper = action[13]
                right_state = arm_states[1].name[:4]

                # Check which arms are intervening
                left_int = "[INT]" if arm_states[0] == SpaceMouseState.INTERVENING else "[AUT]"
                right_int = "[INT]" if arm_states[1] == SpaceMouseState.INTERVENING else "[AUT]"

                print(
                    f"\r"
                    f"LEFT {left_int} Pos:[{left_pos[0]:+.2f},{left_pos[1]:+.2f},{left_pos[2]:+.2f}] "
                    f"Grip:{left_gripper:.1f} | "
                    f"RIGHT {right_int} Pos:[{right_pos[0]:+.2f},{right_pos[1]:+.2f},{right_pos[2]:+.2f}] "
                    f"Grip:{right_gripper:.1f}",
                    end="",
                    flush=True,
                )

            iteration += 1
            time.sleep(0.01)  # 100Hz update

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    finally:
        spacemouse.close()
        print("DualSpaceMouse closed")


def test_intervention_wrapper(use_mock: bool = False, num_arms: int = 2):
    """Test SpacemouseInterventionWrapper with mock environment."""
    print("=" * 70)
    print(f"SpacemouseInterventionWrapper Test (num_arms={num_arms})")
    print("=" * 70)
    print()

    from airbot_rl.envs import MockAirbotEnv, AirbotEnvConfig
    from airbot_rl.envs.wrappers import SpacemouseInterventionWrapper
    from airbot_rl.devices import AirbotConfig

    # Create mock environment
    robot_config = AirbotConfig(
        arm_names=["left", "right"][:num_arms],
        arm_ports=[50051, 50053][:num_arms],
    )
    env_config = AirbotEnvConfig(robot_config=robot_config)
    env = MockAirbotEnv(env_config)

    # Wrap with intervention wrapper
    wrapped_env = SpacemouseInterventionWrapper(
        env,
        num_arms=num_arms,
        use_mock=use_mock,
        auto_calibrate=False,
    )

    print(f"Environment action space: {wrapped_env.action_space.shape}")
    print(f"Expected action dim: {7 * num_arms}")
    print()
    print("Controls:")
    print("  - Each SpaceMouse left button -> toggle intervention for that arm only")
    print("  - Each SpaceMouse right button -> toggle gripper for that arm only")
    print()
    print("Press Ctrl+C to exit")
    print("=" * 70)

    try:
        obs, info = wrapped_env.reset()
        print(f"\nInitial observation shape: state={obs['state'].shape}")
        print(f"Initial arm_intervention_states: {info['arm_intervention_states']}")
        print()

        step = 0
        while True:
            # Generate random policy action
            policy_action = wrapped_env.action_space.sample()

            # Step environment (SpaceMouse can override)
            obs, reward, terminated, truncated, info = wrapped_env.step(policy_action)

            # Print info every 10 steps
            if step % 10 == 0:
                arm_states = info.get("arm_intervention_states", [])
                states_str = ", ".join(
                    f"{['left', 'right'][i]}={'INT' if s else 'AUT'}"
                    for i, s in enumerate(arm_states)
                )
                executed = info.get("executed_action", policy_action)

                print(
                    f"\rStep {step:4d} | {states_str} | "
                    f"is_intervention={info['is_intervention']} | "
                    f"action_dim={len(executed)}",
                    end="",
                    flush=True,
                )

            step += 1
            time.sleep(0.05)

            if terminated or truncated:
                print("\nEpisode ended, resetting...")
                obs, info = wrapped_env.reset()
                step = 0

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    finally:
        wrapped_env.close()
        print("\nIntervention wrapper closed")
        print(f"Final stats: {wrapped_env.intervention_stats}")


def main():
    parser = argparse.ArgumentParser(description="Test SpaceMouse functionality")
    parser.add_argument(
        "--dual", action="store_true",
        help="Test dual SpaceMouse for bimanual control (num_arms=2)"
    )
    parser.add_argument(
        "--auto-calibrate", action="store_true",
        help="Run auto-calibration to detect left/right SpaceMouse"
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use mock SpaceMouse (no hardware required)"
    )
    parser.add_argument(
        "--wrapper", action="store_true",
        help="Test SpacemouseInterventionWrapper with mock environment"
    )
    parser.add_argument(
        "--num-arms", type=int, default=2, choices=[1, 2],
        help="Number of arms for wrapper test (default: 2)"
    )

    args = parser.parse_args()

    if args.wrapper:
        test_intervention_wrapper(use_mock=args.mock, num_arms=args.num_arms)
    elif args.dual:
        test_dual_spacemouse(use_mock=args.mock, auto_calibrate=args.auto_calibrate)
    else:
        test_single_spacemouse(use_mock=args.mock)


if __name__ == '__main__':
    main()
