#!/usr/bin/env python3
"""
Test script for Quest3 VR controller input via ROS2.

This script helps you:
1. Verify ROS2 topics are being received
2. Identify axis/button indices for your Quest3 controller
3. Verify coordinate frame mapping
4. Test hybrid delta computation without robot connection

Usage:
    # Print raw data (identify axes/buttons)
    python test_vr_input.py --print-raw

    # Test delta computation
    python test_vr_input.py

    # Custom topic names
    python test_vr_input.py --tf-topic /tf --joy-topic /quest/joystick
"""

import argparse
import sys
import time
import numpy as np

from vr_input import DualVRInput, VRState


def print_raw_mode(vr: DualVRInput):
    """Print raw ROS2 data to identify axis/button mapping."""
    print("\n" + "=" * 70)
    print("RAW DATA MODE — Move controllers and press buttons to identify mapping")
    print("=" * 70)
    print()
    print("Waiting for VR data...")

    # Wait for data
    timeout = 10.0
    start = time.time()
    while not vr.has_data() and time.time() - start < timeout:
        time.sleep(0.1)

    if not vr.has_data():
        print(f"[ERROR] No VR data received after {timeout}s. Check:")
        print("  1. Quest3 is streaming")
        print("  2. ROS2 topics are published (ros2 topic list)")
        print("  3. Topic names match (--tf-topic, --joy-topic)")
        return

    print("Data received! Streaming raw values...\n")

    try:
        while True:
            # Get raw data
            poses = vr.get_hand_poses()
            axes, buttons = vr.get_joy_data()

            # Print hand poses
            print("\033[2J\033[H")  # Clear screen
            print("=" * 70)
            print("Quest3 VR Raw Data (press Ctrl+C to exit)")
            print("=" * 70)

            print("\n--- HAND POSES (from /tf) ---")
            for side in ["left", "right"]:
                pose = poses.get(side)
                if pose is not None:
                    pos, quat = pose
                    print(f"  {side.upper():6s}: pos=[{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]  "
                          f"quat=[{quat[0]:+.4f}, {quat[1]:+.4f}, {quat[2]:+.4f}, {quat[3]:+.4f}]")
                else:
                    print(f"  {side.upper():6s}: (no data)")

            print(f"\n--- JOYSTICK AXES ({len(axes)} axes) ---")
            for i, val in enumerate(axes):
                bar = "#" * int(abs(val) * 20)
                sign = "+" if val >= 0 else "-"
                highlight = " <<<" if abs(val) > 0.1 else ""
                print(f"  [{i:2d}] {val:+.4f} {sign}{bar}{highlight}")

            print(f"\n--- JOYSTICK BUTTONS ({len(buttons)} buttons) ---")
            active = [f"[{i}]" for i, b in enumerate(buttons) if b]
            if active:
                print(f"  PRESSED: {' '.join(active)}")
            else:
                print("  (none pressed)")

            # Print all button states in grid
            row = "  "
            for i, b in enumerate(buttons):
                row += f"{i:2d}:{'X' if b else '.'} "
                if (i + 1) % 8 == 0:
                    print(row)
                    row = "  "
            if row.strip():
                print(row)

            time.sleep(0.05)  # ~20Hz update

    except KeyboardInterrupt:
        print("\n\nDone.")


def test_delta_mode(vr: DualVRInput):
    """Test hybrid delta computation."""
    print("\n" + "=" * 70)
    print("DELTA MODE TEST — Squeeze grip to engage, move to see deltas")
    print("=" * 70)
    print()
    print("Controls:")
    print("  - Squeeze grip: Engage controller (VR → delta output)")
    print("  - Release grip: Disengage (clutch)")
    print("  - Squeeze trigger: Toggle gripper")
    print()
    print("Waiting for VR data...")

    timeout = 10.0
    start = time.time()
    while not vr.has_data() and time.time() - start < timeout:
        time.sleep(0.1)

    if not vr.has_data():
        print("[ERROR] No VR data received. Check ROS2 topics.")
        return

    print("Data received! Streaming deltas...\n")
    vr.reset()

    iteration = 0
    try:
        while True:
            action, states, changed = vr.get_action()

            if changed:
                print(f"\n{'='*50}")
                for i, name in enumerate(["LEFT", "RIGHT"]):
                    state_str = "ENGAGED" if states[i] == VRState.INTERVENING else "RELEASED"
                    print(f"  {name} arm: {state_str}")
                print(f"{'='*50}\n")

            if iteration % 5 == 0:
                left_state = "[ENG]" if states[0] == VRState.INTERVENING else "[REL]"
                right_state = "[ENG]" if states[1] == VRState.INTERVENING else "[REL]"

                left_pos = action[0:3]
                left_quat = action[3:7]
                left_grip = action[7]
                right_pos = action[8:11]
                right_quat = action[11:15]
                right_grip = action[15]

                status = (
                    f"\rL {left_state} "
                    f"pos[{left_pos[0]:+.4f},{left_pos[1]:+.4f},{left_pos[2]:+.4f}] "
                    f"q[{left_quat[0]:+.3f},{left_quat[1]:+.3f},{left_quat[2]:+.3f},{left_quat[3]:+.3f}] "
                    f"G:{left_grip:.1f} | "
                    f"R {right_state} "
                    f"pos[{right_pos[0]:+.4f},{right_pos[1]:+.4f},{right_pos[2]:+.4f}] "
                    f"G:{right_grip:.1f}"
                )
                print(status, end="", flush=True)

            iteration += 1
            time.sleep(0.02)  # 50Hz

    except KeyboardInterrupt:
        print("\n\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description="Test Quest3 VR controller input",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--print-raw", action="store_true",
        help="Print raw ROS2 data to identify axis/button mapping"
    )
    parser.add_argument(
        "--tf-topic", type=str, default="/tf",
        help="ROS2 topic for TF transforms (default: /tf)"
    )
    parser.add_argument(
        "--joy-topic", type=str, default="/quest/joystick",
        help="ROS2 topic for joystick data (default: /quest/joystick)"
    )
    parser.add_argument(
        "--pos-scale", type=float, default=1.0,
        help="Position delta scale (default: 1.0)"
    )
    parser.add_argument(
        "--rot-scale", type=float, default=1.0,
        help="Rotation delta scale (default: 1.0)"
    )
    parser.add_argument(
        "--grip-threshold", type=float, default=0.5,
        help="Grip axis threshold for engage (default: 0.5)"
    )
    parser.add_argument(
        "--left-grip-axis", type=int, default=6,
        help="Left grip axis index (default: 6)"
    )
    parser.add_argument(
        "--right-grip-axis", type=int, default=7,
        help="Right grip axis index (default: 7)"
    )
    parser.add_argument(
        "--left-trigger-button", type=int, default=8,
        help="Left trigger button index (default: 8)"
    )
    parser.add_argument(
        "--right-trigger-button", type=int, default=9,
        help="Right trigger button index (default: 9)"
    )
    parser.add_argument(
        "--settle-duration", type=float, default=0.5,
        help="Seconds to hold position after re-engaging grip (default: 0.5)"
    )

    args = parser.parse_args()

    vr = DualVRInput(
        pos_scale=args.pos_scale,
        rot_scale=args.rot_scale,
        grip_threshold=args.grip_threshold,
        tf_topic=args.tf_topic,
        joy_topic=args.joy_topic,
        left_grip_axis=args.left_grip_axis,
        right_grip_axis=args.right_grip_axis,
        left_trigger_button=args.left_trigger_button,
        right_trigger_button=args.right_trigger_button,
        settle_duration=args.settle_duration,
    )

    try:
        if args.print_raw:
            print_raw_mode(vr)
        else:
            test_delta_mode(vr)
    finally:
        vr.close()


if __name__ == "__main__":
    main()