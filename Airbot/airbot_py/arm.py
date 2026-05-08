"""
This is the Python SDK for AIRBOT arm product series (AIRBOT Play, AIRBOT Pro, AIRBOT Play Lite
and AIRBOT Replay). SDK for AIRBOT series is a thin gRPC client to communicate with driver.

The examples can also run with shortcuts:

| Example                                          | Shortcut                   | Launch via Python module                               |
| :----------------------------------------------- | :------------------------- | :----------------------------------------------------- |
| Keyboard control                                 | `arm_kbd_ctrl`             | `python3 -m airbot_examples.task_kbd_ctrl`             |
| Motion following                                 | `arm_follow`               | `python3 -m airbot_examples.task_follow`               |
| Switch control mode                              | `arm_switch_mode`          | `python3 -m airbot_examples.switch_mode`               |
| Load app                                         | `arm_load_app`             | `python3 -m airbot_examples.app_load`                  |
| Unload app                                       | `arm_unload_app`           | `python3 -m airbot_examples.app_unload`                |
| Print joint position                             | `arm_joint_state`          | `python3 -m airbot_examples.get_joint_pos`             |
| Print end pose                                   | `arm_end_pose`             | `python3 -m airbot_examples.get_end_pose`              |
| Parameter query                                  | `arm_get_params`           | `python3 -m airbot_examples.get_end_pose`              |
| Parameter update                                 | `arm_set_params`           | `python3 -m airbot_examples.set_params`                |
| Move end effector to a cartesian pose            | `arm_move_cart_pose`       | `python3 -m airbot_examples.move_cart_pose`            |
| Move to a set of joint angles                    | `arm_move_joint`           | `python3 -m airbot_examples.move_joint_pos`            |
| Move along a set of cartesian waypoints          | `arm_move_cart_waypoints`  | `python3 -m airbot_examples.move_with_cart_waypoints`  |
| Move along a set of joint waypoints              | `arm_move_joint_waypoints` | `python3 -m airbot_examples.move_with_joint_waypoints` |
| Swing example: continuous swing in joint space   | `arm_example_swing`        | `python3 -m airbot_examples.task_swing`                |
| Wipe example: continuous wipe in cartesian space | `arm_example_wipe`         | `python3 -m airbot_examples.task_wipe`                 |

"""

import logging
import threading
import time
import uuid
from collections import deque
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Union

import grpc
from airbot_py import MITController, ServoController
from typeguard import typechecked

from .proto_inport import arm_pb2, arm_pb2_grpc, types_pb2, utils_pb2, utils_pb2_grpc


class SpeedProfile(Enum):
    """
    Enum for speed profile of the robot.

    * `DEFAULT`: The default speed profile.
    * `SLOW`: The slow speed profile.
    * `FAST`: The fast speed profile. <span style="color: red">**WARNING**: \
        This speed profile is experimental and may cause the robot to move too \
        fast and cause damage to the robot or the environment. \
        Use at your own risk .</span>
    """

    DEFAULT = 0
    SLOW = 1
    FAST = 2


class RobotMode(Enum):
    """
    Enum for the control mode of the arm.

    * `PLANNING_POS`: planning mode. A single target is sent to driver server, \
        then a execution path is planned and execute. The target could be joint \
        positions or end cartesian pose.
    * `PLANNING_WAYPOINTS_PATH`: planning mode. Perform linear interpolation \
        between multiple waypoints in cartesian space, then execute the path without stopping.
    * `PLANNING_WAYPOINTS`: planning mode. Perform local planning between adjacent \
        waypoints, join the planned paths then execute the path. Deceleration may \
        occur near waypoints.
    * `SERVO_CART_TWIST`: servo mode in cartesian space. Servo would require \
        continuous commands to be sent to the driver server. In this mode, the \
        robot would follow the cartesian twist commands, each of which \
        representing the expected translational and rotational velocity of the \
        end flange or the effecting point of the end effector.
    * `SERVO_CART_POSE`: servo mode in cartesian space. Servo would require \
        continuous commands to be sent to the driver server. In this mode, the \
        robot would follow the cartesian pose commands, each of which \
        representing the expected cartesian pose of the end flange or the \
        effecting point of the end effector.
    * `SERVO_JOINT_POS`: servo mode in joint space. Servo would require \
        continuous commands to be sent to the driver server. In this mode, the \
        robot would follow the joint position commands, each of which \
        representing the expected joint position of the arm.
    * `SERVO_JOINT_VEL`: servo mode in joint space. Servo would require \
        continuous commands to be sent to the driver server. In this mode, the \
        robot would follow the joint velocity commands, each of which \
        representing the expected joint velocity of the arm.
    * `MIT_INTEGRATED`: MIT integrated mode. In this mode, the robot would \
        be in a integrated motor joint control.
    * `GRAVITY_COMP`: gravity compensation mode. In this mode, the robot would \
        be in a free-drive state with gravity compensated. In this mode the robot \
        would not follow any commands, but the robot could be dragged by external \
        forces.
    """

    PLANNING_POS = 10
    PLANNING_WAYPOINTS_PATH = 11
    PLANNING_WAYPOINTS = 12
    SERVO_JOINT_POS = 20
    SERVO_CART_POSE = 21
    SERVO_JOINT_VEL = 24
    SERVO_CART_TWIST = 25
    MIT_INTEGRATED = 80
    GRAVITY_COMP = 90
    INACTIVE = 98
    UNDEFINED = 99


class State(Enum):
    """
    Enum for the state of the driver service.

    * `INIT`: The driver service is initializing.
    * `SHUTDOWN`: The driver service is shutting down.
    * `POWERON`: The driver service is powered on and performing self-check.
    * `IDLE`: The driver service is idle and waiting for external SDK commands.
    * `APPLOADING`: The driver service is loading the application.
    * `APPLOADED`: The driver service has loaded the application and is ready to execute commands. \
        When a read-write app is loaded, the robot is in read-only state and \
        the SDK can only read the state of the robot.
    * `ERROR`: The driver service is in error state.
    """

    INIT = 0
    SHUTDOWN = 1
    POWERON = 2
    IDLE = 3
    APPLOADING = 4
    APPLOADED = 5
    ERROR = 6


def grpc_safe():
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            if not self._feedbacking:
                logging.error("Not connected to the server.")
                raise RuntimeError("Not connected to the server.")
            else:
                try:
                    return func(self, *args, **kwargs)
                except grpc.RpcError as rpc_error:
                    logging.error("gRPC error: %s", rpc_error.details())

        return wrapper

    return decorator


class AIRBOTArm:
    """
    The client instance for AIRBOT arm series.
    """

    def __init__(self, url: str = "localhost", port: int = 50051):
        """
        Initialize SDK instance with default values.
        Would NOT perform any actions including connecting to server.

        Args:
            url (str): The listening url for the driver server. Default: "localhost"
            port (int): The listening port for the driver server. Default: 50051
        """
        self.endpoint = url + ":" + str(port)
        self.logger = logging.getLogger(__name__)
        self.client_id = str(uuid.uuid4())
        self.gap = 1 / 250
        self.logger.info("Client ID: %s", self.client_id)
        self._control_stub = None
        self._feedback_stub = None
        self._utils_stub = None
        self._channel = None
        self._feedback_jointstates = None
        self._feedback_end_poses = None
        self._feedback_thread = None
        self._servo_controller = None
        self._mit_controller = None
        self._end_joint_states = None
        self._feedbacking = False

    def _status(self):
        stamp = time.time_ns()
        request = arm_pb2.GetStateRequest()
        request.stamp.sec = int(stamp / 1e9)
        request.stamp.nanosec = int(stamp % 1e9)
        request.request_id = self.client_id
        request.component_names[:] = ["arm"]
        request.get_joint = True
        request.get_transform = True
        try:
            for fb in self._feedback_stub.GetState(request):
                self._feedback_jointstates = fb.state
                self._feedback_end_poses = fb.transform
                if not self._feedbacking:
                    break
        except grpc.RpcError as _:
            self._feedbacking = False
            self._feedback_end_poses = None
            self._feedback_jointstates = None
            self._feedback_stub = None

            if self._servo_controller is not None:
                self._servo_controller.stop()
                self._servo_controller = None

            if self._control_stub is not None:
                self._control_stub = None
                self._feedback_stub = None
                self._utils_stub = None
                if self._channel is not None:
                    self._channel.close()
                self._channel = None

    @grpc_safe()
    def get_joint_pos(self) -> Optional[List[float]]:
        """
        Read current joint positions (in radian) from the SDK server.

        The returned list only contains the joint positions of the arm itself.
        The position of the end effector (if any) is not included.

        If SDK has not received joint states from the server, None is returned

        Returns:
            joint_pos (list[float] | None): the current positions of the arm joints.
        """
        if self._feedback_jointstates is None:
            logging.error("No joint states received yet.")
            return None
        else:
            return [
                i[1]
                for i in sorted(
                    (
                        (name, position)
                        for name, position in zip(
                            self._feedback_jointstates.name,
                            self._feedback_jointstates.position,
                        )
                    ),
                    key=lambda x: x[0],
                )
                if i[0].startswith("joint")
            ]

    @grpc_safe()
    def get_eef_pos(self) -> Optional[List[float]]:
        """
        Read current end effector width (in meters) from the SDK server.
        Currently only parallel two-finger end effector is supported.

        If no end effector is installed, an empty list is returned.

        If the SDK has not received joint states from the server, None is returned.

        Returns:
            eef_pos (list[float] | None): the current end effector width of the arm. \
            None if SDK has not received joint states or no end \
            effector is installed.
        """
        if self._feedback_jointstates is None:
            logging.error("No joint states received yet.")
            return None
        else:
            return [
                i[1]
                for i in sorted(
                    (
                        (name, position)
                        for name, position in zip(
                            self._feedback_jointstates.name,
                            self._feedback_jointstates.position,
                        )
                    ),
                    key=lambda x: x[0],
                )
                if i[0] == "gripper_mapping_controller"
            ]

    @grpc_safe()
    def get_joint_vel(self) -> Optional[List[float]]:
        """
        Read current joint velocities (in radian/s) from the SDK server.

        The returned list only contains the joint velocities of the arm itself.
        The position of the end effector (if any) is not included.

        If SDK has not received joint states from the server, None is returned

        Returns:
            joint_vel (list[float] | None): the current velocities of the arm joints.
        """
        if self._feedback_jointstates is None:
            logging.error("No joint states received yet.")
            return None
        else:
            return [
                i[1]
                for i in sorted(
                    (
                        (name, velocity)
                        for name, velocity in zip(
                            self._feedback_jointstates.name,
                            self._feedback_jointstates.velocity,
                        )
                    ),
                    key=lambda x: x[0],
                )
                if i[0].startswith("joint")
            ]

    @grpc_safe()
    def get_joint_eff(self) -> Optional[List[float]]:
        """
        Read current joint efforts (in Nm) from the SDK server.

        The returned list only contains the joint efforts of the arm itself.
        The position of the end effector (if any) is not included.

        If SDK has not received joint states from the server, None is returned.

        Returns:
            joint_eff (float | None): the current efforts of the arm joints.
        """
        if self._feedback_jointstates is None:
            logging.error("No joint states received yet.")
            return None
        else:
            return [
                i[1]
                for i in sorted(
                    (
                        (name, effort)
                        for name, effort in zip(
                            self._feedback_jointstates.name,
                            self._feedback_jointstates.effort,
                        )
                    ),
                    key=lambda x: x[0],
                )
                if i[0].startswith("joint")
            ]

    @grpc_safe()
    def get_eef_eff(self) -> Optional[List[float]]:
        """
        Read current end effector torque output (in Nm) from the SDK server.
        Currently only parallel two-finger end effector is supported.

        If no end effector is installed, an empty list is returned.

        If the SDK has not received joint states from the server, None is returned.

        Returns:
            eef_eff (list[float]): the current end effector torque output of the arm. \
            None if SDK has not received joint states or no end \
            effector is installed.
        """
        if self._feedback_jointstates is None:
            logging.error("No joint states received yet.")
            return None
        else:
            return [
                i[1]
                for i in sorted(
                    (
                        (name, effort)
                        for name, effort in zip(
                            self._feedback_jointstates.name,
                            self._feedback_jointstates.effort,
                        )
                    ),
                    key=lambda x: x[0],
                )
                if i[0] == "gripper_mapping_controller"
            ]

    @grpc_safe()
    def get_end_pose(self) -> Optional[List[List[float]]]:
        """
        Read current end pose from the SDK server.

        The zero point of the reference frame for the end pose is the intersection point
        of the rotating axis of first joint and installing plane.
        The axes of reference frame for the end are x axis pointing forward, y axis
        pointing leftward and z axis pointing upward.

        If no end effector is installed, the pose of the end flange is returned.
        If any end effector is installed, the pose of the effecting point
        is returned. For example, the effecting point of AIRBOT G2 is the grasping
        point in the front.

        TODO: Replace with actual doc url
        For detailed explanation of the reference frame and the end link used
        for get_end_pose, please refer to [https://www.bing.com](https://www.bing.com)

        Returns:
            end_pose (list[list[float]]): A list of two float-list. The first list is \
            the cartesian translation of the pose in meters, and the second list \
            is the orientation quaternion (x, y, z, w) of the pose.

        """
        if self._feedback_end_poses is None:
            logging.error("No end poses received yet.")
            return None
        else:
            return [
                [
                    self._feedback_end_poses.translation.x,
                    self._feedback_end_poses.translation.y,
                    self._feedback_end_poses.translation.z,
                ],
                [
                    self._feedback_end_poses.orientation.x,
                    self._feedback_end_poses.orientation.y,
                    self._feedback_end_poses.orientation.z,
                    self._feedback_end_poses.orientation.w,
                ],
            ]

    def connect(self) -> bool:
        """
        Connect to the server if currently not connected.

        Directly calling this method is deprecated in favor of using the `with` statement:

        ```python
        # Use:
        with AIRBOTArm() as robot:
            ...

        # instead of
        robot = AIRBOTArm()
        robot.connect()
        ...
        robot.disconnect()
        ```

        This method is idempotent. Calling this method if SDK has already
        connected to the server would cause no harm.

        Returns:
            result (bool): True if connected successfully
        """
        if self._control_stub is None:
            self._channel = grpc.insecure_channel(self.endpoint)
            self._control_stub = arm_pb2_grpc.ControlServiceStub(self._channel)
            self._feedback_stub = arm_pb2_grpc.FeedbackServiceStub(self._channel)
            self._utils_stub = utils_pb2_grpc.UtilsServiceStub(self._channel)

            self._feedbacking = True
            self._feedback_thread = threading.Thread(target=self._status)
            self._feedback_thread.start()
            start_time = time.time()
            while self._feedback_jointstates is None:
                time.sleep(0.01)
                if time.time() - start_time > 1:
                    logging.error("Failed to connect to the server.")
                    self._feedbacking = False
                    self._feedback_thread.join()
                    self._feedback_thread = None
                    self._control_stub = None
                    self._feedback_stub = None
                    self._utils_stub = None
                    if self._channel is not None:
                        self._channel.close()
                    self._channel = None
                    return False

            self._servo_controller = ServoController(
                self.client_id, self._control_stub, self.logger
            )

            return True
        else:
            logging.warning("Already connected to the server.")
            return True

    def disconnect(self) -> bool:
        """
        Disconnect from to the server if currently not connected.

        Directly calling this method is deprecated in favor of using the `with` statement:

        ```python
        # Use:
        with AIRBOTArm() as robot:
            ...

        # instead of
        robot = AIRBOTArm()
        robot.connect()
        ...
        robot.disconnect()
        ```

        This method is idempotent. Calling this method if SDK has already
        disconnected from the server would cause no harm.

        Returns:
            result(bool): True if connected successfully
        """
        if self._feedbacking:
            self._feedbacking = False
            if self._feedback_thread is not None:
                self._feedback_thread.join()
                self._feedback_thread = None

        if self._servo_controller is not None:
            self._servo_controller.stop()
            self._servo_controller = None

        if self._mit_controller is not None:
            self._mit_controller.stop()
            self._mit_controller = None

        if self._control_stub is not None:
            self._control_stub = None
            self._feedback_stub = None
            self._utils_stub = None
            self._channel.close()
            self._channel = None
            return True
        else:
            logging.warning("Already disconnected from the server.")
            return True

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    @grpc_safe()
    @typechecked
    def switch_mode(self, mode: RobotMode) -> bool:
        """
        Switch the control mode of the robot.

        The switching would not be effective if the sdk is in read-only state (when
        an read-write app is loaded).

        Current supported mode:

        * `RobotMode.PLANNING_POS`: planning mode. A single target is sent to
          driver server, then a execution path is planned and execute. The
          target could be joint positions or end cartesian pose.
        * `RobotMode.SERVO_CART_TWIST`: servo mode in cartesian space. Servo
          would require continuous commands to be sent to the driver server.
          In this mode, the robot would follow the cartesian twist commands,
          each of which representing the expected translational and rotational
          velocity of the end flange or the effecting point of the end effector.
        * `RobotMode.SERVO_CART_POSE`: servo mode in cartesian space. Servo
          would require continuous commands to be sent to the driver server.
          In this mode, the robot would follow the cartesian pose commands,
          each of which representing the expected cartesian pose of the end
          flange or the effecting point of the end effector.
        * `RobotMode.SERVO_JOINT_POS`: servo mode in joint space. Servo would
          require continuous commands to be sent to the driver server. In this
          mode, the robot would follow the joint position commands, each of
          which representing the expected joint position of the arm.
        * `RobotMode.SERVO_JOINT_VEL`: servo mode in joint space. Servo would
          require continuous commands to be sent to the driver server. In this
          mode, the robot would follow the joint velocity commands, each of
          which representing the expected joint velocity of the arm.
        * `RobotMode.GRAVITY_COMP`: gravity compensation mode. In this mode,
          the robot would be in a free-drive state with gravity compensated.
          In this mode the robot would not follow any commands, but the robot
          could be dragged by external forces.

        Args:
            mode (RobotMode): The mode to switch to.

        Returns:
            result (bool): True if the mode was switched successfully, False otherwise.
        """
        # Per request from test team, block invalid switch mode call:
        # RobotMode.INACTIVE
        # RobotMode.UNDEFINED
        if mode in (RobotMode.INACTIVE, RobotMode.UNDEFINED):
            logging.error("Invalid mode: %s", mode.name)
            return False

        if self._control_stub is None:
            logging.error("Not connected to the server.")
            return False

        if self._servo_controller is not None:
            self._servo_controller.reset()

        if self._mit_controller is not None:
            self._mit_controller.reset()

        now = time.time_ns()
        request = arm_pb2.SetModeRequest()
        request.stamp.sec = int(now / 1e9)
        request.stamp.nanosec = int(now % 1e9)
        request.request_id = self.client_id
        request.component_names[:] = ["arm"]
        request.modes[:] = [mode.value]
        response = self._control_stub.SetMode(request)

        if response.modes[0] == mode.value:
            self.logger.info("Switched to %s successfully.", mode.name)
            return True
        else:
            self.logger.error("Failed to %s.", mode.name)
            return False

    @grpc_safe()
    @typechecked
    def servo_joint_vel(self, joint_vel: List[float]) -> None:
        """
        Send continuous joint velocity commands to the robot.
        This method should run in `SERVO_JOINT_VEL` mode.
        """
        if self._servo_controller is None:
            self._servo_controller = ServoController(
                self.client_id, self._control_stub, self.logger
            )
        self._servo_controller.start_thread(self._servo_controller.servo_cmd_gen)
        servo_cmd = types_pb2.JointStates()
        servo_cmd.name[:] = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        servo_cmd.velocity[:] = joint_vel
        self._servo_controller.append_cmd(servo_cmd)

    @grpc_safe()
    @typechecked
    def servo_joint_pos(self, joint_pos: List[float]) -> None:
        """
        Send continuous joint position commands to the robot.
        This method should run in `SERVO_JOINT_POS` mode.
        """
        if self._servo_controller is None:
            self._servo_controller = ServoController(
                self.client_id, self._control_stub, self.logger
            )
        self._servo_controller.start_thread(self._servo_controller.servo_cmd_gen)
        servo_cmd = types_pb2.JointStates()
        servo_cmd.name[:] = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        servo_cmd.position[:] = joint_pos
        self._servo_controller.append_cmd(servo_cmd)

    @grpc_safe()
    @typechecked
    def servo_cart_pose(self, cart_pose: List[List[float]]) -> None:
        """
        Send continuous cartesian pose commands to the robot.
        This method should run in `SERVO_CART_POSE` mode.
        """
        if self._servo_controller is None:
            self._servo_controller = ServoController(
                self.client_id, self._control_stub, self.logger
            )
        self._servo_controller.start_thread(self._servo_controller.servo_cmd_gen)
        servo_cmd = types_pb2.Transform()
        servo_cmd.translation.x = cart_pose[0][0]
        servo_cmd.translation.y = cart_pose[0][1]
        servo_cmd.translation.z = cart_pose[0][2]
        servo_cmd.orientation.x = cart_pose[1][0]
        servo_cmd.orientation.y = cart_pose[1][1]
        servo_cmd.orientation.z = cart_pose[1][2]
        servo_cmd.orientation.w = cart_pose[1][3]
        self._servo_controller.append_cmd(servo_cmd)

    @grpc_safe()
    @typechecked
    def servo_cart_twist(self, cart_twist: List[List[float]]) -> None:
        """
        Send continuous cartesian twist commands to the robot.
        This method should run in `SERVO_CART_TWIST` mode.
        """
        if self._servo_controller is None:
            self._servo_controller = ServoController(
                self.client_id, self._control_stub, self.logger
            )
        self._servo_controller.start_thread(self._servo_controller.servo_cmd_gen)
        servo_cmd = types_pb2.Twist()
        servo_cmd.linear.x = cart_twist[0][0]
        servo_cmd.linear.y = cart_twist[0][1]
        servo_cmd.linear.z = cart_twist[0][2]
        servo_cmd.angular.x = cart_twist[1][0]
        servo_cmd.angular.y = cart_twist[1][1]
        servo_cmd.angular.z = cart_twist[1][2]
        self._servo_controller.append_cmd(servo_cmd)

    @grpc_safe()
    @typechecked
    def servo_eef_pos(self, pos: Union[List[float], float]) -> None:
        """
        Send continuous end effector width commands to the robot.
        This method should run in any `SERVO*` mode.
        """
        if not isinstance(pos, list):
            pos = [pos]
        if self._servo_controller is None:
            self._servo_controller = ServoController(
                self.client_id, self._control_stub, self.logger
            )
        self._servo_controller.start_thread(self._servo_controller.servo_cmd_gen)
        servo_cmd = types_pb2.JointStates()
        servo_cmd.name[:] = ["eef"]
        servo_cmd.position[:] = pos
        self._servo_controller.append_eef_cmd(servo_cmd)

    @grpc_safe()
    @typechecked
    def servo_eef_force(self, force: List[float]) -> None:
        """
        UNIMPLEMENTED — This method has been removed and will be replaced.

        .. warning::
            This method is currently **UNIMPLEMENTED** and will raise a `NotImplementedError`
            if called.

            A new force command interface will be available in **version 5.3**.
            Until then, this method should not be used.

            Please refer to the upcoming 5.3 API for the replacement (e.g. `send_eef_force_command()`).
        """
        self.logger.warning(
            "servo_eef_force() is UNIMPLEMENTED. \nThis method has been removed and will be implemented in version 5.3."
        )

        # if self._servo_controller is None:
        #     self._servo_controller = ServoController(
        #         self.client_id, self._control_stub, self.logger
        #     )
        # self._servo_controller.start_thread(self._servo_controller.servo_cmd_gen)
        # servo_cmd = types_pb2.JointStates()
        # servo_cmd.name[:] = ["eef"]
        # servo_cmd.effort[:] = force
        # self._servo_controller.append_eef_cmd(servo_cmd)

    @grpc_safe()
    @typechecked
    def move_to_joint_pos(
        self,
        joint_pos: List[float],
        blocking: bool = True,
    ) -> bool:
        """
        Send one set of joint position command to the robot.
        This method should run in `PLANNING_POS` mode.

        The joint position command is sent to the driver server, and the robot
        would plan a path to reach the target joint position then execute it.

        Before calling this method, the robot should be in `PLANNING_POS` mode,
        otherwise the command would be ignored.

        Args:
            joint_pos (list[float]): The joint position commands in radian.
            blocking (bool): Whether to block until the command is finished.
                Default: True

        Returns:
            result (bool): True if the command was sent successfully, False otherwise.
        """

        def _generate_joint_pos(
            joint_pos: List[float],
        ) -> Iterator[arm_pb2.SetStateRequest]:
            request = arm_pb2.SetStateRequest()
            joint_states = types_pb2.JointStates()
            now = time.time_ns()
            request.stamp.sec = int(now / 1e9)
            request.stamp.nanosec = int(now % 1e9)
            request.request_id = self.client_id
            request.component_names[:] = ["arm"]
            joint_states.name[:] = [
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
            ]
            joint_states.position[:] = joint_pos
            request.commands.add(joint_states=joint_states)
            yield request

        result_iter = self._control_stub.SetState(_generate_joint_pos(joint_pos))

        first_result = next(result_iter)
        if first_result.status != arm_pb2.Status.ACCEPTED:
            self.logger.error("Failed to move to joint position")
            return False

        if blocking:
            for result in result_iter:
                if result.status == arm_pb2.Status.FINISHED:
                    self.logger.info("Moving to joint position...")
                    return True
            return False
        else:
            return True

    @grpc_safe()
    @typechecked
    def move_eff_pos(
        self,
        joint_pos: Union[List[float], float],
        blocking: bool = True,
    ) -> bool:
        """
        Previously name with typo for sending the command to the end effector.
        Renamed to `move_eef_pos`, but kept for backward compatibility.
        """
        return self.move_eef_pos(joint_pos, blocking)

    @grpc_safe()
    @typechecked
    def move_eef_pos(
        self,
        joint_pos: Union[List[float], float],
        blocking: bool = True,
    ) -> bool:
        """
        Send one joint position command to the end effector.
        This method should run in `PLANNING_POS` mode.

        Effective for parallel two-finger end effector only.

        Before calling this method, the robot should be in `PLANNING_POS` mode,
        otherwise the command would be ignored.

        Args:
            joint_pos (list[float]): The end effector width commands in meters.

        Returns:
            result (bool): True if the command was sent successfully, False otherwise.
        """
        # For backward compatibility
        if not isinstance(joint_pos, list):
            joint_pos = [joint_pos]

        def _generate_joint_pos(
            joint_pos: List[float],
        ) -> Iterator[arm_pb2.SetStateRequest]:
            request = arm_pb2.SetStateRequest()
            joint_states = types_pb2.JointStates()
            now = time.time_ns()
            request.stamp.sec = int(now / 1e9)
            request.stamp.nanosec = int(now % 1e9)
            request.request_id = self.client_id
            request.component_names[:] = ["eef"]
            joint_states.name[:] = ["eef"]
            joint_states.position[:] = joint_pos
            request.commands.add(joint_states=joint_states)
            yield request

        result_iter = self._control_stub.SetState(_generate_joint_pos(joint_pos))

        first_result = next(result_iter, None)
        if first_result is not None and first_result.status != arm_pb2.Status.ACCEPTED:
            self.logger.error("Failed to move to joint position")
            return False

        return True

    @grpc_safe()
    @typechecked
    def move_to_cart_pose(
        self,
        cart_pose: List[List[float]],
        blocking: bool = True,
    ) -> bool:
        """
        Send one set of cartesian pose command to the robot.
        This method should run in `PLANNING_POS` mode.

        The cartesian pose command is sent to the driver server, and the robot
        would plan a path to reach the target cartesian pose then execute it.

        The zero point of the reference frame for the end pose is the intersection point
        of the rotating axis of first joint and installing plane.
        The axes of reference frame for the end are x axis pointing forward, y axis
        pointing leftward and z axis pointing upward.

        Args:
            cart_pose (list[list[float]]): The cartesian pose commands. \
                The first list is the cartesian translation of the pose in meters, \
                and the second list is the orientation quaternion (x, y, z, w) of the pose.
            blocking (bool): Whether to block until the command is finished. \
                Default: True

        Returns:
            result (bool): True if the command was sent successfully, False otherwise.
        """

        def _generate_cart_pose(
            cart_pose: List[List[float]],
        ) -> Iterator[arm_pb2.SetStateRequest]:
            transform = types_pb2.Transform()
            request = arm_pb2.SetStateRequest()
            now = time.time_ns()
            request.stamp.sec = int(now / 1e9)
            request.stamp.nanosec = int(now % 1e9)
            request.request_id = self.client_id
            request.component_names[:] = ["arm"]
            transform.translation.x = cart_pose[0][0]
            transform.translation.y = cart_pose[0][1]
            transform.translation.z = cart_pose[0][2]
            transform.orientation.x = cart_pose[1][0]
            transform.orientation.y = cart_pose[1][1]
            transform.orientation.z = cart_pose[1][2]
            transform.orientation.w = cart_pose[1][3]
            request.commands.add(transform=transform)
            yield request

        result_iter = self._control_stub.SetState(_generate_cart_pose(cart_pose))

        first_result = next(result_iter)
        if first_result.status != arm_pb2.Status.ACCEPTED:
            self.logger.error("Failed to move to joint position")
            return False

        if blocking:
            for result in result_iter:
                if result.status == arm_pb2.Status.FINISHED:
                    self.logger.info("Moving to joint position...")
                    return True
            return False
        else:
            return True

    @grpc_safe()
    @typechecked
    def move_with_cart_waypoints(
        self,
        waypoints: List[List[List[float]]],
        blocking: bool = True,
    ) -> bool:
        """
        Move the robot arm along a series of cartesian waypoints.
        This method should run in `PLANNING_WAYPOINTS` or `PLANNING_WAYPOINTS_PATH` mode.

        The robot will interpolate through the given list of poses.

        Args:
            waypoints (list[list[list[float]]]): A list of poses, where each pose is a
                2-element list: [translation, orientation]
                - translation: list of 3 floats (x, y, z) in meters
                - orientation: list of 4 floats (x, y, z, w) as quaternion
            blocking (bool): Whether to wait until execution is finished.

        Returns:
            bool: True if the command was accepted and (optionally) completed.
        """

        def _generate_waypoints(
            cart_pose: List[List[List[float]]],
        ) -> Iterator[arm_pb2.SetStateRequest]:
            waypoints_obj = types_pb2.WayPoints()
            request = arm_pb2.SetStateRequest()
            now = time.time_ns()
            request.stamp.sec = int(now / 1e9)
            request.stamp.nanosec = int(now % 1e9)
            request.request_id = self.client_id
            request.component_names[:] = ["arm"]
            for e in cart_pose:
                transform = types_pb2.Transform()
                transform.translation.x = e[0][0]
                transform.translation.y = e[0][1]
                transform.translation.z = e[0][2]
                transform.orientation.x = e[1][0]
                transform.orientation.y = e[1][1]
                transform.orientation.z = e[1][2]
                transform.orientation.w = e[1][3]
                waypoints_obj.transform_array.extend([transform])

            command = types_pb2.Command()
            command.waypoints.CopyFrom(waypoints_obj)
            request.commands.append(command)
            yield request

        result_iter = self._control_stub.SetState(_generate_waypoints(waypoints))
        first_result = next(result_iter)
        if first_result.status != arm_pb2.Status.ACCEPTED:
            self.logger.error("Failed to move to joint position")
            return False
        if blocking:
            print("Waiting for arm to finish moving...")
            for result in result_iter:
                if result.status == arm_pb2.Status.FINISHED:
                    self.logger.info("Arm movement finished")
                    return True
            return False
        else:
            return True

    @grpc_safe()
    @typechecked
    def move_with_joint_waypoints(
        self,
        waypoints: List[List[float]],
        blocking: bool = True,
    ) -> bool:
        """
        Move the robot arm along a series of joint space waypoints.
        This method should run in `PLANNING_WAYPOINTS` or `PLANNING_WAYPOINTS_PATH` mode.

        Args:
            waypoints (list[list[list[float]]]): A list of joint angle sets. Each element is a list of
                joint positions (floats) for the 6 joints [joint1, ..., joint6].
            blocking (bool): Whether to wait until execution is finished.

        Returns:
            bool: True if the command was accepted and (optionally) completed.
        """

        def _generate_waypoints(
            joint_pose: List[List[List[float]]],
        ) -> Iterator[arm_pb2.SetStateRequest]:
            waypoints_obj = types_pb2.WayPoints()
            request = arm_pb2.SetStateRequest()
            now = time.time_ns()
            request.stamp.sec = int(now / 1e9)
            request.stamp.nanosec = int(now % 1e9)
            request.request_id = self.client_id
            request.component_names[:] = ["arm"]

            for e in joint_pose:
                joint_states = types_pb2.JointStates()
                joint_states.name[:] = [
                    "joint1",
                    "joint2",
                    "joint3",
                    "joint4",
                    "joint5",
                    "joint6",
                ]
                joint_states.position[:] = e
                waypoints_obj.joint_states_array.extend([joint_states])

            command = types_pb2.Command()
            command.waypoints.CopyFrom(waypoints_obj)
            request.commands.append(command)
            yield request

        result_iter = self._control_stub.SetState(_generate_waypoints(waypoints))
        first_result = next(result_iter)
        if first_result.status != arm_pb2.Status.ACCEPTED:
            self.logger.error("Failed to move to joint position")
            return False
        if blocking:
            print("Waiting for arm to finish moving...")
            for result in result_iter:
                if result.status == arm_pb2.Status.FINISHED:
                    self.logger.info("Arm movement finished")
                    return True
            return False
        else:
            return True

    @grpc_safe()
    @typechecked
    def mit_joint_integrated_control(
        self,
        joint_pos: List[float],
        joint_vel: List[float],
        joint_eff: List[float],
        joint_kp: List[float],
        joint_kd: List[float],
    ) -> bool:
        """
        Send integrated 6 joint position command to the robot.
        This method should run in `MIT_INTEGRATED` mode.

        The joint position command is sent to the driver server, and the robot
        would in MIT_INTEGRATED mode.
        """
        if not all(
            len(param_list) == 6
            for param_list in [joint_pos, joint_vel, joint_eff, joint_kp, joint_kd]
        ):
            logging.error(
                "All joint command lists (pos, vel, eff, kp, kd) must be lists of 6 floats."
            )
            return False

        if self._mit_controller is None:
            self._mit_controller = MITController(
                self.gap, self.client_id, self._control_stub, self.logger
            )
        self._mit_controller.start_thread(self._mit_controller.mit_cmd_gen)
        mit_cmd = types_pb2.MITControl()
        mit_cmd.name[:] = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        mit_cmd.position[:] = joint_pos
        mit_cmd.velocity[:] = joint_vel
        mit_cmd.effort[:] = joint_eff
        mit_cmd.kp[:] = joint_kp
        mit_cmd.kd[:] = joint_kd
        self._mit_controller.append_cmd(mit_cmd)
        return True

    @grpc_safe()
    @typechecked
    def get_state(self) -> State:
        """
        Obtain current state of the driver service.

        Returns:
            state (State): The current state of the driver service.
        """
        stamp = time.time_ns()
        request = utils_pb2.GetStateRequest()
        request.stamp.sec = int(stamp / 1e9)
        request.stamp.nanosec = int(stamp % 1e9)
        request.request_id = self.client_id

        response = self._utils_stub.GetState(request)
        return State(response.state)

    @grpc_safe()
    @typechecked
    def get_control_mode(self) -> Optional[RobotMode]:
        """
        Obtain current control mode of the driver service.

        Returns:
            mode (RobotMode): The current control mode of the driver service.
        """
        stamp = time.time_ns()
        request = arm_pb2.GetModeRequest()
        request.stamp.sec = int(stamp / 1e9)
        request.stamp.nanosec = int(stamp % 1e9)
        request.request_id = self.client_id
        request.component_names[:] = ["arm"]

        response = self._feedback_stub.GetMode(request)
        return RobotMode(response.modes[0]) if response.modes else None

    @grpc_safe()
    @typechecked
    def load_app(self, name: str, params: Optional[dict] = None) -> bool:
        """
        Load an app to the driver server.

        When an read-write app is loaded, the SDK would enter read-only state,
        in which get methods are available but set methods are blocked. See
        [available states](./#airbot_py.arm--internal-states-of-the-driver)

        Currently available apps:
        * `record_replay_app/record_replay_app::RecordReplayApp`

        Args:
            name (str): The name of the app to load.
            params (dict): The parameters for the app. Default: {}

        Returns:
            result (bool): True if the app was loaded successfully, False otherwise.
        """
        stamp = time.time_ns()
        request = utils_pb2.LoadAppRequest()
        request.stamp.sec = int(stamp / 1e9)
        request.stamp.nanosec = int(stamp % 1e9)
        request.request_id = self.client_id
        request.app_name = name
        if params is not None:
            for key, value in params.items():
                request.params_key.append(key)
                request.params_value.append(str(value))

        response = self._utils_stub.LoadApp(request)
        return response.ok

    @grpc_safe()
    @typechecked
    def get_params(self, names: List[str]) -> Dict[str, Any]:
        """
        Get parameters from the driver server.

        Args:
            names (list[str]): The names of the parameters to get.
        Returns:
            params (dict): A dictionary of the parameters.
        """
        if len(names) == 0:
            logging.error("No parameter names provided.")
            return {}
        stamp = time.time_ns()
        request = arm_pb2.GetParamsRequest()
        request.stamp.sec = int(stamp / 1e9)
        request.stamp.nanosec = int(stamp % 1e9)
        request.request_id = self.client_id
        request.keys[:] = names

        response = self._feedback_stub.GetParams(request)
        return {
            k: getattr(v, v.WhichOneof("union"))
            for k, v in zip(response.keys, response.values)
        }

    @grpc_safe()
    @typechecked
    def set_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set parameters to the driver server.

        Only bool, integer, double and string types are supported.

        Args:
            params (dict): A dictionary of the parameters to set.
        Return:
            params (dict): A dictionary of the parameters with the given keys and actual \
                values after setting. The values are the same type as the input values.
        """
        if self._control_stub is None:
            return {}
        if len(params) == 0:
            logging.error("No parameter names provided.")
            return {}
        stamp = time.time_ns()
        request = arm_pb2.SetParamsRequest()
        request.stamp.sec = int(stamp / 1e9)
        request.stamp.nanosec = int(stamp % 1e9)
        request.request_id = self.client_id
        for key, value in params.items():
            request.keys.append(key)
            if isinstance(value, bool):
                request.values.add(bool_val=value)
            elif isinstance(value, float):
                request.values.add(double_val=value)
            elif isinstance(value, str):
                request.values.add(string_val=value)
            elif isinstance(value, int):
                request.values.add(int32_val=value)
            else:
                raise TypeError(f"Unsupported type: {type(value)}")
        response = self._control_stub.SetParams(request)
        return {
            k: getattr(v, v.WhichOneof("union"))
            for k, v in zip(response.keys, response.values)
        }

    @grpc_safe()
    @typechecked
    def set_speed_profile(self, profile: SpeedProfile):
        """
        Set the speed profile of the robot.

        Three speed profiles are available:

        * `SpeedProfile.DEFAULT`: Default speed profile.
        * `SpeedProfile.SLOW`: Slow speed profile.
        * `SpeedProfile.FAST`: Fast speed profile. \
            <span style="color: red">**Warning**: \
            This speed profile is experimental and may cause the robot to move too fast. \
            It is dangerous not to make sure the robot is in a safe environment. \
            Use at your own risk.</span>

        Args:
            profile (SpeedProfile): The speed profile to set.
        """
        if profile not in SpeedProfile:
            raise ValueError(f"Invalid speed profile: {profile}")
        elif profile == SpeedProfile.DEFAULT:
            self.set_params(
                {
                    "servo_node.moveit_servo.scale.linear": 0.2,
                    "servo_node.moveit_servo.scale.rotational": 0.2,
                    "servo_node.moveit_servo.scale.joint": 0.1,
                    "sdk_server.max_velocity_scaling_factor": 0.5,
                    "sdk_server.max_acceleration_scaling_factor": 0.1,
                }
            )
        elif profile == SpeedProfile.SLOW:
            self.set_params(
                {
                    "servo_node.moveit_servo.scale.linear": 0.05,
                    "servo_node.moveit_servo.scale.rotational": 0.05,
                    "servo_node.moveit_servo.scale.joint": 0.05,
                    "sdk_server.max_velocity_scaling_factor": 0.1,
                    "sdk_server.max_acceleration_scaling_factor": 0.02,
                }
            )
        elif profile == SpeedProfile.FAST:
            self.set_params(
                {
                    "servo_node.moveit_servo.scale.linear": 10.0,
                    "servo_node.moveit_servo.scale.rotational": 10.0,
                    "servo_node.moveit_servo.scale.joint": 1.0,
                    "sdk_server.max_velocity_scaling_factor": 1.0,
                    "sdk_server.max_acceleration_scaling_factor": 0.5,
                }
            )

    @grpc_safe()
    def unload_app(self) -> bool:
        """
        Unload the app from the driver server.

        When an read-write app is unloaded, the SDK would enter read-write state,
        in which set methods are available. See [available states]
        (#airbot_py.arm--internal-states-of-the-driver)

        Returns:
            result (bool): True if the app was unloaded successfully, False otherwise.
        """
        stamp = time.time_ns()
        request = utils_pb2.UnloadAppRequest()
        request.stamp.sec = int(stamp / 1e9)
        request.stamp.nanosec = int(stamp % 1e9)
        request.request_id = self.client_id

        response = self._utils_stub.UnloadApp(request)
        return response.ok

    @grpc_safe()
    def get_product_info(self) -> dict:
        stamp = time.time_ns()
        request = utils_pb2.GetProductInfoRequest()
        request.stamp.sec = int(stamp / 1e9)
        request.stamp.nanosec = int(stamp % 1e9)
        request.request_id = self.client_id

        response = self._utils_stub.GetProductInfo(request)
        return {
            "product_type": response.product_type,
            "sn": response.sn,
            "is_sim": response.is_sim,
            "interfaces": response.interfaces,
            "arm_types": response.arm_types,
            "eef_types": response.eef_types,
            "fw_versions": response.fw_versions,
        }


AIRBOTPlay = AIRBOTArm

"""Alias for `AIBROTArm`"""

AIRBOTPro = AIRBOTArm
"""Alias for `AIBROTArm`"""

AIRBOTPlayLite = AIRBOTArm
"""Alias for `AIBROTArm`"""
