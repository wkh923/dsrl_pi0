import threading
import time
from collections import deque

from airbot_py import RealtimeControllerBase

from .proto_inport import arm_pb2, arm_pb2_grpc, types_pb2, utils_pb2, utils_pb2_grpc


class ServoController(RealtimeControllerBase):
    def __init__(self, client_id, control_stub, logger):
        super().__init__(client_id, control_stub, logger, 10)
        self._eef_cmd_queue = deque(maxlen=10)

    def reset(self):
        super().reset()
        self._eef_cmd_queue.clear()

    def append_eef_cmd(self, cmd):
        self._eef_cmd_queue.append(cmd)
        self._condition.release()

    def servo_cmd_gen(self):
        def _gen_request():
            request = arm_pb2.SetStateRequest()
            while not self._stop_event.is_set():
                del request.component_names[:]
                del request.commands[:]
                if not self._condition.acquire(timeout=0.1):
                    continue
                if len(self._cmd_queue) > 0:
                    servo_cmd = self._cmd_queue.popleft()
                    request.component_names.append("arm")
                    if isinstance(servo_cmd, types_pb2.JointStates):
                        request.commands.add(joint_states=servo_cmd)
                    if isinstance(servo_cmd, types_pb2.Transform):
                        request.commands.add(transform=servo_cmd)
                    if isinstance(servo_cmd, types_pb2.Twist):
                        request.commands.add(twist=servo_cmd)
                if len(self._eef_cmd_queue) > 0:
                    eef_cmd = self._eef_cmd_queue.popleft()
                    request.component_names.append("eef")
                    if isinstance(eef_cmd, types_pb2.JointStates):
                        request.commands.add(joint_states=eef_cmd)
                if len(request.component_names) == 2:
                    self._condition.acquire(timeout=0)
                elif len(request.component_names) == 0:
                    self._condition.release()
                    self.logger.debug("Empty command but condition acquired!")
                    continue
                now = time.time_ns()
                request.stamp.sec = int(now / 1e9)
                request.stamp.nanosec = int(now % 1e9)
                request.request_id = self.client_id
                yield request

        for result in self._control_stub.SetState(_gen_request()):
            if not result.status == arm_pb2.Status.RUNNING:
                self.logger.error("Failed to servo")
