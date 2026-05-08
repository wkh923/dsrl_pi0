import time

from airbot_py import RealtimeControllerBase

from .proto_inport import arm_pb2, arm_pb2_grpc, types_pb2, utils_pb2, utils_pb2_grpc


class MITController(RealtimeControllerBase):
    def __init__(self, gap, client_id, control_stub, logger):
        super().__init__(client_id, control_stub, logger)
        self._gap: float = gap

    def mit_cmd_gen(self):
        def _gen_request():
            request = arm_pb2.SetStateRequest()
            last_time = time.time()
            while not self._stop_event.is_set():
                del request.component_names[:]
                del request.commands[:]
                if not self._condition.acquire(timeout=0.1):
                    continue
                if len(self._cmd_queue) > 0:
                    mit_cmd = self._cmd_queue.popleft()
                    request.component_names.append("arm")
                    if isinstance(mit_cmd, types_pb2.MITControl):
                        request.commands.add(mit_control=mit_cmd)
                now = time.time_ns()
                request.stamp.sec = int(now / 1e9)
                request.stamp.nanosec = int(now % 1e9)
                request.request_id = self.client_id
                yield request
                sleep_time = self._gap - (time.time() - last_time)
                if sleep_time > 0:
                    time.sleep(sleep_time * 0.8)
                last_time = time.time()

        for result in self._control_stub.SetState(_gen_request()):
            if not result.status == arm_pb2.Status.RUNNING:
                self.logger.error("Failed to MIT control")
