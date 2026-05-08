import threading
from collections import deque


class RealtimeControllerBase:
    def __init__(self, client_id, control_stub, logger, max_queue_len=None):
        self.client_id = client_id
        self._control_stub = control_stub
        self.logger = logger
        self._cmd_queue = deque(maxlen=max_queue_len) if max_queue_len else deque()
        self._condition = threading.Semaphore(0)
        self._stop_event = threading.Event()
        self._thread = None

    def reset(self):
        self._cmd_queue.clear()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self._stop_event.clear()
        self._condition = threading.Semaphore(0)

    def start_thread(self, cmd_gen_func):
        if self._thread is None:
            self._thread = threading.Thread(target=cmd_gen_func)
            self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self._stop_event.clear()
        self._condition = threading.Semaphore(0)

    def append_cmd(self, cmd):
        self._cmd_queue.append(cmd)
        self._condition.release()
