import math
import threading
import time
import traceback
from typing import Union, Dict, Optional
import numpy as np
from airbot_data_collection.common.devices.cameras.utils import (
    CameraRGBDConfig,
    find_video_capture_devices,
)
from airbot_data_collection.basis import Sensor
from pyrealsense2 import config as RSConfig  # noqa: N812
from pyrealsense2 import format as RSFormat  # noqa: N812
from pyrealsense2 import pipeline as RSPipeline  # noqa: N812
from pyrealsense2 import stream as RSStream  # noqa: N812
from pyrealsense2 import align as RSAlign  # noqa: N812
from pyrealsense2 import camera_info as RSCameraInfo  # noqa: N812
from pyrealsense2 import context as RSContext  # noqa: N812
from pyrealsense2 import option as RSOption  # noqa: N812


def find_camera_indices(
    raise_when_empty=True, serial_number_index: int = 1
) -> list[str]:
    """
    Find the serial numbers of the Intel RealSense cameras
    connected to the computer.
    """
    camera_ids = []
    for device in RSContext().query_devices():
        serial_number = device.get_info(RSCameraInfo(serial_number_index))
        camera_ids.append(serial_number)

    if raise_when_empty and len(camera_ids) == 0:
        raise OSError(
            "No camera was detected. Try re-plugging, or re-installing `librealsense` and its python wrapper `pyrealsense2`, or updating the firmware."
        )

    return camera_ids


def find_camera_device_ids(
    bus_to_serial: bool = False,
) -> dict[str, list[Union[str, int]]]:
    """Find the video capture devices corresponding to Intel RealSense cameras."""
    ctx = RSContext()
    devices = find_video_capture_devices()

    mappings = {}

    for dev in ctx.devices:
        serial = dev.get_info(RSCameraInfo.serial_number)
        physical_port: str = dev.get_info(RSCameraInfo.physical_port)
        device_path = physical_port.rsplit("/", 1)[-1]  # Get the last part of the path
        for bus_info, video_devices in devices.items():
            if f"/dev/{device_path}" in video_devices:
                if bus_to_serial:
                    mappings[bus_info] = serial
                else:
                    mappings[serial] = video_devices
    return mappings


class IntelRealSenseCameraConfig(CameraRGBDConfig):
    force_hardware_reset: bool = True

    # Lock exposure / gain / white balance so RGB looks consistent across
    # rollouts. Auto-exposure left enabled previously caused periodic ~2 s
    # `wait_for_frames` stalls every ~16 control steps because AE re-metering
    # paused frame production on one of the cameras. Mirrors the lock applied
    # in VLA-RL's data_collection/common/devices/cameras/intelrealsense.py.
    # Tune via realsense-viewer for your room lighting if defaults look off.
    enable_auto_exposure: bool = False
    exposure_value: int = 200          # ×100 μs (= 20 ms)
    gain_value: int = 8
    enable_auto_white_balance: bool = False
    white_balance_value: int = 3270    # Kelvin; only used when auto_wb=False.
                                        # Measured via auto-WB convergence on
                                        # base camera 243222074218 under the
                                        # lab's warm indoor lighting. Re-probe
                                        # with /tmp/probe_wb.py if room
                                        # lighting changes or colors look off.

    def model_post_init(self, context):
        at_least_one_is_not_none = (
            self.fps is not None or self.width is not None or self.height is not None
        )
        at_least_one_is_none = (
            self.fps is None or self.width is None or self.height is None
        )
        if at_least_one_is_not_none and at_least_one_is_none:
            raise ValueError(
                "For `fps`, `width` and `height`, either all of them need to be set, or none of them, "
                f"but {self.fps=}, {self.width=}, {self.height=} were provided."
            )
        self.camera_index = (
            str(self.camera_index) if self.camera_index is not None else None
        )


class IntelRealSenseCamera(Sensor):
    """Interface for Intel RealSense cameras."""

    config: IntelRealSenseCameraConfig

    def on_configure(self):
        config = self.config
        self.fps = config.fps
        self.width = config.width
        self.height = config.height
        self._rs_pipe = None
        self.logs = {}
        # Latest-frame cache, populated by librealsense's worker thread via
        # _on_frame callback. capture_observation() reads from this cache
        # instead of calling wait_for_frames(), which was deterministically
        # deadlocking after 16 successful calls in dsrl_pi0's Python env.
        self._latest_frame_lock = threading.Lock()
        self._latest_frame = None
        self._connect()
        return True

    def _on_frame(self, frame):
        """Invoked from librealsense's worker thread on each new frameset.

        The callback delivers a base ``rs.frame``; cast it to a
        ``composite_frame`` (frameset) so downstream code can call
        ``get_color_frame() / get_depth_frame()``. Single-stream pipelines
        deliver a non-frameset ``video_frame`` which we keep as-is and
        handle on the read side.

        Must call ``frame.keep()`` to extend the frame's lifetime past this
        callback return — otherwise librealsense reclaims it to the pool
        and the Python-side reference becomes invalid.
        """
        try:
            # Cast to composite_frame when the pipeline is multi-stream.
            try:
                if frame.is_frameset():
                    frame = frame.as_frameset()
            except Exception:
                pass  # leave as-is; capture_observation handles either type
            frame.keep()
            with self._latest_frame_lock:
                self._latest_frame = frame
        except Exception as e:
            # Don't let an exception in the callback kill the pipeline thread.
            print(f"[IntelRealSenseCamera({self.config.camera_index})] "
                  f"frame callback exception: {e}")

    def _connect(self):
        rs_config = RSConfig()
        if self.config.camera_index:
            rs_config.enable_device(self.config.camera_index)

        use_full_config = self.fps and self.width and self.height

        assert self.config.enable_color, "Now require color to be enabled. "

        if use_full_config:
            # TODO(rcadene): can we set rgb8 directly?
            rs_config.enable_stream(
                RSStream.color, self.width, self.height, RSFormat.rgb8, self.fps
            )
        else:
            rs_config.enable_stream(RSStream.color)

        if self.config.enable_depth:
            if use_full_config:
                rs_config.enable_stream(
                    RSStream.depth, self.width, self.height, RSFormat.z16, self.fps
                )
            else:
                rs_config.enable_stream(RSStream.depth)

        self._rs_pipe = RSPipeline()
        try:
            # Start with frame-callback variant: librealsense delivers
            # framesets to self._on_frame from its own worker thread.
            # This bypasses wait_for_frames / poll_for_frames entirely,
            # avoiding the 16-call deadlock observed with the polling API.
            profile = self._rs_pipe.start(rs_config, self._on_frame)
            # Lock exposure / gain / WB so RGB is consistent across rollouts
            # and to avoid the periodic AE-re-metering stalls (every ~16 ctrl
            # steps in `wait_for_frames`) observed with defaults.
            try:
                color_sensor = profile.get_device().first_color_sensor()
                color_sensor.set_option(
                    RSOption.enable_auto_exposure,
                    1 if self.config.enable_auto_exposure else 0,
                )
                if not self.config.enable_auto_exposure:
                    color_sensor.set_option(RSOption.exposure, self.config.exposure_value)
                    color_sensor.set_option(RSOption.gain, self.config.gain_value)
                color_sensor.set_option(
                    RSOption.enable_auto_white_balance,
                    1 if self.config.enable_auto_white_balance else 0,
                )
                if not self.config.enable_auto_white_balance:
                    color_sensor.set_option(
                        RSOption.white_balance, self.config.white_balance_value
                    )
                # Latest-frame-only queue: drop older buffered frames at the
                # producer side. Default 16 → after 16 successful reads the
                # pipeline state machine gets stuck in dsrl_pi0's Python env
                # (16-read failure pattern observed across single-cam, 3-cam,
                # every-step, every-5-step configurations). Setting to 1
                # eliminates that buffer entirely; we always get the latest
                # frame, which is what a real-time control loop wants anyway.
                color_sensor.set_option(RSOption.frames_queue_size, 1)
            except Exception as e:
                print(f"[IntelRealSenseCamera] Failed to lock exposure/WB: {e}")
            is_camera_open = True
        except RuntimeError:
            is_camera_open = False
            traceback.print_exc()
            available_cam_ids = find_camera_indices()

        # If the camera doesn't work, display the camera indices corresponding to
        # valid cameras.
        if not is_camera_open:
            # Verify that the provided `camera_index` is valid before printing the traceback
            if self.config.camera_index not in available_cam_ids:
                raise ValueError(
                    f"`camera_index` is expected to be one of these available cameras {available_cam_ids}, but {self.config.camera_index} is provided instead. "
                    "To find the camera index you should use, run `python lerobot/common/devices/cameras/intelrealsense.py`."
                )

            raise OSError(
                f"Can't access IntelRealSenseCamera({self.config.camera_index})."
            )

        if self.config.align_depth:
            self.align = RSAlign(RSStream.color)

        color_stream = profile.get_stream(RSStream.color)
        color_profile = color_stream.as_video_stream_profile()
        actual_fps = color_profile.fps()
        actual_width = color_profile.width()
        actual_height = color_profile.height()

        # Using `math.isclose` since actual fps can be a float (e.g. 29.9 instead of 30)
        if self.fps is not None and not math.isclose(
            self.fps, actual_fps, rel_tol=1e-3
        ):
            # Using `OSError` since it's a broad that encompasses issues related to device communication
            raise OSError(
                f"Can't set {self.fps=} for IntelRealSenseCamera({self.camera_index}). Actual value is {actual_fps}."
            )
        if self.width is not None and self.width != actual_width:
            raise OSError(
                f"Can't set {self.width=} for IntelRealSenseCamera({self.camera_index}). Actual value is {actual_width}."
            )
        if self.height is not None and self.height != actual_height:
            raise OSError(
                f"Can't set {self.height=} for IntelRealSenseCamera({self.camera_index}). Actual value is {actual_height}."
            )

        self.fps = round(actual_fps)
        self.width = round(actual_width)
        self.height = round(actual_height)

    def capture_observation(
        self, timeout: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """Read the latest frame cached by the librealsense callback thread.

        Returns:
            A dictionary containing the captured images.
        Raises:
            OSError: If no frame has been received within ``timeout`` seconds.
        """
        # Wait briefly for the callback thread to have populated _latest_frame.
        # In steady state this loop spins at most once (frame is already cached
        # at 30 Hz). The deadline only matters on the very first call after
        # _connect() and on real pipeline death.
        deadline_s = timeout if timeout else 5.0
        t_start = time.monotonic()
        while True:
            with self._latest_frame_lock:
                frame = self._latest_frame
            if frame is not None:
                break
            if time.monotonic() - t_start > deadline_s:
                raise OSError(
                    f"No frame received within {deadline_s}s from "
                    f"IntelRealSenseCamera({self.config.camera_index})"
                )
            time.sleep(0.001)

        # Normalize: cached frame may be a composite_frame (frameset) for
        # multi-stream pipelines, or a single video_frame for color-only.
        if hasattr(frame, 'get_color_frame'):
            # Composite frameset path.
            if self.config.align_depth:
                frame = self.align.process(frame)
            color_frame = frame.get_color_frame()
        else:
            # Single-stream pipeline: the frame itself is the color frame.
            color_frame = frame
        if not color_frame:
            raise OSError(
                f"Can't capture color image from IntelRealSenseCamera({self.camera_index})."
            )
        color_image = np.asanyarray(color_frame.get_data())
        # IntelRealSense uses RGB format as default (red, green, blue).
        if self.config.color_mode == "bgr":
            color_image = color_image[..., ::-1]  # Convert RGB to BGR

        outputs = {"color/image_raw": color_image}

        if self.config.enable_depth:
            depth_frame = frame.get_depth_frame()
            if not depth_frame:
                raise OSError(
                    f"Can't capture depth image from IntelRealSenseCamera({self.config.camera_index})."
                )
            depth_map = np.asanyarray(depth_frame.get_data())
            if self.config.align_depth:
                key = "aligned_depth_to_color/image_raw"
            else:
                key = "depth/image_rect_raw"
            outputs[key] = depth_map
        return outputs

    def get_info(self):
        """Get the camera information."""
        info = self.config.model_dump()
        info["serial_number"] = info.pop("camera_index")
        return info

    def shutdown(self) -> bool:
        self._rs_pipe.stop()
        self._rs_pipe = None
        return True


if __name__ == "__main__":
    print(find_camera_indices())
    print(find_camera_device_ids())
