from airbot_data_collection.common.devices.cameras.intelrealsense import (
    IntelRealSenseCamera,
    IntelRealSenseCameraConfig,
)
from time import time_ns
from typing import Dict, Union, Optional
from numpy import ndarray


class RealSense(IntelRealSenseCamera):
    config: IntelRealSenseCameraConfig

    def capture_observation(self, timeout: Optional[float] = None):
        obs = {}
        output = super().capture_observation(timeout)
        for key, value in output.items():
            obs[key] = self._get_value(value)
        return obs

    def _get_value(self, data) -> Dict[str, Union[int, ndarray]]:
        return {
            "t": time_ns(),
            "data": data,
        }
