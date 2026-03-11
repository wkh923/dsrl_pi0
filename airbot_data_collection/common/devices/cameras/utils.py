import platform
from enum import Enum
from typing import List, Dict, Tuple, Optional, Union, Literal
from pydantic import BaseModel, NonNegativeInt, PositiveInt, field_validator
from collections import defaultdict


class CameraRGBConfig(BaseModel):
    camera_index: Optional[Union[int, str]] = None
    fps: Optional[PositiveInt] = None
    width: Optional[PositiveInt] = None
    height: Optional[PositiveInt] = None
    color_mode: Literal["bgr", "rgb"] = "bgr"
    pixel_format: Optional[Union[str, Enum]] = None


class CameraRGBDConfig(CameraRGBConfig):
    enable_depth: bool = False
    enable_color: bool = True
    align_depth: bool = False

    @field_validator("align_depth", mode="after")
    def check_align_depth(cls, align_depth, values):
        if not values.data.get("enable_depth", False):
            return False
        return align_depth


class RegionOfInterest(BaseModel):
    x_offset: NonNegativeInt = 0
    y_offset: NonNegativeInt = 0
    height: NonNegativeInt = 0
    width: NonNegativeInt = 0
    do_rectify: bool = False


class CameraInfo(BaseModel):
    width: NonNegativeInt
    height: NonNegativeInt
    distortion_model: str = ""
    d: List[float] = []
    k: List[float] = []
    r: List[float] = []
    p: List[float] = []
    binning_x: NonNegativeInt = 0
    binning_y: NonNegativeInt = 0
    roi: RegionOfInterest = RegionOfInterest()

    def model_post_init(self, context):
        assert len(self.k) in {0, 9}, "Camera matrix K must be 3x3"
        assert len(self.r) in {0, 9}, "Camera matrix R must be 3x3"
        assert len(self.p) in {0, 12}, "Camera matrix P must be 3x4"


class CameraControl(BaseModel):
    brightness: NonNegativeInt
    contrast: NonNegativeInt
    saturation: NonNegativeInt
    hue: NonNegativeInt
    white_balance_automatic: bool
    gamma: PositiveInt
    power_line_frequency: int = 1
    white_balance_temperature: PositiveInt
    sharpness: NonNegativeInt
    backlight_compensation: NonNegativeInt
    auto_exposure: int = 3
    exposure_time_absolute: NonNegativeInt
    exposure_dynamic_framerate: bool


def find_video_capture_devices(
    card_key: bool = False, cards: Optional[List[str]] = None
) -> Dict[Tuple[str, str], List[str]]:
    """
    Finds all video capture devices on the system and returns a dictionary
    with device information.
    """
    from linuxpy.video.device import iter_video_capture_devices

    cards = cards or []
    if isinstance(cards, str):
        cards = [cards]
    slices = slice(None, 2) if card_key else 1
    devices = defaultdict(list)
    for dev in iter_video_capture_devices():
        with dev:
            key = (dev.info.card, dev.info.bus_info)
            use = True
            if cards:
                for card in cards:
                    if card in key[0]:
                        break
                else:
                    use = False
            if use:
                devices[key[slices]].append(str(dev.filename))
    return dict(devices)


def find_camera_indices(
    raise_when_empty: bool = False,
    max_index_search_range: int = 10,
    filt_mode: str = "none",
    sorting: bool = True,
) -> list[int]:
    """Finds the available camera (video capture devices) indices on the system.
    # The maximum opencv device index depends on your operating system. For instance,
    # if you have 3 cameras, they should be associated to index 0, 1, and 2. This is the case
    # on MacOS. However, on Ubuntu, the indices are different like 6, 16, 23.
    # When you change the USB port or reboot the computer, the operating system might
    # treat the same cameras as new devices. Thus we select a higher bound to search indices.
    """
    if platform.system() == "Linux":
        possible_camera_ids = [
            int(device[0].removeprefix("/dev/video"))
            for device in find_video_capture_devices().values()
        ]
    else:
        print(
            "Mac or Windows detected. Finding available camera indices through "
            f"scanning all indices from 0 to {max_index_search_range}"
        )
        possible_camera_ids = range(max_index_search_range)

    camera_ids = possible_camera_ids

    if filt_mode in {"even", "odd"}:
        remainder = 4 - len(filt_mode)
        camera_ids = [
            camera_id for camera_id in camera_ids if camera_id % 2 == remainder
        ]
    if sorting:
        camera_ids = sorted(camera_ids)

    if raise_when_empty and len(camera_ids) == 0:
        raise OSError(
            "Not a single camera was detected. Try re-plugging, or re-installing `opencv2`, "
            "or your camera driver, or make sure your camera is compatible with opencv2."
        )

    return camera_ids


def get_video_device_bus_info():
    """Returns a dictionary mapping video device paths to their bus info."""
    from linuxpy.video.device import iter_video_capture_devices

    device_bus_info = {}
    for dev in iter_video_capture_devices():
        with dev:
            device_bus_info[str(dev.filename)] = dev.info.bus_info
    return device_bus_info


def get_camera_index_by_bus_info(
    bus_info: str, sorting: bool = False, allow_empty: bool = False
) -> List[str]:
    """
    Get the camera index by its bus info.
    Args:
        bus_info: The bus info of the camera.
        sorting: Whether to sort the camera indices.
    Return: The camera indices that match the bus info.
    """
    devices = find_video_capture_devices(False).get(bus_info, [])
    assert allow_empty or devices, f"No camera indexes found with bus info: {bus_info}"
    if sorting:
        devices = sorted(devices)
    return devices


if __name__ == "__main__":
    print("RealSense cameras:", find_video_capture_devices(False, "RealSense"))
    # print("LRCP cameras:", find_video_capture_devices(False, "LRCP"))
    # print("Webcam cameras:", find_video_capture_devices(False, "Webcam"))
    # print("cam:", find_video_capture_devices(False, "cam"))
    # print("All cameras:", find_video_capture_devices(False, ""))
    # print(get_video_device_bus_info())
