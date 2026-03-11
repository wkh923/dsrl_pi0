import asyncio
import logging
import math
import os
import threading
import time
import numpy as np
import subprocess
import sys
from enum import Enum
from typing import List, Optional
from functools import partial


def get_stamp_ms() -> int:
    return int(time.time() * 1e3)


class bcolors:
    MAGENTA = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def find_matching_files(
    search_dirs: tuple[str, ...],
    filenames: tuple[str, ...],
    end_with: tuple[str, ...] = (".yaml", ".yml"),
    strict: bool = False,
    ignore_path: bool = False,
    ignore_empty: bool = True,
) -> list[Optional[str]]:
    result: list[Optional[str]] = []
    search_dirs = [os.path.abspath(dir) for dir in search_dirs]
    for name in filenames:
        if ignore_empty and not name:
            result.append(name)
            continue
        elif ignore_path and "/" in name:
            assert os.path.exists(name), f"File {os.path.abspath(name)} does not exist."
            result.append(name)
            continue
        target_base = os.path.splitext(name)[0]
        found_path = None
        for search_dir in search_dirs:
            for root, _, files in os.walk(search_dir):
                for file in files:
                    if file.endswith(end_with):
                        file_base = os.path.splitext(file)[0]
                        if file_base == target_base:
                            found_path = os.path.abspath(os.path.join(root, file))
                            break
                if found_path:
                    break
            if found_path:
                break
        else:
            if strict:
                raise FileNotFoundError(
                    f"File {name} not found in searching directories: {search_dirs}"
                )
        result.append(found_path)  # None if not found
    return result


class ReprEnum(Enum):
    """
    Only changes the repr(), leaving str() and format() to the mixed-in type.
    """


class StrEnum(str, ReprEnum):
    """
    Enum where members are also (and must be) strings
    """

    def __new__(cls, *values):
        "values must already be of type `str`"
        if len(values) > 3:
            raise TypeError(f"too many arguments for str(): {values!r}")
        if len(values) == 1:
            # it must be a string
            if not isinstance(values[0], str):
                raise TypeError(f"{values[0]!r} is not a string")
        if len(values) >= 2:
            # check that encoding argument is a string
            if not isinstance(values[1], str):
                raise TypeError(f"encoding must be a string, not {values[1]!r}")
        if len(values) == 3:
            # check that errors argument is a string
            if not isinstance(values[2], str):
                raise TypeError("errors must be a string, not %r" % (values[2]))
        value = str(*values)
        member = str.__new__(cls, value)
        member._value_ = value
        return member

    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Return the lower-cased version of the member name.
        """
        return name.lower()

    def __str__(self):
        return self.value


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = (
        "[%(levelname)s] %(asctime)s %(name)s: %(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def init_logging(level=logging.INFO):
    logging.basicConfig(level=level)
    ch = logging.StreamHandler()
    # ch.setLevel(level)
    ch.setFormatter(CustomFormatter())
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.root.addHandler(ch)


def run_event_loop() -> asyncio.AbstractEventLoop:
    assert threading.current_thread() == threading.main_thread(), (
        "Event loop must be run in the main thread"
    )
    event_loop = asyncio.get_event_loop()
    if not event_loop.is_running():
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
        threading.Thread(target=event_loop.run_forever, daemon=True).start()
    return event_loop


def get_items_by_ext(
    directory: str, extension: str, with_directory: bool = False
) -> List[str]:
    """Get all files or directories in a directory with a specific extension (suffix).
    Args:
        directory (str): The directory to search in.
        extension (str): The file extension to filter by. If empty, return directories.
            If extension is ".", return all files.
    Returns:
        List[str]: A list of file or directory names that match the extension.
    """
    if not os.path.exists(directory):
        return []
    entries = os.scandir(directory)
    if with_directory:
        prefix = directory.removesuffix("/") + "/"
    else:
        prefix = ""
    if extension == ".":
        return [prefix + entry.name for entry in entries if entry.is_file()]
    elif not extension:
        return [entry.name for entry in entries if entry.is_dir()]
    else:
        if not extension.startswith("."):
            extension = "." + extension
        return [
            prefix + entry.name
            for entry in entries
            if entry.name.endswith(extension) and entry.is_file()
        ]


def optimal_grid(
    N: int, screen_width: int, screen_height: int, image_aspect_ratio: float = 1.0
):
    """Determine the best grid layout for a given screen resolution

    Args:
    - N: number of images
    - screen_width: screen width in pixels
    - screen_height: screen height in pixels
    - image_aspect_ratio: image aspect ratio (default 1.0 for square)

    Returns:
    - rows: number of rows
    - cols: number of columns
    """
    ratio_adjustment = (screen_height / screen_width) * image_aspect_ratio
    ideal_rows = math.sqrt(N * ratio_adjustment)
    best_error = float("inf")
    optimal_rows, optimal_cols = 1, N
    for r in range(max(1, int(ideal_rows * 0.7)), int(ideal_rows * 1.3) + 1):
        c = math.ceil(N / r)

        cell_width = screen_width / c
        cell_height = screen_height / r
        cell_aspect_ratio = cell_width / cell_height

        aspect_error = abs(cell_aspect_ratio - image_aspect_ratio) / image_aspect_ratio
        wasted_space = (r * c - N) / N
        total_error = aspect_error + wasted_space

        if total_error < best_error:
            best_error = total_error
            optimal_rows, optimal_cols = r, c

    return optimal_rows, optimal_cols


def get_dpi() -> float:
    import tkinter

    root = tkinter.Tk()
    dpi = root.winfo_fpixels("1i")  # 水平方向的DPI
    root.destroy()
    return dpi


def resolution_to_inches(width: int, height: int) -> tuple[float, float]:
    dpi = get_dpi()
    return width / dpi, height / dpi


class ProgressBar:
    def __init__(self, total: int, desc: str):
        self.total = total
        self.desc = desc
        from tqdm import tqdm
        # from tqdm.asyncio import tqdm

        self.progress_bar = tqdm(
            total=total or self.total, desc=desc or self.desc, unit="step"
        )
        self.progress_bar.clear()

    def update(self, index: int):
        self.progress_bar.n = index
        self.progress_bar.set_postfix(
            {"Percentage": f"{index / self.total * 100:.1f}%"}
        )
        self.progress_bar.refresh()

    def reset(self, total: int = 0, desc: Optional[str] = None):
        self.progress_bar.reset(total=total or self.total)
        self.progress_bar.desc = desc
        self.progress_bar.clear()

    def close(self):
        self.progress_bar.close()


class ImageCoder:
    @staticmethod
    def rgb2yuv(rgb: np.ndarray) -> np.ndarray:
        # The coefficients were taken from OpenCV https://github.com/opencv/opencv
        # I'm not sure if the values should be clipped, in my (limited) testing it looks alright
        #   but don't hesitate to add rgb.clip(0, 1, rgb) & yuv.clip(0, 1, yuv)
        #
        # Input for these functions is a numpy array with shape (height, width, 3)
        # Change '+= 0.5' to '+= 127.5' & '-= 0.5' to '-= 127.5' for values in range [0, 255]

        m = np.array(
            [
                [0.29900, -0.147108, 0.614777],
                [0.58700, -0.288804, -0.514799],
                [0.11400, 0.435912, -0.099978],
            ]
        )
        yuv = np.dot(rgb, m)
        yuv[:, :, 1:] += 127.5
        return yuv

    @staticmethod
    def yuv2rgb(yuv: np.ndarray) -> np.ndarray:
        # The coefficients were taken from OpenCV https://github.com/opencv/opencv
        # I'm not sure if the values should be clipped, in my (limited) testing it looks alright
        #   but don't hesitate to add rgb.clip(0, 1, rgb) & yuv.clip(0, 1, yuv)
        #
        # Input for these functions is a numpy array with shape (height, width, 3)
        # Change '+= 0.5' to '+= 127.5' & '-= 0.5' to '-= 127.5' for values in range [0, 255]

        m = np.array(
            [
                [1.000, 1.000, 1.000],
                [0.000, -0.394, 2.032],
                [1.140, -0.581, 0.000],
            ]
        )
        yuv[:, :, 1:] -= 127.5
        rgb = np.dot(yuv, m)
        return rgb

    @staticmethod
    def yuyv2bgr(data: bytes, width: int, height: int) -> np.ndarray:
        """Convert YUYV image bytes data to BGR array using numpy.
        Args:
            data: bytes of YUYV image
            height: image height
            width: image width
        Returns:
            numpy image array in bgr
        """
        yuyv = np.frombuffer(data, np.uint8).reshape((height, width, 2))
        y = yuyv[:, :, 0].astype(np.float32)
        u = np.zeros((height, width), dtype=np.float32)
        v = np.zeros((height, width), dtype=np.float32)
        u[:, 0::2] = yuyv[:, 0::2, 1]
        u[:, 1::2] = u[:, 0::2]
        v[:, 1::2] = yuyv[:, 1::2, 1]
        v[:, 0::2] = v[:, 1::2]
        r = y + 1.403 * (v - 128)
        g = y - 0.344 * (u - 128) - 0.714 * (v - 128)
        b = y + 1.770 * (u - 128)
        bgr = np.zeros((height, width, 3), dtype=np.uint8)
        bgr[:, :, 0] = np.clip(b, 0, 255).astype(np.uint8)  # B
        bgr[:, :, 1] = np.clip(g, 0, 255).astype(np.uint8)  # G
        bgr[:, :, 2] = np.clip(r, 0, 255).astype(np.uint8)  # R
        return bgr


def execute_shell_script(
    script_path: str,
    args=None,
    env=None,
    timeout=None,
    check=True,
    with_sudo: bool = False,
):
    """
    执行shell脚本并返回执行结果

    参数:
        script_path (str): 脚本文件路径
        args (list): 传递给脚本的参数列表
        env (dict): 自定义环境变量
        timeout (int): 脚本执行超时时间(秒)
        check (bool): 是否在返回非零退出状态时抛出异常

    返回:
        subprocess.CompletedProcess: 包含执行结果的对象
    """
    # 确保脚本存在且可执行
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"脚本文件不存在: {script_path}")

    if not os.access(script_path, os.X_OK):
        # 尝试添加可执行权限
        try:
            os.chmod(script_path, os.stat(script_path).st_mode | 0o111)
            print(f"已为脚本添加可执行权限: {script_path}")
        except OSError as e:
            raise PermissionError(f"脚本不可执行且无法添加权限: {e}")

    # 构建命令
    cmd = [os.path.abspath(script_path)]
    if with_sudo:
        cmd.insert(0, "sudo")
    if args:
        cmd.extend(args)

    # 合并环境变量
    new_env = os.environ.copy()
    if env:
        new_env.update(env)

    print(f"执行命令: {' '.join(cmd)}")

    try:
        # 执行脚本
        result = subprocess.run(
            cmd,
            env=new_env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=check,
        )
        return result
    except subprocess.TimeoutExpired as e:
        print(f"脚本执行超时: {e}")
        # 返回部分结果
        return subprocess.CompletedProcess(
            args=e.cmd, returncode=-1, stdout=e.stdout, stderr=e.stderr
        )
    except subprocess.CalledProcessError as e:
        print(f"脚本执行失败，返回代码: {e.returncode}")
        print(f"标准输出:\n{e.stdout}")
        print(f"错误输出:\n{e.stderr}")
        raise


def get_can_interfaces():
    try:
        # 执行 ip l 命令
        ip_result = subprocess.run(
            ["ip", "l"], capture_output=True, text=True, check=True
        )

        # 获取输出并按行分割
        output = ip_result.stdout
        lines = output.split("\n")

        # 筛选包含 'can' 的行并提取设备名称
        can_interfaces = []
        for line in lines:
            if "can" in line:
                # 提取设备名称（格式通常为数字: 设备名: <...>）
                parts = line.strip().split(": ")
                if len(parts) > 1:
                    can_interfaces.append(parts[1])

        return can_interfaces

    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {e.stderr}")
        return []
    except Exception as e:
        print(f"发生错误: {e}")
        return []


def linear_map(
    x, raw_range: tuple[float, float], target_range: tuple[float, float]
) -> float:
    a, b = raw_range
    c, d = target_range
    return (x - a) * (d - c) / (b - a) + c


def sort_index(order: list, name_list: list, value_list: list) -> tuple:
    order_dict = {val: idx for idx, val in enumerate(order)}
    sorted_pairs = sorted(zip(value_list, name_list), key=lambda x: order_dict[x[1]])
    sorted_value_list, _ = zip(*sorted_pairs)
    return sorted_value_list


if sys.version_info >= (3, 10):
    zip = partial(zip, strict=True)
else:
    from more_itertools import zip_equal as zip  # noqa: F401


if __name__ == "__main__":
    # bar = ProgressBar(100, "Round 0")
    # for rd in range(10):
    #     for i in range(5):
    #         input("Press Enter to continue...")
    #         bar.update(i + 1)
    #     bar.reset(desc=f"Round {rd + 1}")

    # print(linear_map(-0.1, (0, 1), (0, 100)))

    for item in zip([1, 2], [3, 4, 5]):
        print(item)
