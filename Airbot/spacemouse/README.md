# Airbot/spacemouse/ — SpaceMouse 驱动 + 真机测试脚本

三个测试脚本，按复杂度递进：

```
test_spacemouse.py              只测 SpaceMouse 硬件
       ↓ 加真机
test_spacemouse_robot.py        SpaceMouse 遥操 1 或 N 条机械臂
       ↓ 加 π0 policy
test_inference_with_spacemouse.py   π0 推理 + 按键切换人机介入
```

所有脚本的共用特性：
- **单/双臂通过 `--ports` / `--robot-config.robot_ports` 列表自动识别**（给 1 个端口 = 单臂，给 2 个 = 双臂）
- **SpaceMouse 按键**：左键 = 切换介入/自主，右键 = 切换夹爪开合（介入状态下）
- **SpaceMouse 摇杆**：控制末端 6DOF 位姿（介入状态下）

---

## 1. `test_spacemouse.py` — 纯硬件测试

**用途**：不连机器人，只验证 SpaceMouse 本身能连、按键能读、6DOF 能动。

```bash
# 单 SpaceMouse
python Airbot/spacemouse/test_spacemouse.py

# 双 SpaceMouse（双臂 bimanual）
python Airbot/spacemouse/test_spacemouse.py --dual

# 双 SpaceMouse + 自动校准（首次使用推荐）
python Airbot/spacemouse/test_spacemouse.py --dual --auto-calibrate

# 全 mock（完全不需要硬件，CI 用）
python Airbot/spacemouse/test_spacemouse.py --mock
```

**会打印**：连接的 SpaceMouse 数量、每次按键事件、6DOF 读数。

---

## 2. `test_spacemouse_robot.py` — SpaceMouse 遥操真机

**用途**：纯遥操 —— 没有 policy，SpaceMouse 直接控机械臂。可录像验证运动。

### 基本用法

```bash
# 单臂（连端口 50051）
python Airbot/spacemouse/test_spacemouse_robot.py --ports 50051

# 双臂（默认）
python Airbot/spacemouse/test_spacemouse_robot.py

# 双臂 + 自动校准
python Airbot/spacemouse/test_spacemouse_robot.py --ports 50051 50053 --auto-calibrate

# 只测 SpaceMouse，不连真机
python Airbot/spacemouse/test_spacemouse_robot.py --no-robot
```

### 每臂独立设置 pos/rot 灵敏度

```bash
# 单臂
python Airbot/spacemouse/test_spacemouse_robot.py --ports 50051 --pos-scales 0.3 --rot-scales 0.4

# 双臂各自不同
python Airbot/spacemouse/test_spacemouse_robot.py --ports 50051 50053 --pos-scales 0.5 0.3 --rot-scales 0.5 0.3
```

### 录像

```bash
# USB 相机（device 0 和 2）
python Airbot/spacemouse/test_spacemouse_robot.py --cameras 0 2

# RealSense（按序列号）
python Airbot/spacemouse/test_spacemouse_robot.py --realsense 243222071389

# 输出目录 / 帧率 / 分辨率
python Airbot/spacemouse/test_spacemouse_robot.py --cameras 0 --video-dir ./my_videos --video-fps 60 --video-width 1280 --video-height 720
```

**录像按键**：介入模式下按 `ENTER` 开始 / 停止录像。

### 自定义 reset 位姿

```bash
# 单臂：7 个值（j1-j6 + gripper）
python Airbot/spacemouse/test_spacemouse_robot.py --ports 50051 \
    --reset-actions -0.001618 -1.036 0.842 -1.615 0.634 1.695 0.0

# 双臂：14 个值（两臂串起来）
python Airbot/spacemouse/test_spacemouse_robot.py --ports 50051 50053 \
    --reset-actions \
        -0.001618 -1.036 0.842 -1.615 0.634 1.695 0.0 \
        -0.001618 -1.036 0.842 -1.615 0.634 1.695 0.0

# 跳过 reset
python Airbot/spacemouse/test_spacemouse_robot.py --no-reset
```

### 所有 CLI 参数

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `--ports` | list[int] | `[50051, 50053]` | gRPC 端口，决定臂数 |
| `--robot-url` | str | `localhost` | |
| `--pos-scale` / `--rot-scale` | float | `0.3` / `0.3` | 共享 SpaceMouse 灵敏度 |
| `--pos-scales` / `--rot-scales` | list[float] | `None` | 每臂独立灵敏度（长度必须 = 臂数）|
| `--auto-calibrate` | flag | off | 自动识别多个 SpaceMouse 对应哪只手臂 |
| `--mock-spacemouse` | flag | off | 使用 mock SpaceMouse |
| `--no-robot` | flag | off | 不连机械臂，只测 SpaceMouse |
| `--reset-actions` | list[float] | built-in | 扁平 list，长度 = 臂数 × 7 |
| `--no-reset` | flag | off | 跳过初始 reset |
| `--cameras` | list[int] | `[]` | USB 相机 device 号 |
| `--realsense` | str | `None` | RealSense 序列号 |
| `--video-dir` | str | `./videos` | 录像输出目录 |
| `--video-fps/width/height` | int | `30 / 640 / 480` | 录像参数 |

---

## 3. `test_inference_with_spacemouse.py` — π0 推理 + 介入

**用途**：跑 π0 policy + SpaceMouse 允许人实时接管纠正。

### 关键特性（两个模式同时存在）

- **AUTONOMOUS**：policy 输出 joint 动作 → `SERVO_JOINT_POS` 下发
- **INTERVENING**：SpaceMouse 输出 pose 增量 → `SERVO_CART_POSE` 直接 gRPC
- 左键切换。每条臂独立切换。
- Reset 始终走 `PLANNING_POS`（joint 空间，规划平滑）
- **z-min 安全保护**：所有臂、策略输出和人工介入都受保护。默认值来自 `Airbot/safe_arm.py` 的 `DEFAULT_Z_MIN_PER_ARM`（per-arm，机器物理地板）；CLI 可用 `--z-mins 0.005` 单值广播或 `--z-mins 0.005 0.003` 双臂分别设

### 单臂推理

```bash
cd /path/to/openpi

uv run /path/to/Airbot-VLA-RL/Airbot/spacemouse/test_inference_with_spacemouse.py \
    policy-config:local-policy-config \
    --policy-config.config-path data/pick_and_place \
    --policy-config.checkpoint-dir "/path/to/checkpoint/19999" \
    --robot-config.robot_ports 50051 \
    --robot-config.camera-index 243322074422 243522071794 \
    --reset-action -0.001618 -1.036 0.842 -1.615 0.634 1.695 0.0
```

### 双臂推理

```bash
uv run /path/to/Airbot-VLA-RL/Airbot/spacemouse/test_inference_with_spacemouse.py \
    policy-config:local-policy-config \
    --policy-config.config-path data/fold_towel \
    --policy-config.checkpoint-dir "/path/to/checkpoint/19999" \
    --robot-config.robot_ports 50051 50053 \
    --robot-config.camera-index 243222074218 243522071794 243222071389 \
    --step-rate 20 \
    --z-mins 0.002 \
    --auto-calibrate \
    --reset-action \
        -0.001618 -1.036 0.842 -1.615 0.634 1.695 0.0 \
        -0.001618 -1.036 0.842 -1.615 0.634 1.695 0.0
```

### 每臂独立 z-min

```bash
# 单值（广播到所有臂）
--z-mins 0.002

# 双臂各自设
--z-mins 0.001 0.0012
```

### 关键 CLI 参数

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `policy-config:local-policy-config` | subcmd | — | 用本地 checkpoint（`:remote-policy-config` 用远程 WebSocket）|
| `--policy-config.config-path` | str | — | 任务配置路径（如 `data/pick_and_place`）|
| `--policy-config.checkpoint-dir` | str | — | 训好的 checkpoint 目录 |
| `--robot-config.robot_ports` | list[int] | `[50051, 50053]` | 机械臂端口，**决定臂数** |
| `--robot-config.camera-index` | list[int \| str] | — | 相机序号（int）或 RealSense 序列号（str）|
| `--reset-action` | list[float] | built-in (7D) | joint 形式 reset，长度 = 臂数 × 7 |
| `--step-rate` | int | `20` | 控制频率（Hz） |
| `--step-length` | list[float] | 7 值自动 tile | 插值步长（为单臂尺寸，双臂时自动扩展） |
| `--interpolate` | flag | off | policy 动作是否做插值平滑 |
| `--chunk-size-execute` | int | `25` | policy chunk 执行长度 |
| `--spacemouse-pos-scale` / `--spacemouse-rot-scale` | float | `0.3` / `0.3` | SpaceMouse 位姿增量缩放 |
| `--auto-calibrate` | flag | off | 自动识别 SpaceMouse 对应手臂 |
| `--z-mins` | list[float] | `safe_arm.DEFAULT_Z_MIN_PER_ARM` | 每臂 z 下限（米，base 坐标）；不传按 arm name 查 dict |
| `--eef-type` | str | `G2` | 末端类型（用于 FK/IK 计算） |
| `--max-steps` | int | `250000` | 单次 episode 最大步数 |

### 安全机制

z-min 在两个通道上都生效：

1. **Policy 输出**：joint 动作经 FK 算出末端 xyz，若 z < z_min → 用 IK 解一个 z=z_min 的新 joint（保持 x/y/rot 不变）
2. **SpaceMouse 介入**：pose 动作的 z 坐标被裁到 z_min

第一次触发会 log warning，之后每 100 次一次。

---

## 常见坑

### SpaceMouse 连错手臂
双 SpaceMouse 时可能连反了（左 SpaceMouse 实际控制右臂）。加 `--auto-calibrate`，按提示动左 SpaceMouse、再动右 SpaceMouse，脚本自动识别绑定。

### π0 推理 import 失败
`test_inference_with_spacemouse.py` 必须在 openpi 工程根目录用 `uv run` 跑：
```bash
cd /path/to/openpi && uv run /path/to/Airbot/spacemouse/test_inference_with_spacemouse.py ...
```
直接 `python` 会因为 openpi 依赖找不到而报错。

### `reset-action` 长度不匹配
新脚本统一要求 `len(reset_action) == num_arms * 7`。单臂 7 值、双臂 14 值。传错了会 assert 退出并告诉你期望几维。

### Ports 和 cameras 数量
`--robot-config.robot_ports` 决定臂数。`--robot-config.camera-index` 给几个相机都可以，常见搭配：
- 单臂：`env_cam wrist_cam`（2 个）
- 双臂：`env_cam left_wrist right_wrist`（3 个）
