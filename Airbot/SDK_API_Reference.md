# AIRBOT Python SDK — API 速查

> 对应 `airbot_py == 5.0.0`（本仓库 `Airbot/airbot_py/` 解包内容）。
> 升级 wheel 后请重新校对此文件。
>
> 源文件入口：
> - [airbot_py/arm.py](airbot_py/arm.py) — `AIRBOTArm` 主类 + `RobotMode` / `State` / `SpeedProfile` 枚举
> - [airbot_py/__init__.py](airbot_py/__init__.py) — 导出 `RealtimeControllerBase` / `MITController` / `ServoController`
>
> 安全包装：[safe_arm.py](safe_arm.py) 的 `SafeAIRBOTArm` 把 `servo_cart_pose` clamp 到 `z >= z_min`；其他方法 `__getattr__` 透传。

---

## 模块速览

```python
from airbot_py.arm import AIRBOTArm, RobotMode, SpeedProfile, State
```

`AIRBOTArm` 本质是一个**很薄的 gRPC 客户端**，跟驱动服务（driver server，默认 `localhost:50051`）通信。

`AIRBOTPlay`、`AIRBOTPlayLite`、`AIRBOTPro` 都是 `AIRBOTArm` 的别名（同一个类）。

---

## 连接生命周期

```python
# 推荐写法（自动 connect / disconnect）：
with AIRBOTArm() as robot:
    pos = robot.get_joint_pos()
    ...

# 等价写法：
robot = AIRBOTArm(url="localhost", port=50051)
robot.connect()
...
robot.disconnect()
```

| 方法 | 说明 |
|---|---|
| `__init__(url='localhost', port=50051)` | 仅记录端点，**不连接**、不发任何指令 |
| `connect() -> bool` | 启动 gRPC 通道 + 后台 feedback 线程；幂等 |
| `disconnect() -> bool` | 停 servo / MIT controller，关 channel；幂等 |
| `__enter__ / __exit__` | `with` 上下文管理器，推荐用法 |

---

## 状态读取

> 反馈通过后台线程持续推送。**首次调用前**若服务还没把状态推过来，返回 `None`。

| 方法 | 返回 | 说明 |
|---|---|---|
| `get_joint_pos()` | `list[float] \| None` | 6 个关节位置（rad）；不含末端 |
| `get_joint_vel()` | `list[float] \| None` | 6 个关节速度（rad/s） |
| `get_joint_eff()` | `list[float] \| None` | 6 个关节力矩（Nm） |
| `get_eef_pos()` | `list[float] \| None` | 末端夹爪宽度（m）；没装末端返回 `[]` |
| `get_eef_eff()` | `list[float] \| None` | 末端夹爪力矩（Nm）；没装末端返回 `[]` |
| `get_end_pose()` | `list[list[float]] \| None` | `[[x,y,z], [qx,qy,qz,qw]]`（m + 四元数） |

末端坐标系：原点为 joint1 旋转轴和安装平面交点；x 前、y 左、z 上。装了末端时返回的是 effecting point（如 G2 夹爪是夹取点）。

---

## 模式切换

```python
robot.switch_mode(RobotMode.SERVO_CART_POSE)
```

| 方法 | 返回 |
|---|---|
| `get_state() -> State` | driver 服务状态 |
| `get_control_mode() -> RobotMode \| None` | 当前控制模式 |
| `switch_mode(mode: RobotMode) -> bool` | 切换控制模式；`INACTIVE` / `UNDEFINED` 会被拒绝 |

### `RobotMode`

| 模式 | 值 | 含义 |
|---|---|---|
| `PLANNING_POS` | 10 | 规划模式：单点目标，server 自己规划路径 |
| `PLANNING_WAYPOINTS_PATH` | 11 | 笛卡尔路点线性插值，不停顿 |
| `PLANNING_WAYPOINTS` | 12 | 路点局部规划，路点附近会减速 |
| `SERVO_JOINT_POS` | 20 | 关节位置 servo（连续命令） |
| `SERVO_CART_POSE` | 21 | 笛卡尔位姿 servo |
| `SERVO_JOINT_VEL` | 24 | 关节速度 servo |
| `SERVO_CART_TWIST` | 25 | 笛卡尔速度 servo（线速度 + 角速度） |
| `MIT_INTEGRATED` | 80 | MIT 集成关节控制（pos/vel/eff/kp/kd） |
| `GRAVITY_COMP` | 90 | 重力补偿 / free-drive，**机械臂不响应命令、可手动拖动** |
| `INACTIVE` | 98 | 不可切到 |
| `UNDEFINED` | 99 | 不可切到 |

### `State`

| 状态 | 值 | 含义 |
|---|---|---|
| `INIT` | 0 | 初始化中 |
| `SHUTDOWN` | 1 | 关闭中 |
| `POWERON` | 2 | 上电自检 |
| `IDLE` | 3 | 空闲，等 SDK 命令 |
| `APPLOADING` | 4 | 加载 app 中 |
| `APPLOADED` | 5 | app 已加载（rw app 加载时 SDK 进入 read-only） |
| `ERROR` | 6 | 错误状态 |

---

## 规划模式（单点 / 路点）

> 都需要 `RobotMode.PLANNING_POS` 或 `PLANNING_WAYPOINTS*`，`blocking=True` 默认会阻塞到 server 报 `FINISHED`。

| 方法 | 模式 | 说明 |
|---|---|---|
| `move_to_joint_pos(joint_pos, blocking=True) -> bool` | `PLANNING_POS` | 6 个关节角（rad） |
| `move_to_cart_pose(cart_pose, blocking=True) -> bool` | `PLANNING_POS` | `[[x,y,z], [qx,qy,qz,qw]]` |
| `move_eef_pos(joint_pos, blocking=True) -> bool` | `PLANNING_POS` | 末端夹爪宽度（m） |
| `move_eff_pos(...)` | — | ⚠️ `move_eef_pos` 的拼写错误版别名，仅向后兼容 |
| `move_with_joint_waypoints(waypoints, blocking=True) -> bool` | `PLANNING_WAYPOINTS*` | 多组关节角顺次执行 |
| `move_with_cart_waypoints(waypoints, blocking=True) -> bool` | `PLANNING_WAYPOINTS*` | 多组笛卡尔位姿顺次执行 |

```python
robot.switch_mode(RobotMode.PLANNING_POS)
robot.move_to_cart_pose([[0.3, 0.0, 0.2], [0, 0, 0, 1]])

robot.switch_mode(RobotMode.PLANNING_WAYPOINTS)
robot.move_with_cart_waypoints([
    [[0.3, 0.0, 0.20], [0,0,0,1]],
    [[0.3, 0.0, 0.05], [0,0,0,1]],
])
```

---

## Servo 模式（连续命令）

> 需要先 `switch_mode(...)` 到对应的 `SERVO_*`。每次调用都把命令塞进后台 ServoController 的队列。

| 方法 | 模式 | 命令 |
|---|---|---|
| `servo_joint_pos(joint_pos)` | `SERVO_JOINT_POS` | 6 个关节角（rad） |
| `servo_joint_vel(joint_vel)` | `SERVO_JOINT_VEL` | 6 个关节速度（rad/s） |
| `servo_cart_pose(cart_pose)` | `SERVO_CART_POSE` | `[[x,y,z], [qx,qy,qz,qw]]` |
| `servo_cart_twist(cart_twist)` | `SERVO_CART_TWIST` | `[[vx,vy,vz], [wx,wy,wz]]` |
| `servo_eef_pos(pos)` | 任意 `SERVO_*` | 夹爪宽度（m），`float` 或 `[float]` |
| `servo_eef_force(force)` | — | ⚠️ **5.0.0 未实现**，5.3 版本会换新接口 |

```python
robot.switch_mode(RobotMode.SERVO_CART_POSE)
while running:
    robot.servo_cart_pose([[x, y, z], [qx, qy, qz, qw]])
    time.sleep(1/50)
```

---

## MIT 集成模式

```python
robot.switch_mode(RobotMode.MIT_INTEGRATED)
robot.mit_joint_integrated_control(
    joint_pos=[...],   # 6 个 rad
    joint_vel=[...],   # 6 个 rad/s
    joint_eff=[...],   # 6 个 Nm
    joint_kp=[...],    # 6 个 P 增益
    joint_kd=[...],    # 6 个 D 增益
)
```

5 个参数都必须长度 6，否则方法返回 `False`。背后是 `MITController` 在 ~250 Hz 推送命令。

---

## 参数 / App / 速度档

```python
robot.set_speed_profile(SpeedProfile.SLOW)         # 一键调慢
robot.set_params({"sdk_server.max_velocity_scaling_factor": 0.3})
val = robot.get_params(["sdk_server.max_velocity_scaling_factor"])
```

| 方法 | 返回 | 说明 |
|---|---|---|
| `get_params(names: list[str]) -> dict` | `dict` | 批量读 |
| `set_params(params: dict) -> dict` | `dict` | 批量写；只支持 `bool/int/float/str` |
| `set_speed_profile(profile: SpeedProfile)` | — | 一键设置 servo 缩放 + 规划速度 |
| `load_app(name, params=None) -> bool` | `bool` | 加载 app（如 `record_replay_app/...:RecordReplayApp`） |
| `unload_app() -> bool` | `bool` | 卸载 app |
| `get_product_info() -> dict` | `dict` | 产品信息：`product_type / sn / is_sim / interfaces / arm_types / eef_types / fw_versions` |

### `SpeedProfile`

| 档位 | servo 线速度缩放 | servo 关节缩放 | velocity_scaling | acc_scaling |
|---|---|---|---|---|
| `DEFAULT` | 0.2 | 0.1 | 0.5 | 0.1 |
| `SLOW` | 0.05 | 0.05 | 0.1 | 0.02 |
| `FAST` ⚠️ | 10.0 | 1.0 | 1.0 | 0.5 |

`FAST` 是实验性的，可能让机械臂跑得过快撞坏自己 / 环境，**用前确认安全**。

---

## SafeAIRBOTArm 包装

[safe_arm.py](safe_arm.py) 提供 `SafeAIRBOTArm`，包在 `AIRBOTArm` 外面拦截 `servo_cart_pose` 强制 `z >= z_min`，防止末端撞桌：

```python
from safe_arm import SafeAIRBOTArm   # Airbot/ 在 sys.path 上

raw = AIRBOTArm("localhost", 50051)
raw.connect()
arm = SafeAIRBOTArm(raw, z_min=0.05)   # 5 cm 离台面
arm.servo_cart_pose([[x, y, 0.001], [0,0,0,1]])  # 自动 clamp 成 z=0.05
arm.get_joint_pos()  # 透传给 raw.get_joint_pos()
```

只拦截 `servo_cart_pose`，其他方法靠 `__getattr__` 透传。

---

## 完整 import 速查

```python
# 主类 + 枚举
from airbot_py.arm import AIRBOTArm, RobotMode, SpeedProfile, State
from airbot_py.arm import AIRBOTPlay, AIRBOTPlayLite, AIRBOTPro  # 别名

# 实时 controller（一般不直接用，AIRBOTArm 内部会管）
from airbot_py import RealtimeControllerBase, MITController, ServoController

# 自定义安全包装
from safe_arm import SafeAIRBOTArm, DEFAULT_Z_MIN
```
