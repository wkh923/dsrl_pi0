# Airbot/ — 硬件层（厂家 SDK + 自定义封装）

仓库的硬件接口层。所有跟 AIRBOT 机械臂、外接输入设备、相机交互的代码都在这。

```
Airbot/
├── airbot_py/              ← 厂家 Python SDK（airbot_py-X.Y.Z 解包内容）
├── airbot_proto/           ← 厂家 protobuf 接口（gRPC stubs）
├── airbot_kdl/             ← 厂家正逆运动学求解器
├── airbot_server_quiet/    ← 静音版 airbot_server 启动脚本（过滤已知噪音日志）
├── safe_arm.py             ★ 安全包装：clamp Z 防撞桌（透传其他方法）
├── realsense.py            ★ RealSense 相机捕获（pyrealsense2 包装）
├── spacemouse/             ★ SpaceMouse 驱动 + 测试脚本
│   ├── __init__.py             重导出公开 API（from spacemouse import ...）
│   ├── spacemouse.py           驱动（单/双臂 + 自动校准 + Mock）
│   ├── test_spacemouse.py            纯 SpaceMouse 硬件测试
│   ├── test_spacemouse_robot.py      SpaceMouse 遥操（1-N 臂）
│   ├── test_inference_with_spacemouse.py   π0 推理 + 介入
│   └── README.md
├── vr/                        Meta Quest3 VR 手柄方案
│   ├── scripts/                Python 端：ROS2 订阅 + DualVRInput 处理
│   └── unity_tcp_ws/           Unity 端：APK + ROS-TCP-Endpoint
├── SDK_API_Reference.md       AIRBOTArm SDK 方法速查（基于 airbot_py 5.0.0 源码生成）
└── README.md
```

★ = 我们写的（非厂家代码）

---

## 谁在用这里的东西

所有调用方都通过 `sys.path.insert(... / "Airbot")` 进来：

### `safe_arm.py` → `SafeAIRBOTArm` / `clamp_cart_pose` / `clamp_joint_pos`
- `Airbot/spacemouse/test_inference_with_spacemouse.py`
- `Airbot/spacemouse/test_spacemouse_robot.py`
- `VLA/airbot-data/data_collection/airbot/teleoprators/spacemouse_dual_arm.py`
- `VLA/airbot-data/data_collection/airbot/teleoprators/vr_dual_arm.py`
- `VLA/airbot-pi0/airbot/core/play_operator.py`（VLA 推理 Robot 类，joint 空间 FK+IK）
- `SERL/serl_robot_infra/airbot_env/devices/safe_arm.py`（re-export shim）

### `spacemouse/` (driver + 3 tests) → `AirbotSpaceMouse / DualAirbotSpaceMouse / create_*`
- VLA 数据采集 SpaceMouse 双臂遥操器（`spacemouse_dual_arm.py`）
- 包内 3 个 `test_*.py` 测试脚本
- `SERL/serl_robot_infra/airbot_env/devices/spacemouse.py`（re-export shim）

### `realsense.py` → `RSCapture`
- `SERL/serl_robot_infra/airbot_env/camera/rs_capture.py`（re-export shim）

### `vr/`
- `vr/scripts/test_vr_dual_robot.py` — VR 双臂遥操（连真机）
- `vr/unity_tcp_ws/` — Quest3 ↔ Ubuntu 的 ROS2 桥接（Unity 项目）
- 上层调用方：`data_collection.airbot.teleoprators.vr_dual_arm`

---

## 跟厂家 SDK 的关系

**`airbot_py/`**、**`airbot_proto/`**、**`airbot_kdl/`** 是从厂家 wheel 包里解出来的源码。**真正的 import 解析靠 pip 安装的 `airbot_py-X.Y.Z.whl`** —— 这里只是源码参考（IDE 跳转定义会进到这里）。

升级厂家 SDK 时：
```bash
# 1. pip 安装新 wheel（实际生效的安装路径）
pip install airbot_py-X.Y.Z-py3-none-any.whl

# 2. 替换 Airbot/airbot_py/ 等内容（保持 IDE 跳转可用）
cd Airbot
unzip -o airbot_py-X.Y.Z-py3-none-any.whl -d _tmp
rm -rf airbot_py airbot_proto airbot_kdl
mv _tmp/airbot_py _tmp/airbot_proto _tmp/airbot_kdl ./
rm -rf _tmp

# 3. 把 airbot_kdl/{mmk2_kdl,mmk2_kdl_ops}.py 重命名回 dual_arm_kdl{,_ops}.py
#    并改类名 MMK2Kdl → DualArmKdl、MMK2KdlNumerical → DualArmKdlNumerical
#    （MMK2 是另一型号机器人，跟 AIRBOT Play 单臂无关；本仓库实际不调用它）
git mv airbot_kdl/mmk2_kdl.py airbot_kdl/dual_arm_kdl.py
git mv airbot_kdl/mmk2_kdl_ops.py airbot_kdl/dual_arm_kdl_ops.py
# 然后批量替换 MMK2Kdl → DualArmKdl、MMK2KdlNumerical → DualArmKdlNumerical、mmk2 → robot
```

---

## 为什么这层要单独存在

4 个子工程（`SERL/`、`VLA/airbot-pi0/`、`VLA/airbot-data/`、`Reward-Model-MVP/`）都需要跟 AIRBOT 硬件 / 输入设备 / 相机说话。把这些底层抽象集中到一个目录的好处：

1. **单一来源**：safe_arm 的 Z 钳位逻辑只在一处定义，不会几个项目各搞一套
2. **方便升级**：换 SpaceMouse 库 / 改 z_min 默认值 / 升级 SDK 时，只动 `Airbot/`
3. **依赖清晰**：每个子工程的 `sys.path.insert(0, .../"Airbot")` 一目了然
4. **冷启动新人**：进仓库先看 `Airbot/README.md` 就知道硬件层是啥
