"""Airbot data configuration for loading SFT checkpoints with correct transforms.

Adapted from VLA-RL/airbot-pi0/openpi/examples/airbot/airbot_data_config.py
to work with dsrl_pi0's DSRL-modified openpi (supports noise injection and get_prefix_rep).
"""
import dataclasses
import importlib.util
from pathlib import Path
import sys

from examples.airbot.airbot_policy import AirbotInputs, AirbotOutputs
import flax.nnx as nnx
from typing_extensions import override

import openpi.models.model as _model
import openpi.models.pi0 as pi0
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory, TrainConfig
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms


@dataclasses.dataclass(frozen=True)
class AirbotTaskConfig:
    task_name: str
    robot_type: str  # "1-1" (single-arm) or "ptk" (dual-arm)
    camera_topics: dict[str, str]
    delta_action_mask: tuple[int, ...]
    base_model: str = "pi0"
    using_lora: bool = False
    # Fields below are not used during DSRL but kept for config compatibility
    folders: list[str] = dataclasses.field(default_factory=list)
    state_topics: list[str] = dataclasses.field(default_factory=list)
    action_topics: list[str] = dataclasses.field(default_factory=list)
    fps: int = 20
    exp_name: str = "dsrl"
    batch_size: int = 32
    num_workers: int = 2
    num_train_steps: int = 30_000
    log_interval: int = 100
    save_interval: int = 1000
    keep_period: int = 5000
    over_write: bool = True
    resume: bool = False
    log_method: str = "wandb"
    fsdp_devices: int = 1


@dataclasses.dataclass(frozen=True)
class LeRobotAirbotDataConfig(DataConfigFactory):
    task_config: AirbotTaskConfig = dataclasses.field(default_factory=AirbotTaskConfig)

    @override
    def create(self, assets_dirs: Path, model_config: _model.BaseModelConfig) -> DataConfig:
        config = self.task_config

        repack_transform_dict = {}
        for camera in config.camera_topics:
            repack_transform_dict[f"observation/{camera}"] = f"{camera}"
        repack_transform_dict["observation/state"] = "state"
        repack_transform_dict["actions"] = "actions"
        repack_transform_dict["prompt"] = "prompt"
        repack_transform = _transforms.Group(inputs=[_transforms.RepackTransform(repack_transform_dict)])

        data_transforms = _transforms.Group(
            inputs=[AirbotInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[AirbotOutputs()],
        )

        delta_action_mask = _transforms.make_bool_mask(*config.delta_action_mask)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


def get_task_config(config_path: str) -> AirbotTaskConfig:
    """Load task config from a Python config file (e.g., VLA-RL's config.py)."""
    config_path = Path(config_path)
    if not config_path.is_file():
        config_path = config_path / "config.py"
        if not config_path.is_file():
            raise ValueError(f"Config path {config_path} does not exist or is not a file.")

    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config
    spec.loader.exec_module(config)
    return AirbotTaskConfig(
        task_name=config.TASK_NAME,
        robot_type=config.ROBOT_TYPE,
        folders=getattr(config, "FOLDERS", []),
        state_topics=getattr(config, "STATE_TOPICS", []),
        action_topics=getattr(config, "ACTION_TOPICS", []),
        camera_topics=config.CAMERA_TOPICS,
        fps=getattr(config, "FPS", 20),
        delta_action_mask=tuple(config.DELTA_ACTION_MASK),
        base_model=getattr(config, "BASE_MODEL", "pi0"),
        using_lora=getattr(config, "USING_LORA", False),
        exp_name=getattr(config, "EXP_NAME", "dsrl"),
        batch_size=getattr(config, "BATCH_SIZE", 32),
        num_workers=getattr(config, "NUM_WORKERS", 2),
        num_train_steps=getattr(config, "NUM_TRAIN_STEPS", 30_000),
        log_interval=getattr(config, "LOG_INTERVAL", 100),
        save_interval=getattr(config, "SAVE_INTERVAL", 1000),
        keep_period=getattr(config, "KEEP_PERIOD", 5000),
        over_write=getattr(config, "OVER_WRITE", True),
        resume=getattr(config, "RESUME", False),
        log_method=getattr(config, "LOG_METHOD", "wandb"),
        fsdp_devices=getattr(config, "FSDP_DEVICES", 1),
    )


def get_config(task_config: AirbotTaskConfig) -> TrainConfig:
    """Create a TrainConfig for loading an airbot SFT checkpoint."""
    if task_config.base_model == "pi0":
        if task_config.using_lora:
            model = pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
        else:
            model = pi0.Pi0Config()
    else:
        raise ValueError(f"Unknown BASE_MODEL: {task_config.base_model}. Currently only 'pi0' is supported.")

    return TrainConfig(
        name=f"{task_config.robot_type}_{task_config.task_name}",
        exp_name=task_config.exp_name,
        overwrite=task_config.over_write,
        model=model,
        data=LeRobotAirbotDataConfig(
            repo_id=task_config.task_name,
            base_config=DataConfig(prompt_from_task=True),
            task_config=task_config,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        batch_size=task_config.batch_size,
        num_workers=task_config.num_workers,
        num_train_steps=task_config.num_train_steps,
        log_interval=task_config.log_interval,
        save_interval=task_config.save_interval,
        keep_period=task_config.keep_period,
        resume=task_config.resume,
        fsdp_devices=task_config.fsdp_devices,
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter()
        if task_config.using_lora
        else nnx.Nothing(),
        ema_decay=None if task_config.using_lora else 0.99,
    )
