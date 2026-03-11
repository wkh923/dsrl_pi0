"""Airbot-specific transforms for pi0 policy input/output.

Copied from VLA-RL/airbot-pi0/openpi/examples/airbot/airbot_policy.py
to work with dsrl_pi0's DSRL-modified openpi.
"""
import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return image


@dataclasses.dataclass(frozen=True)
class AirbotInputs(transforms.DataTransformFn):
    """Convert inputs to the format expected by the pi0 model.
    Used for both training and inference.
    """
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # We only mask padding for pi0 model, not pi0-FAST.
        mask_padding = self.model_type == _model.ModelType.PI0

        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)

        image_dict = {}
        image_mask_dict = {}
        image_name = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
        for name in image_name:
            if "observation/" + name in data:
                image_dict[name] = _parse_image(data["observation/" + name])
                image_mask_dict[name] = np.True_
        for name in image_name:
            if name not in image_dict:
                image_dict[name] = np.zeros_like(image_dict[next(iter(image_dict.keys()))])
                image_mask_dict[name] = np.False_ if mask_padding else np.True_
        inputs = {
            "state": state,
            "image": image_dict,
            "image_mask": image_mask_dict,
        }
        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AirbotOutputs(transforms.DataTransformFn):
    """Convert outputs from the model back to dataset-specific format.
    Used for inference only.
    """
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])}
