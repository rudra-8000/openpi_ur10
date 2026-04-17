# src/openpi/policies/ur10_policy.py
import dataclasses
import numpy as np

import openpi.models.model as _model
import openpi.transforms as transforms


def _parse_image(img):
    """LeRobot stores images as float32 (C,H,W); openpi expects uint8 (H,W,C)."""
    if isinstance(img, np.ndarray) and img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))
    return img


@dataclasses.dataclass(frozen=True)
class UR10Inputs(transforms.DataTransformFn):
    # action_dim is required by the framework when called from DataConfig
    action_dim: int = 7
    model_type: _model.ModelType = _model.ModelType.PI0_FAST  # pi0.5 = PI0_FAST

    def __call__(self, data: dict) -> dict:
        # observation.state is already a flat (7,) vector: [j0..j5, gripper]
        state = np.asarray(data["observation.state"], dtype=np.float32)

        # Images come in after repack — keys are "top_rgb" and "wrist_rgb"
        top_image   = _parse_image(data["observation.images.cam_high"])   # D415 top camera
        wrist_image = _parse_image(data["observation.images.cam_right_wrist"]) # D435i wrist camera

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb":        top_image,
                "left_wrist_0_rgb":  wrist_image,
                # No right wrist on UR10 — fill the slot with zeros
                "right_wrist_0_rgb": np.zeros_like(top_image),
            },
            "image_mask": {
                "base_0_rgb":       np.True_,
                "left_wrist_0_rgb": np.True_,
                # pi0.5 (PI0_FAST) attends to all 3 image slots; set True so
                # the model doesn't ignore it. For pi0 base set False instead.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        if "prompt" not in data or not data["prompt"]:
            inputs["prompt"] = "pick up the object and place it in the box"

        return inputs


@dataclasses.dataclass(frozen=True)
class UR10Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # 7 action dims: joint_0..joint_5 + gripper
        return {"actions": np.asarray(data["actions"][:, :7], dtype=np.float32)}