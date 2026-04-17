# src/openpi/policies/ur10_policy.py
import dataclasses
import numpy as np
import openpi.models.model as _model
import openpi.transforms as transforms


def _parse_image(img):
    """LeRobot stores images as float32 (C,H,W); convert to uint8 (H,W,C)."""
    if isinstance(img, np.ndarray) and img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))
    return img


@dataclasses.dataclass(frozen=True)
class UR10Inputs(transforms.DataTransformFn):
    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        state = np.concatenate([data["joints"], data["gripper"]])

        base_image = _parse_image(data["top_rgb"])       # D415 top cam
        wrist_image = _parse_image(data["wrist_rgb"])    # D435i wrist cam

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # π0.5 (PI05) uses the right-wrist slot too — set True
                # so the model doesn't mask it out entirely
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI05 else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UR10Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # 7 dims: 6 joints + 1 gripper
        return {"actions": np.asarray(data["actions"][:, :7])}