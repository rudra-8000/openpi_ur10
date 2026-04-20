# src/openpi/policies/ur10_policy.py
import dataclasses
import numpy as np

import openpi.models.model as _model
import openpi.transforms as transforms


def _parse_image(img):
    """LeRobot stores images as float32 (C,H,W) tensor or ndarray; openpi expects uint8 (H,W,C) ndarray."""
    import torch
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    # Now it's ndarray — convert float [0,1] CHW → uint8 HWC
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[0] in (1, 3, 4):  # CHW → HWC
        img = np.transpose(img, (1, 2, 0))
    return img


@dataclasses.dataclass(frozen=True)
class UR10Inputs(transforms.DataTransformFn):
    # action_dim is required by the framework when called from DataConfig
    action_dim: int = 7
    model_type: _model.ModelType = _model.ModelType.PI0_FAST  # pi0.5 = PI0_FAST

    # def __call__(self, data: dict) -> dict:
    #     # observation.state is already a flat (7,) vector: [j0..j5, gripper]
    #     state = np.asarray(data["observation.state"], dtype=np.float32)

    #     # Images come in after repack — keys are "top_rgb" and "wrist_rgb"
    #     top_image   = _parse_image(data["observation.images.cam_high"])   # D415 top camera
    #     wrist_image = _parse_image(data["observation.images.cam_right_wrist"]) # D435i wrist camera

    #     inputs = {
    #         "state": state,
    #         "image": {
    #             "base_0_rgb":        top_image,
    #             "left_wrist_0_rgb":  wrist_image,
    #             # No right wrist on UR10 — fill the slot with zeros
    #             "right_wrist_0_rgb": np.zeros_like(top_image),
    #         },
    #         "image_mask": {
    #             "base_0_rgb":       np.True_,
    #             "left_wrist_0_rgb": np.True_,
    #             # pi0.5 (PI0_FAST) attends to all 3 image slots; set True so
    #             # the model doesn't ignore it. For pi0 base set False instead.
    #             "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
    #         },
    #     }

    #     if "actions" in data:
    #         inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
    #     if "prompt" in data:
    #         inputs["prompt"] = data["prompt"]
    #     if "prompt" not in data or not data["prompt"]:
    #         inputs["prompt"] = "pick up the object and place it in the box"

    #     return inputs
    # def __call__(self, data: dict) -> dict:
    #     state = np.asarray(data["observation.state"], dtype=np.float32)

    #     top_image   = _parse_image(data["observation.images.cam_high"])    # ← renamed by RepackTransform
    #     wrist_image = _parse_image(data["observation.images.cam_right_wrist"])  # ← renamed by RepackTransform

    #     inputs = {
    #         "state": state,
    #         "image": {
    #             "base_0_rgb":        top_image,
    #             "left_wrist_0_rgb":  wrist_image,
    #             "right_wrist_0_rgb": np.zeros_like(top_image),
    #         },
    #         "image_mask": {
    #             "base_0_rgb":        np.True_,
    #             "left_wrist_0_rgb":  np.True_,
    #             "right_wrist_0_rgb": np.False_,  # zero-filled slot, mask it out
    #         },
    #     }

    #     if "actions" in data:
    #         inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
    #     if "prompt" in data and data["prompt"]:
    #         inputs["prompt"] = data["prompt"]
    #     else:
    #         inputs["prompt"] = "pick up the object and place it in the box"

    #     return inputs
    def __call__(self, data: dict) -> dict:
        # ── State: handle both inference (flat keys) and training (pre-repacked) ──
        if "observation.state" in data:
            # Training path: RepackTransform already assembled this
            state = np.asarray(data["observation.state"], dtype=np.float32)
        else:
            # Inference path: flat keys from the robot client
            joints = [float(data[f"joint_{i}"]) for i in range(6)]
            gripper = float(data.get("gripper", 0.0))
            state = np.array(joints + [gripper], dtype=np.float32)

        # ── Images: handle both inference (flat cam names) and training (nested keys) ──
        if "observation.images.cam_high" in data:
            top_image   = _parse_image(data["observation.images.cam_high"])
            wrist_image = _parse_image(data["observation.images.cam_right_wrist"])
        else:
            # Inference path: camera names directly from robot.get_observation()
            top_image   = _parse_image(data["cam_high"])
            wrist_image = _parse_image(data["cam_right_wrist"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb":        top_image,
                "left_wrist_0_rgb":  wrist_image,
                "right_wrist_0_rgb": np.zeros_like(top_image),
            },
            "image_mask": {
                "base_0_rgb":        np.True_,
                "left_wrist_0_rgb":  np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data and data["prompt"]:
            inputs["prompt"] = data["prompt"]
        else:
            inputs["prompt"] = "grasp the object and place it in the box"

        return inputs

# @dataclasses.dataclass(frozen=True)
# class UR10Outputs(transforms.DataTransformFn):
#     def __call__(self, data: dict) -> dict:
#         # 7 action dims: joint_0..joint_5 + gripper
#         return {"actions": np.asarray(data["actions"][:, :7], dtype=np.float32)}

@dataclasses.dataclass(frozen=True)
class UR10Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)
        # actions shape: (action_horizon, action_dim) or (action_dim,)
        # Take the first step of the predicted action sequence
        if actions.ndim == 2:
            a = actions[0]
        else:
            a = actions
        return {
            **{f"joint_{i}": float(a[i]) for i in range(6)},
            "gripper": float(np.clip(a[6], 0.0, 1.0)),
        }