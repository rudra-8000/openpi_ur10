# src/openpi/training/ur10_config.py
"""
UR10 CB3 + PincOpen gripper finetuning config for pi0.5.
Dataset: rudra-8000/grasp_place  (40 episodes, pick-and-place)
Cameras: D415 top (cam_high) + D435i wrist (cam_right_wrist)
State/Action: 7-dim flat vector [joint_0..joint_5, gripper]
"""
from __future__ import annotations

import dataclasses
import pathlib
from typing_extensions import override

import openpi.models.model as _model
import openpi.models.pi0_fast as pi0_fast
import openpi.transforms as _transforms
from openpi.training.config import (
    DataConfig,
    DataConfigFactory,
    ModelTransformFactory,
    TrainConfig,
    AssetsConfig,
)
import openpi.training.weight_loaders as weight_loaders
import openpi.policies.ur10_policy as ur10_policy


@dataclasses.dataclass(frozen=True)
class LeRobotUR10DataConfig(DataConfigFactory):

    @override
    def create(self, assets_dir: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:

        # Map exact LeRobot dataset keys → internal names used by ur10_policy.py
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation.images.cam_high":        "top_rgb",
                        "observation.images.cam_right_wrist": "wrist_rgb",
                        "observation.state":                  "state",
                        "action":                             "actions",
                        "prompt":                             "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[ur10_policy.UR10Inputs(
                action_dim=model_config.action_dim,
                model_type=model_config.model_type,
            )],
            outputs=[ur10_policy.UR10Outputs()],
        )

        # Actions are absolute joint angles → convert to delta for training.
        # make_bool_mask(6, -1): apply delta to first 6 dims (joints),
        # keep last 1 dim (gripper) absolute — gripper is open/close, not delta.
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dir),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

