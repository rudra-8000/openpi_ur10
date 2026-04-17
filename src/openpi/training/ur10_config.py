@dataclasses.dataclass(frozen=True)
class LeRobotUR10DataConfig(DataConfigFactory):

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:

        # Maps the exact keys from your info.json → what ur10_policy.py expects
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation.images.cam_high":         "cam_high",
                        "observation.images.cam_right_wrist":  "cam_right_wrist",
                        "observation.state":                   "state",
                        "action":                              "actions",
                        "prompt":                              "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[ur10_policy.UR10Inputs(model_type=model_config.model_type)],
            outputs=[ur10_policy.UR10Outputs()],
        )

        # Your actions are ABSOLUTE joint angles → apply delta transform for training
        # make_bool_mask(6, -1) means: delta on first 6 dims, absolute on last 1 (gripper)
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )