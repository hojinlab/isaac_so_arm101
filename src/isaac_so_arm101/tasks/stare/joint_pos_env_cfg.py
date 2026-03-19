# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


# import mdp
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab.utils import configclass
from isaac_so_arm101.robots import SO_ARM100_CFG, SO_ARM101_CFG  # noqa: F401
from isaac_so_arm101.tasks.stare.stare_env_cfg import StareEnvCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Scene definition
##


@configclass
class SoArm100StareEnvCfg(StareEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to franka
        self.scene.robot = SO_ARM100_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override rewards
        # self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["gripper"]
        # self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["gripper"]
        # self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["gripper"]

        # TODO: reorient command target

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            scale=0.5,
            use_default_offset=True,
        )
        # override command generator body
        # end-effector is along z-direction
        # self.commands.ee_pose.body_name = ["gripper"]
        # self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)


@configclass
class SoArm100StareEnvCfg_PLAY(SoArm100StareEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


@configclass
class SoArm101StareEnvCfg(StareEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to franka
        self.scene.robot = SO_ARM101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override rewards
        # self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["gripper_link"]
        # self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["gripper_link"]
        # self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["gripper_link"]

        # self.rewards.end_effector_orientation_tracking.weight = 0.0

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=0.5,
            use_default_offset=True,
        )
        # override command generator body
        # end-effector is along z-direction
        # self.commands.ee_pose.body_name = ["gripper_link"]
        # self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)

        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.2, 0.0, 0.015], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                # usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
                scale=(0.5, 0.5, 0.5),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                semantic_tags=[("class", "red_block")],
            ),
        )


@configclass
class SoArm101StareEnvCfg_PLAY(SoArm101StareEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
