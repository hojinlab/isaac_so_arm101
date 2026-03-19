# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def object_ee_distance_and_lifted(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Combined reward for reaching the object AND lifting it."""
    # Get reaching reward
    stare_reward = object_ee_distance(env, std, object_cfg, ee_frame_cfg)
    # Get lifting reward
    lift_reward = object_is_lifted(env, minimal_height, object_cfg)
    # Combine rewards multiplicatively
    return stare_reward * lift_reward

def object_in_camera_fov(
    env: ManagerBasedRLEnv,
    std: float,
    camera_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:

    camera: Camera = env.scene[camera_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    camera_pos_w = camera.data.pos_w
    camera_quat_w = camera.data.quat_w_world
    object_pos_w = object.data.root_pos_w[:, :3]

    object_pos_cam, _ = subtract_frame_transforms(camera_pos_w, camera_quat_w, object_pos_w)

    lateral = torch.sqrt(object_pos_cam[:, 1]**2 + object_pos_cam[:, 2]**2)
    forward = object_pos_cam[:, 0]
    angle = torch.atan2(lateral, forward)

    return 1.0 - torch.tanh(angle / std)

def centroid_center_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.3,
    camera_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
) -> torch.Tensor:

    camera: Camera = env.scene[camera_cfg.name]
    seg = camera.data.output["semantic_segmentation"]

    H, W = seg.shape[1], seg.shape[2]
    mask = (seg[..., 0] == 93) & (seg[..., 1] == 220) & (seg[..., 2] == 11)

    reward = torch.zeros(env.num_envs, device=env.device)

    for i in range(env.num_envs):
        ys, xs = torch.where(mask[i])
        if len(xs) > 0:
            cx = xs.float().mean() / W
            cy = ys.float().mean() / H
            dist = torch.sqrt((cx - 0.5)**2 + (cy - 0.5)**2)
            reward[i] = 1.0 - torch.tanh(dist / std)
        else:
            reward[i] = -1.0

    return reward
