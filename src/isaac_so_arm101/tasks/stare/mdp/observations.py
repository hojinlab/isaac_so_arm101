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
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sensors import Camera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b

# def object_position_in_camera_frame(
#     env: ManagerBasedRLEnv,
#     camera_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
#     ) -> torch.Tensor:
#
#     camera: Camera = env.scene[camera_cfg.name]
#     object: RigidObject = env.scene[object_cfg.name]
#
#     camera_pos_w = camera.data.pos_w
#     camera_quat_w = camera.data.quat_w_world
#
#     object_pos_w = object.data.root_pos_w[:, :3]
#     object_pos_cam, _ = subtract_frame_transforms(camera_pos_w, camera_quat_w, object_pos_w)
#
#     return object_pos_cam

def red_block_centroid_in_camera(
    env: ManagerBasedRLEnv,
    camera_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
    ) -> torch.Tensor:
    camera: Camera = env.scene[camera_cfg.name]
    seg = camera.data.output["semantic_segmentation"]

    H, W = seg.shape[1], seg.shape[2]
    result = torch.zeros(env.num_envs, 2, device=env.device)

    mask = (seg[..., 0] == 93) & (seg[..., 1] == 220) & (seg[..., 2] == 11)

    for i in range(env.num_envs):
        ys, xs = torch.where(mask[i])
        if len(xs) > 0:
            cx = xs.float().mean() / W
            cy = ys.float().mean() / H
            result[i] = torch.stack([cx, cy])
        else:
            result[i] = torch.tensor([-1.0, -1.0], device=env.device)

    return result
