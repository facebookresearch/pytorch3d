# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# TODO: all this potentially goes to PyTorch3D

import math
from typing import Tuple

import pytorch3d as pt3d
import torch
from pytorch3d.renderer.cameras import CamerasBase


def jitter_extrinsics(
    R: torch.Tensor,
    T: torch.Tensor,
    max_angle: float = (math.pi * 2.0),
    translation_std: float = 1.0,
    scale_std: float = 0.3,
):
    """
    Jitter the extrinsic camera parameters `R` and `T` with a random similarity
    transformation. The transformation rotates by a random angle between [0, max_angle];
    scales by a random factor exp(N(0, scale_std)), where N(0, scale_std) is
    a random sample from a normal distrubtion with zero mean and variance scale_std;
    and translates by a 3D offset sampled from N(0, translation_std).
    """
    assert all(x >= 0.0 for x in (max_angle, translation_std, scale_std))
    N = R.shape[0]
    R_jit = pt3d.transforms.random_rotations(1, device=R.device)
    R_jit = pt3d.transforms.so3_exponential_map(
        pt3d.transforms.so3_log_map(R_jit) * max_angle
    )
    T_jit = torch.randn_like(R_jit[:1, :, 0]) * translation_std
    rigid_transform = pt3d.ops.eyes(dim=4, N=N, device=R.device)
    rigid_transform[:, :3, :3] = R_jit.expand(N, 3, 3)
    rigid_transform[:, 3, :3] = T_jit.expand(N, 3)
    scale_jit = torch.exp(torch.randn_like(T_jit[:, 0]) * scale_std).expand(N)
    return apply_camera_alignment(R, T, rigid_transform, scale_jit)


def apply_camera_alignment(
    R: torch.Tensor,
    T: torch.Tensor,
    rigid_transform: torch.Tensor,
    scale: torch.Tensor,
):
    """
    Args:
        R: Camera rotation matrix of shape (N, 3, 3).
        T: Camera translation  of shape (N, 3).
        rigid_transform: A tensor of shape (N, 4, 4) representing a batch of
            N 4x4 tensors that map the scene pointcloud from misaligned coords
            to the aligned space.
        scale: A list of N scaling factors. A tensor of shape (N,)

    Returns:
        R_aligned: The aligned rotations R.
        T_aligned: The aligned translations T.
    """
    R_rigid = rigid_transform[:, :3, :3]
    T_rigid = rigid_transform[:, 3:, :3]
    R_aligned = R_rigid.permute(0, 2, 1).bmm(R)
    T_aligned = scale[:, None] * (T - (T_rigid @ R_aligned)[:, 0])
    return R_aligned, T_aligned


def get_min_max_depth_bounds(cameras, scene_center, scene_extent):
    """
    Estimate near/far depth plane as:
    near = dist(cam_center, self.scene_center) - self.scene_extent
    far  = dist(cam_center, self.scene_center) + self.scene_extent
    """
    cam_center = cameras.get_camera_center()
    center_dist = (
        ((cam_center - scene_center.to(cameras.R)[None]) ** 2)
        .sum(dim=-1)
        .clamp(0.001)
        .sqrt()
    )
    center_dist = center_dist.clamp(scene_extent + 1e-3)
    min_depth = center_dist - scene_extent
    max_depth = center_dist + scene_extent
    return min_depth, max_depth


def volumetric_camera_overlaps(
    cameras: CamerasBase,
    scene_extent: float = 8.0,
    scene_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    resol: int = 16,
    weigh_by_ray_angle: bool = True,
):
    """
    Compute the overlaps between viewing frustrums of all pairs of cameras
    in `cameras`.
    """
    device = cameras.device
    ba = cameras.R.shape[0]
    n_vox = int(resol**3)
    grid = pt3d.structures.Volumes(
        densities=torch.zeros([1, 1, resol, resol, resol], device=device),
        volume_translation=-torch.FloatTensor(scene_center)[None].to(device),
        voxel_size=2.0 * scene_extent / resol,
    ).get_coord_grid(world_coordinates=True)

    grid = grid.view(1, n_vox, 3).expand(ba, n_vox, 3)
    gridp = cameras.transform_points(grid, eps=1e-2)
    proj_in_camera = (
        torch.prod((gridp[..., :2].abs() <= 1.0), dim=-1)
        * (gridp[..., 2] > 0.0).float()
    )  # ba x n_vox

    if weigh_by_ray_angle:
        rays = torch.nn.functional.normalize(
            grid - cameras.get_camera_center()[:, None], dim=-1
        )
        rays_masked = rays * proj_in_camera[..., None]

        # - slow and readable:
        # inter = torch.zeros(ba, ba)
        # for i1 in range(ba):
        #     for i2 in range(ba):
        #         inter[i1, i2] = (
        #             1 + (rays_masked[i1] * rays_masked[i2]
        #         ).sum(dim=-1)).sum()

        # - fast:
        rays_masked = rays_masked.view(ba, n_vox * 3)
        inter = n_vox + (rays_masked @ rays_masked.t())

    else:
        inter = proj_in_camera @ proj_in_camera.t()

    mass = torch.diag(inter)
    iou = inter / (mass[:, None] + mass[None, :] - inter).clamp(0.1)

    return iou
