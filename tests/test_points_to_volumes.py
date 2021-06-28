# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Tuple

import numpy as np
import torch
from common_testing import TestCaseMixin
from pytorch3d.ops import add_pointclouds_to_volumes
from pytorch3d.ops.sample_points_from_meshes import sample_points_from_meshes
from pytorch3d.structures.meshes import Meshes
from pytorch3d.structures.pointclouds import Pointclouds
from pytorch3d.structures.volumes import Volumes
from pytorch3d.transforms.so3 import so3_exp_map


DEBUG = False
if DEBUG:
    import os
    import tempfile

    from PIL import Image


def init_cube_point_cloud(
    batch_size: int = 10, n_points: int = 100000, rotate_y: bool = True
):
    """
    Generate a random point cloud of `n_points` whose points of
    which are sampled from faces of a 3D cube.
    """

    # create the cube mesh batch_size times
    meshes = TestPointsToVolumes.init_cube_mesh(batch_size)

    # generate point clouds by sampling points from the meshes
    pcl = sample_points_from_meshes(meshes, num_samples=n_points, return_normals=False)

    # colors of the cube sides
    clrs = [
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
    ]

    # init the color tensor "rgb"
    rgb = torch.zeros_like(pcl)

    # color each side of the cube with a constant color
    clri = 0
    for dim in (0, 1, 2):
        for offs in (0.0, 1.0):
            current_face_verts = (pcl[:, :, dim] - offs).abs() <= 1e-2
            for bi in range(batch_size):
                rgb[bi, current_face_verts[bi], :] = torch.tensor(clrs[clri]).type_as(
                    pcl
                )
            clri += 1

    if rotate_y:
        # uniformly spaced rotations around y axis
        R = init_uniform_y_rotations(batch_size=batch_size)
        # rotate the point clouds around y axis
        pcl = torch.bmm(pcl - 0.5, R) + 0.5

    return pcl, rgb


def init_volume_boundary_pointcloud(
    batch_size: int,
    volume_size: Tuple[int, int, int],
    n_points: int,
    interp_mode: str,
    require_grad: bool = False,
):
    """
    Initialize a point cloud that closely follows a boundary of
    a volume with a given size. The volume buffer is initialized as well.
    """

    # generate a 3D point cloud sampled from sides of a [0,1] cube
    xyz, rgb = init_cube_point_cloud(batch_size, n_points=n_points, rotate_y=True)

    # make volume_size tensor
    volume_size_t = torch.tensor(volume_size, dtype=xyz.dtype, device=xyz.device)

    if interp_mode == "trilinear":
        # make the xyz locations fall on the boundary of the
        # first/last two voxels along each spatial dimension of the
        # volume - this properly checks the correctness of the
        # trilinear interpolation scheme
        xyz = (xyz - 0.5) * ((volume_size_t - 2) / (volume_size_t - 1))[[2, 1, 0]] + 0.5

    # rescale the cube pointcloud to overlap with the volume sides
    # of the volume
    rel_scale = volume_size_t / volume_size[0]
    xyz = xyz * rel_scale[[2, 1, 0]][None, None]

    # enable grad accumulation for the differentiability check
    xyz.requires_grad = require_grad
    rgb.requires_grad = require_grad

    # create the pointclouds structure
    pointclouds = Pointclouds(xyz, features=rgb)

    # set the volume translation so that the point cloud is centered
    # around 0
    volume_translation = -0.5 * rel_scale[[2, 1, 0]]

    # set the voxel size to 1 / (volume_size-1)
    volume_voxel_size = 1 / (volume_size[0] - 1.0)

    # instantiate the volumes
    initial_volumes = Volumes(
        features=xyz.new_zeros(batch_size, 3, *volume_size),
        densities=xyz.new_zeros(batch_size, 1, *volume_size),
        volume_translation=volume_translation,
        voxel_size=volume_voxel_size,
    )

    return pointclouds, initial_volumes


def init_uniform_y_rotations(batch_size: int = 10):
    """
    Generate a batch of `batch_size` 3x3 rotation matrices around y-axis
    whose angles are uniformly distributed between 0 and 2 pi.
    """
    device = torch.device("cuda:0")
    axis = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32)
    angles = torch.linspace(0, 2.0 * np.pi, batch_size + 1, device=device)
    angles = angles[:batch_size]
    log_rots = axis[None, :] * angles[:, None]
    R = so3_exp_map(log_rots)
    return R


class TestPointsToVolumes(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)
        torch.manual_seed(42)

    @staticmethod
    def add_points_to_volumes(
        batch_size: int,
        volume_size: Tuple[int, int, int],
        n_points: int,
        interp_mode: str,
    ):
        (pointclouds, initial_volumes) = init_volume_boundary_pointcloud(
            batch_size=batch_size,
            volume_size=volume_size,
            n_points=n_points,
            interp_mode=interp_mode,
            require_grad=False,
        )

        def _add_points_to_volumes():
            add_pointclouds_to_volumes(pointclouds, initial_volumes, mode=interp_mode)

        return _add_points_to_volumes

    @staticmethod
    def stack_4d_tensor_to_3d(arr):
        n = arr.shape[0]
        H = int(np.ceil(np.sqrt(n)))
        W = int(np.ceil(n / H))
        n_add = H * W - n
        arr = torch.cat((arr, torch.zeros_like(arr[:1]).repeat(n_add, 1, 1, 1)))
        rows = torch.chunk(arr, chunks=W, dim=0)
        arr3d = torch.cat([torch.cat(list(row), dim=2) for row in rows], dim=1)
        return arr3d

    @staticmethod
    def init_cube_mesh(batch_size: int = 10):
        """
        Generate a batch of `batch_size` cube meshes.
        """

        device = torch.device("cuda:0")

        verts, faces = [], []

        for _ in range(batch_size):
            v = torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
                device=device,
            )
            verts.append(v)
            faces.append(
                torch.tensor(
                    [
                        [0, 2, 1],
                        [0, 3, 2],
                        [2, 3, 4],
                        [2, 4, 5],
                        [1, 2, 5],
                        [1, 5, 6],
                        [0, 7, 4],
                        [0, 4, 3],
                        [5, 4, 7],
                        [5, 7, 6],
                        [0, 6, 7],
                        [0, 1, 6],
                    ],
                    dtype=torch.int64,
                    device=device,
                )
            )

        faces = torch.stack(faces)
        verts = torch.stack(verts)

        simpleces = Meshes(verts=verts, faces=faces)

        return simpleces

    def test_from_point_cloud(self, interp_mode="trilinear"):
        """
        Generates a volume from a random point cloud sampled from faces
        of a 3D cube. Since each side of the cube is homogeneously colored with
        a different color, this should result in a volume with a
        predefined homogeneous color of the cells along its borders
        and black interior. The test is run for both cube and non-cube shaped
        volumes.
        """

        # batch_size = 4 sides of the cube
        batch_size = 4

        for volume_size in ([25, 25, 25], [30, 25, 15]):

            for interp_mode in ("trilinear", "nearest"):

                (pointclouds, initial_volumes) = init_volume_boundary_pointcloud(
                    volume_size=volume_size,
                    n_points=int(1e5),
                    interp_mode=interp_mode,
                    batch_size=batch_size,
                    require_grad=True,
                )

                volumes = add_pointclouds_to_volumes(
                    pointclouds, initial_volumes, mode=interp_mode
                )

                V_color, V_density = volumes.features(), volumes.densities()

                # expected colors of different cube sides
                clr_sides = torch.tensor(
                    [
                        [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]],
                        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                        [[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
                        [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
                    ],
                    dtype=V_color.dtype,
                    device=V_color.device,
                )
                clr_ambient = torch.tensor(
                    [0.0, 0.0, 0.0], dtype=V_color.dtype, device=V_color.device
                )
                clr_top_bot = torch.tensor(
                    [[0.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
                    dtype=V_color.dtype,
                    device=V_color.device,
                )

                if DEBUG:
                    outdir = tempfile.gettempdir() + "/test_points_to_volumes"
                    os.makedirs(outdir, exist_ok=True)

                    for slice_dim in (1, 2):
                        for vidx in range(V_color.shape[0]):
                            vim = V_color.detach()[vidx].split(1, dim=slice_dim)
                            vim = torch.stack([v.squeeze() for v in vim])
                            vim = TestPointsToVolumes.stack_4d_tensor_to_3d(vim.cpu())
                            im = Image.fromarray(
                                (vim.numpy() * 255.0)
                                .astype(np.uint8)
                                .transpose(1, 2, 0)
                            )
                            outfile = (
                                outdir
                                + f"/rgb_{interp_mode}"
                                + f"_{str(volume_size).replace(' ','')}"
                                + f"_{vidx:003d}_sldim{slice_dim}.png"
                            )
                            im.save(outfile)
                            print("exported %s" % outfile)

                # check the density V_density
                # first binarize the density
                V_density_bin = (V_density > 1e-4).type_as(V_density)
                d_one = V_density.new_ones(1)
                d_zero = V_density.new_zeros(1)
                for vidx in range(V_color.shape[0]):
                    # the first/last depth-wise slice has to be filled with 1.0
                    self._check_volume_slice_color_density(
                        V_density_bin[vidx], 1, interp_mode, d_one, "first"
                    )
                    self._check_volume_slice_color_density(
                        V_density_bin[vidx], 1, interp_mode, d_one, "last"
                    )
                    # the middle depth-wise slices have to be empty
                    self._check_volume_slice_color_density(
                        V_density_bin[vidx], 1, interp_mode, d_zero, "middle"
                    )
                    # the top/bottom slices have to be filled with 1.0
                    self._check_volume_slice_color_density(
                        V_density_bin[vidx], 2, interp_mode, d_one, "first"
                    )
                    self._check_volume_slice_color_density(
                        V_density_bin[vidx], 2, interp_mode, d_one, "last"
                    )

                # check the colors
                for vidx in range(V_color.shape[0]):
                    self._check_volume_slice_color_density(
                        V_color[vidx], 1, interp_mode, clr_sides[vidx][0], "first"
                    )
                    self._check_volume_slice_color_density(
                        V_color[vidx], 1, interp_mode, clr_sides[vidx][1], "last"
                    )
                    self._check_volume_slice_color_density(
                        V_color[vidx], 1, interp_mode, clr_ambient, "middle"
                    )
                    self._check_volume_slice_color_density(
                        V_color[vidx], 2, interp_mode, clr_top_bot[0], "first"
                    )
                    self._check_volume_slice_color_density(
                        V_color[vidx], 2, interp_mode, clr_top_bot[1], "last"
                    )

                # check differentiability
                loss = V_color.mean() + V_density.mean()
                loss.backward()
                rgb = pointclouds.features_padded()
                xyz = pointclouds.points_padded()
                for field in (xyz, rgb):
                    if interp_mode == "nearest" and (field is xyz):
                        # this does not produce grads w.r.t. xyz
                        self.assertIsNone(field.grad)
                    else:
                        self.assertTrue(torch.isfinite(field.grad.data).all())

    def _check_volume_slice_color_density(
        self, V, split_dim, interp_mode, clr_gt, slice_type, border=3
    ):
        # decompose the volume to individual slices along split_dim
        vim = V.detach().split(1, dim=split_dim)
        vim = torch.stack([v.squeeze(split_dim) for v in vim])

        # determine which slices should be compared to clr_gt based on
        # the 'slice_type' input
        if slice_type == "first":
            slice_dims = (0, 1) if interp_mode == "trilinear" else (0,)
        elif slice_type == "last":
            slice_dims = (-1, -2) if interp_mode == "trilinear" else (-1,)
        elif slice_type == "middle":
            internal_border = 2 if interp_mode == "trilinear" else 1
            slice_dims = torch.arange(internal_border, vim.shape[0] - internal_border)
        else:
            raise ValueError(slice_type)

        # compute the average error within each slice
        clr_diff = (
            vim[slice_dims, :, border:-border, border:-border]
            - clr_gt[None, :, None, None]
        )
        clr_diff = clr_diff.abs().mean(dim=(2, 3)).view(-1)

        # check that all per-slice avg errors vanish
        self.assertClose(clr_diff, torch.zeros_like(clr_diff), atol=1e-2)
