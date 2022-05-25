# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from functools import partial
from itertools import product
from typing import Tuple

import numpy as np
import torch
from pytorch3d.ops import (
    add_pointclouds_to_volumes,
    add_points_features_to_volume_densities_features,
)
from pytorch3d.ops.points_to_volumes import _points_to_volumes
from pytorch3d.ops.sample_points_from_meshes import sample_points_from_meshes
from pytorch3d.structures.meshes import Meshes
from pytorch3d.structures.pointclouds import Pointclouds
from pytorch3d.structures.volumes import Volumes
from pytorch3d.transforms.so3 import so3_exp_map

from .common_testing import TestCaseMixin


DEBUG = False
if DEBUG:
    import os
    import tempfile

    from PIL import Image


def init_cube_point_cloud(batch_size: int, n_points: int, device: str, rotate_y: bool):
    """
    Generate a random point cloud of `n_points` whose points
    are sampled from faces of a 3D cube.
    """

    # create the cube mesh batch_size times
    meshes = TestPointsToVolumes.init_cube_mesh(batch_size=batch_size, device=device)

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
        R = init_uniform_y_rotations(batch_size=batch_size, device=device)
        # rotate the point clouds around y axis
        pcl = torch.bmm(pcl - 0.5, R) + 0.5

    return pcl, rgb


def init_volume_boundary_pointcloud(
    batch_size: int,
    volume_size: Tuple[int, int, int],
    n_points: int,
    interp_mode: str,
    device: str,
    require_grad: bool = False,
):
    """
    Initialize a point cloud that closely follows a boundary of
    a volume with a given size. The volume buffer is initialized as well.
    """

    # generate a 3D point cloud sampled from sides of a [0,1] cube
    xyz, rgb = init_cube_point_cloud(
        batch_size, n_points=n_points, device=device, rotate_y=True
    )

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


def init_uniform_y_rotations(batch_size: int, device: torch.device):
    """
    Generate a batch of `batch_size` 3x3 rotation matrices around y-axis
    whose angles are uniformly distributed between 0 and 2 pi.
    """
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
        device: str,
    ):
        (pointclouds, initial_volumes) = init_volume_boundary_pointcloud(
            batch_size=batch_size,
            volume_size=volume_size,
            n_points=n_points,
            interp_mode=interp_mode,
            require_grad=False,
            device=device,
        )

        torch.cuda.synchronize()

        def _add_points_to_volumes():
            add_pointclouds_to_volumes(pointclouds, initial_volumes, mode=interp_mode)
            torch.cuda.synchronize()

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
    def init_cube_mesh(batch_size: int, device: str):
        """
        Generate a batch of `batch_size` cube meshes.
        """

        device = torch.device(device)

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

            for python, interp_mode in product([True, False], ["trilinear", "nearest"]):

                (pointclouds, initial_volumes) = init_volume_boundary_pointcloud(
                    volume_size=volume_size,
                    n_points=int(1e5),
                    interp_mode=interp_mode,
                    batch_size=batch_size,
                    require_grad=True,
                    device="cuda:0",
                )

                volumes = add_pointclouds_to_volumes(
                    pointclouds,
                    initial_volumes,
                    mode=interp_mode,
                    _python=python,
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

    def test_defaulted_arguments(self):
        points = torch.rand(30, 1000, 3)
        features = torch.rand(30, 1000, 5)
        _, densities = add_points_features_to_volume_densities_features(
            points,
            features,
            torch.zeros(30, 1, 32, 32, 32),
            torch.zeros(30, 5, 32, 32, 32),
        )
        self.assertClose(torch.sum(densities), torch.tensor(30 * 1000.0), atol=0.1)

    def test_unscaled(self):
        D = 5
        P = 1000
        B, C, H, W = 2, 3, D, D
        densities = torch.zeros(B, 1, D, H, W)
        features = torch.zeros(B, C, D, H, W)
        volumes = Volumes(densities=densities, features=features)
        points = torch.rand(B, 1000, 3) * (D - 1) - ((D - 1) * 0.5)
        point_features = torch.rand(B, 1000, C)
        pointclouds = Pointclouds(points=points, features=point_features)

        volumes2 = add_pointclouds_to_volumes(
            pointclouds, volumes, rescale_features=False
        )
        self.assertConstant(volumes2.densities().sum([2, 3, 4]) / P, 1, atol=1e-5)
        self.assertConstant(volumes2.features().sum([2, 3, 4]) / P, 0.5, atol=0.03)

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


class TestRawFunction(TestCaseMixin, unittest.TestCase):
    """
    Testing the _C.points_to_volumes function through its wrapper
    _points_to_volumes.
    """

    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_grad_corners_splat_cpu(self):
        self.do_gradcheck(torch.device("cpu"), True, True)

    def test_grad_corners_round_cpu(self):
        self.do_gradcheck(torch.device("cpu"), False, True)

    def test_grad_splat_cpu(self):
        self.do_gradcheck(torch.device("cpu"), True, False)

    def test_grad_round_cpu(self):
        self.do_gradcheck(torch.device("cpu"), False, False)

    def test_grad_corners_splat_cuda(self):
        self.do_gradcheck(torch.device("cuda:0"), True, True)

    def test_grad_corners_round_cuda(self):
        self.do_gradcheck(torch.device("cuda:0"), False, True)

    def test_grad_splat_cuda(self):
        self.do_gradcheck(torch.device("cuda:0"), True, False)

    def test_grad_round_cuda(self):
        self.do_gradcheck(torch.device("cuda:0"), False, False)

    def do_gradcheck(self, device, splat: bool, align_corners: bool):
        """
        Use gradcheck to verify the gradient of _points_to_volumes
        with random input.
        """
        N, C, D, H, W, P = 2, 4, 5, 6, 7, 5
        points_3d = (
            torch.rand((N, P, 3), device=device, dtype=torch.float64) * 0.8 + 0.1
        )
        points_features = torch.rand((N, P, C), device=device, dtype=torch.float64)
        volume_densities = torch.zeros((N, 1, D, H, W), device=device)
        volume_features = torch.zeros((N, C, D, H, W), device=device)
        volume_densities_scale = torch.rand_like(volume_densities)
        volume_features_scale = torch.rand_like(volume_features)
        grid_sizes = torch.tensor([D, H, W], dtype=torch.int64, device=device).expand(
            N, 3
        )
        mask = torch.ones((N, P), device=device)
        mask[:, 0] = 0
        align_corners = False

        def f(points_3d_, points_features_):
            (volume_densities_, volume_features_) = _points_to_volumes(
                points_3d_.to(torch.float32),
                points_features_.to(torch.float32),
                volume_densities.clone(),
                volume_features.clone(),
                grid_sizes,
                2.0,
                mask,
                align_corners,
                splat,
            )
            density = (volume_densities_ * volume_densities_scale).sum()
            features = (volume_features_ * volume_features_scale).sum()
            return density, features

        base = f(points_3d.clone(), points_features.clone())
        self.assertGreater(base[0], 0)
        self.assertGreater(base[1], 0)

        points_features.requires_grad = True
        if splat:
            points_3d.requires_grad = True
            torch.autograd.gradcheck(
                f,
                (points_3d, points_features),
                check_undefined_grad=False,
                eps=2e-4,
                atol=0.01,
            )
        else:
            torch.autograd.gradcheck(
                partial(f, points_3d),
                points_features,
                check_undefined_grad=False,
                eps=2e-3,
                atol=0.001,
            )

    def test_single_corners_round_cpu(self):
        self.single_point(torch.device("cpu"), False, True)

    def test_single_corners_splat_cpu(self):
        self.single_point(torch.device("cpu"), True, True)

    def test_single_round_cpu(self):
        self.single_point(torch.device("cpu"), False, False)

    def test_single_splat_cpu(self):
        self.single_point(torch.device("cpu"), True, False)

    def test_single_corners_round_cuda(self):
        self.single_point(torch.device("cuda:0"), False, True)

    def test_single_corners_splat_cuda(self):
        self.single_point(torch.device("cuda:0"), True, True)

    def test_single_round_cuda(self):
        self.single_point(torch.device("cuda:0"), False, False)

    def test_single_splat_cuda(self):
        self.single_point(torch.device("cuda:0"), True, False)

    def single_point(self, device, splat: bool, align_corners: bool):
        """
        Check the outcome of _points_to_volumes where a single point
        exists which lines up with a single voxel.
        """
        D, H, W = (6, 6, 11) if align_corners else (5, 5, 10)
        N, C, P = 1, 1, 1
        if align_corners:
            points_3d = torch.tensor([[[-0.2, 0.2, -0.2]]], device=device)
        else:
            points_3d = torch.tensor([[[-0.3, 0.4, -0.4]]], device=device)
        points_features = torch.zeros((N, P, C), device=device)
        volume_densities = torch.zeros((N, 1, D, H, W), device=device)
        volume_densities_expected = torch.zeros((N, 1, D, H, W), device=device)
        volume_features = torch.zeros((N, C, D, H, W), device=device)
        grid_sizes = torch.tensor([D, H, W], dtype=torch.int64, device=device).expand(
            N, 3
        )
        mask = torch.ones((N, P), device=device)
        point_weight = 19.0

        volume_densities_, volume_features_ = _points_to_volumes(
            points_3d,
            points_features,
            volume_densities,
            volume_features,
            grid_sizes,
            point_weight,
            mask,
            align_corners,
            splat,
        )

        self.assertTrue(volume_densities_.is_set_to(volume_densities))
        self.assertTrue(volume_features_.is_set_to(volume_features))

        if align_corners:
            volume_densities_expected[0, 0, 2, 3, 4] = point_weight
        else:
            volume_densities_expected[0, 0, 1, 3, 3] = point_weight

        self.assertClose(volume_densities, volume_densities_expected)
