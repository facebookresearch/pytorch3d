# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch

from omegaconf import DictConfig, OmegaConf
from pytorch3d.implicitron.models.implicit_function.voxel_grid_implicit_function import (
    VoxelGridImplicitFunction,
)
from pytorch3d.implicitron.models.renderer.base import ImplicitronRayBundle

from pytorch3d.implicitron.tools.config import expand_args_fields, get_default_args
from pytorch3d.renderer import ray_bundle_to_ray_points
from tests.common_testing import TestCaseMixin


class TestVoxelGridImplicitFunction(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        expand_args_fields(VoxelGridImplicitFunction)

    def _get_simple_implicit_function(self, scaffold_res=16):
        default_cfg = get_default_args(VoxelGridImplicitFunction)
        custom_cfg = DictConfig(
            {
                "voxel_grid_density_args": {
                    "voxel_grid_FullResolutionVoxelGrid_args": {"n_features": 7}
                },
                "decoder_density_class_type": "ElementwiseDecoder",
                "decoder_color_class_type": "MLPDecoder",
                "decoder_color_MLPDecoder_args": {
                    "network_args": {
                        "n_layers": 2,
                        "output_dim": 3,
                        "hidden_dim": 128,
                    }
                },
                "scaffold_resolution": (scaffold_res, scaffold_res, scaffold_res),
            }
        )
        cfg = OmegaConf.merge(default_cfg, custom_cfg)
        return VoxelGridImplicitFunction(**cfg)

    def test_forward(self) -> None:
        """
        Test one forward of VoxelGridImplicitFunction.
        """
        func = self._get_simple_implicit_function()

        n_grids, n_points = 10, 9
        raybundle = ImplicitronRayBundle(
            origins=torch.randn(n_grids, 2, 3, 3),
            directions=torch.randn(n_grids, 2, 3, 3),
            lengths=torch.randn(n_grids, 2, 3, n_points),
            xys=0,
        )
        func(raybundle)

    def test_scaffold_formation(self):
        """
        Test calculating the scaffold.

        We define a custom density function and make the implicit function use it
        After calculating the scaffold we compare the density of our custom
        density function with densities from the scaffold.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        func = self._get_simple_implicit_function().to(device)
        func.scaffold_max_pool_kernel_size = 1

        def new_density(points):
            """
            Density function which returns 1 if p>(0.5, 0.5, 0.5) or
            p < (-0.5, -0.5, -0.5) else 0
            """
            inshape = points.shape
            points = points.view(-1, 3)
            out = []
            for p in points:
                if torch.all(p > 0.5) or torch.all(p < -0.5):
                    out.append(torch.tensor([[1.0]]))
                else:
                    out.append(torch.tensor([[0.0]]))
            return torch.cat(out).view(*inshape[:-1], 1).to(device)

        func._get_density = new_density
        func._get_scaffold(0)

        points = torch.tensor(
            [
                [0, 0, 0],
                [1, 1, 1],
                [1, 0, 0],
                [0.1, 0, 0],
                [10, 1, -1],
                [-0.8, -0.7, -0.9],
            ]
        ).to(device)
        expected = new_density(points).float().to(device)
        assert torch.allclose(func.voxel_grid_scaffold(points), expected), (
            func.voxel_grid_scaffold(points),
            expected,
        )

    def test_scaffold_filtering(self, n_test_points=100):
        """
        Test that filtering points with scaffold works.

        We define a scaffold and make the implicit function use it. We also
        define new density and color functions which check that all passed
        points are not in empty space (with scaffold function). In the end
        we compare the result from the implicit function with one calculated
        simple python, this checks that the points were merged correectly.
        """
        device = "cuda"
        func = self._get_simple_implicit_function().to(device)

        def scaffold(points):
            """'
            Function to deterministically and randomly enough assign a point
            to empty or occupied space.
            Return 1 if second digit of sum after 0 is odd else 0
            """
            return (
                ((points.sum(dim=-1, keepdim=True) * 10**2 % 10).long() % 2) == 1
            ).float()

        def new_density(points):
            # check if all passed points should be passed here
            assert torch.all(scaffold(points)), (scaffold(points), points.shape)
            return points.sum(dim=-1, keepdim=True)

        def new_color(points, camera, directions, non_empty_points, num_points_per_ray):
            # check if all passed points should be passed here
            assert torch.all(scaffold(points))  # , (scaffold(points), points)
            return points * 2

        # check both computation paths that they contain only points
        # which are not in empty space
        func._get_density = new_density
        func._get_color = new_color
        func.voxel_grid_scaffold.forward = scaffold
        func._scaffold_ready = True

        bundle = ImplicitronRayBundle(
            origins=torch.rand((n_test_points, 2, 1, 3), device=device),
            directions=torch.rand((n_test_points, 2, 1, 3), device=device),
            lengths=torch.rand((n_test_points, 2, 1, 4), device=device),
            xys=None,
        )
        points = ray_bundle_to_ray_points(bundle)
        result_density, result_color, _ = func(bundle)

        # construct the wanted result 'by hand'
        flat_points = points.view(-1, 3)
        expected_result_density, expected_result_color = [], []
        for point in flat_points:
            if scaffold(point) == 1:
                expected_result_density.append(point.sum(dim=-1, keepdim=True))
                expected_result_color.append(point * 2)
            else:
                expected_result_density.append(point.new_zeros((1,)))
                expected_result_color.append(point.new_zeros((3,)))
        expected_result_density = torch.stack(expected_result_density, dim=0).view(
            *points.shape[:-1], 1
        )
        expected_result_color = torch.stack(expected_result_color, dim=0).view(
            *points.shape[:-1], 3
        )

        # check that thre result is expected
        assert torch.allclose(result_density, expected_result_density), (
            result_density,
            expected_result_density,
        )
        assert torch.allclose(result_color, expected_result_color), (
            result_color,
            expected_result_color,
        )

    def test_cropping(self, scaffold_res=9):
        """
        Tests whether implicit function finds the bounding box of the object and sends
        correct min and max points to voxel grids for rescaling.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        func = self._get_simple_implicit_function(scaffold_res=scaffold_res).to(device)

        assert scaffold_res >= 8
        div = (scaffold_res - 1) / 2
        true_min_point = torch.tensor(
            [-3 / div, 0 / div, -3 / div],
            device=device,
        )
        true_max_point = torch.tensor(
            [1 / div, 2 / div, 3 / div],
            device=device,
        )

        def new_scaffold(points):
            # 1 if between true_min and true_max point else 0
            # return points.new_ones((*points.shape[:-1], 1))
            return (
                torch.logical_and(true_min_point <= points, points <= true_max_point)
                .all(dim=-1)
                .float()[..., None]
            )

        called_crop = []

        def assert_min_max_points(min_point, max_point):
            called_crop.append(1)
            self.assertClose(min_point, true_min_point)
            self.assertClose(max_point, true_max_point)

        func.voxel_grid_density.crop_self = assert_min_max_points
        func.voxel_grid_color.crop_self = assert_min_max_points
        func.voxel_grid_scaffold.forward = new_scaffold
        func._scaffold_ready = True
        func._crop(epoch=0)
        assert len(called_crop) == 2
