# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import numpy as np
import torch
from pytorch3d.common.compat import meshgrid_ij
from pytorch3d.ops import eyes
from pytorch3d.renderer import (
    AlphaCompositor,
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
)
from pytorch3d.renderer.utils import (
    ndc_grid_sample,
    ndc_to_grid_sample_coords,
    TensorProperties,
)
from pytorch3d.structures import Pointclouds

from .common_testing import TestCaseMixin


# Example class for testing
class TensorPropertiesTestClass(TensorProperties):
    def __init__(self, x=None, y=None, device="cpu"):
        super().__init__(device=device, x=x, y=y)

    def clone(self):
        other = TensorPropertiesTestClass()
        return super().clone(other)


class TestTensorProperties(TestCaseMixin, unittest.TestCase):
    def test_init(self):
        example = TensorPropertiesTestClass(x=10.0, y=(100.0, 200.0))
        # Check kwargs set as attributes + converted to tensors
        self.assertTrue(torch.is_tensor(example.x))
        self.assertTrue(torch.is_tensor(example.y))
        # Check broadcasting
        self.assertTrue(example.x.shape == (2,))
        self.assertTrue(example.y.shape == (2,))
        self.assertTrue(len(example) == 2)

    def test_to(self):
        # Check to method
        example = TensorPropertiesTestClass(x=10.0, y=(100.0, 200.0))
        device = torch.device("cuda:0")
        new_example = example.to(device=device)
        self.assertEqual(new_example.device, device)

        example_cpu = example.cpu()
        self.assertEqual(example_cpu.device, torch.device("cpu"))

        example_gpu = example.cuda()
        self.assertEqual(example_gpu.device.type, "cuda")
        self.assertIsNotNone(example_gpu.device.index)

        example_gpu1 = example.cuda(1)
        self.assertEqual(example_gpu1.device, torch.device("cuda:1"))

    def test_clone(self):
        # Check clone method
        example = TensorPropertiesTestClass(x=10.0, y=(100.0, 200.0))
        new_example = example.clone()
        self.assertSeparate(example.x, new_example.x)
        self.assertSeparate(example.y, new_example.y)

    def test_get_set(self):
        # Test getitem returns an accessor which can be used to modify
        # attributes at a particular index
        example = TensorPropertiesTestClass(x=10.0, y=(100.0, 200.0, 300.0))

        # update y1
        example[1].y = 5.0
        self.assertTrue(example.y[1] == 5.0)

        # Get item and get value
        ex0 = example[0]
        self.assertTrue(ex0.y == 100.0)

    def test_empty_input(self):
        example = TensorPropertiesTestClass(x=(), y=())
        self.assertTrue(len(example) == 0)
        self.assertTrue(example.isempty())

    def test_gather_props(self):
        N = 4
        x = torch.randn((N, 3, 4))
        y = torch.randn((N, 5))
        test_class = TensorPropertiesTestClass(x=x, y=y)

        S = 15
        idx = torch.tensor(np.random.choice(N, S))
        test_class_gathered = test_class.gather_props(idx)

        self.assertTrue(test_class_gathered.x.shape == (S, 3, 4))
        self.assertTrue(test_class_gathered.y.shape == (S, 5))

        for i in range(N):
            inds = idx == i
            if inds.sum() > 0:
                # Check the gathered points in the output have the same value from
                # the input.
                self.assertClose(test_class_gathered.x[inds].mean(dim=0), x[i, ...])
                self.assertClose(test_class_gathered.y[inds].mean(dim=0), y[i, ...])

    def test_ndc_grid_sample_rendering(self):
        """
        Use PyTorch3D point renderer to render a colored point cloud, then
        sample the image at the locations of the point projections with
        `ndc_grid_sample`. Finally, assert that the sampled colors are equal to the
        original point cloud colors.

        Note that, in order to ensure correctness, we use a nearest-neighbor
        assignment point renderer (i.e. no soft splatting).
        """

        # generate a bunch of 3D points on a regular grid lying in the z-plane
        n_grid_pts = 10
        grid_scale = 0.9
        z_plane = 2.0
        image_size = [128, 128]
        point_radius = 0.015
        n_pts = n_grid_pts * n_grid_pts
        pts = torch.stack(
            meshgrid_ij(
                [torch.linspace(-grid_scale, grid_scale, n_grid_pts)] * 2,
            ),
            dim=-1,
        )
        pts = torch.cat([pts, z_plane * torch.ones_like(pts[..., :1])], dim=-1)
        pts = pts.reshape(1, n_pts, 3)

        # color the points randomly
        pts_colors = torch.rand(1, n_pts, 3)

        # make trivial rendering cameras
        cameras = PerspectiveCameras(
            R=eyes(dim=3, N=1),
            device=pts.device,
            T=torch.zeros(1, 3, dtype=torch.float32, device=pts.device),
        )

        # render the point cloud
        pcl = Pointclouds(points=pts, features=pts_colors)
        renderer = NearestNeighborPointsRenderer(
            rasterizer=PointsRasterizer(
                cameras=cameras,
                raster_settings=PointsRasterizationSettings(
                    image_size=image_size,
                    radius=point_radius,
                    points_per_pixel=1,
                ),
            ),
            compositor=AlphaCompositor(),
        )
        im_render = renderer(pcl)

        # sample the render at projected pts
        pts_proj = cameras.transform_points(pcl.points_padded())[..., :2]
        pts_colors_sampled = ndc_grid_sample(
            im_render,
            pts_proj,
            mode="nearest",
            align_corners=False,
        ).permute(0, 2, 1)

        # assert that the samples are the same as original points
        self.assertClose(pts_colors, pts_colors_sampled, atol=1e-4)

    def test_ndc_to_grid_sample_coords(self):
        """
        Test the conversion from ndc to grid_sample coords by comparing
        to known conversion results.
        """

        # square image tests
        image_size_square = [100, 100]
        xy_ndc_gs_square = torch.FloatTensor(
            [
                # 4 corners
                [[-1.0, -1.0], [1.0, 1.0]],
                [[1.0, 1.0], [-1.0, -1.0]],
                [[1.0, -1.0], [-1.0, 1.0]],
                [[1.0, 1.0], [-1.0, -1.0]],
                # center
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        )

        # non-batched version
        for xy_ndc, xy_gs in xy_ndc_gs_square:
            xy_gs_predicted = ndc_to_grid_sample_coords(
                xy_ndc,
                image_size_square,
            )
            self.assertClose(xy_gs_predicted, xy_gs)

        # batched version
        xy_ndc, xy_gs = xy_ndc_gs_square[:, 0], xy_ndc_gs_square[:, 1]
        xy_gs_predicted = ndc_to_grid_sample_coords(
            xy_ndc,
            image_size_square,
        )
        self.assertClose(xy_gs_predicted, xy_gs)

        # non-square image tests
        image_size = [100, 200]
        xy_ndc_gs = torch.FloatTensor(
            [
                # 4 corners
                [[-2.0, -1.0], [1.0, 1.0]],
                [[2.0, -1.0], [-1.0, 1.0]],
                [[-2.0, 1.0], [1.0, -1.0]],
                [[2.0, 1.0], [-1.0, -1.0]],
                # center
                [[0.0, 0.0], [0.0, 0.0]],
                # non-corner points
                [[4.0, 0.5], [-2.0, -0.5]],
                [[1.0, -0.5], [-0.5, 0.5]],
            ]
        )

        # check both H > W and W > H
        for flip_axes in [False, True]:

            # non-batched version
            for xy_ndc, xy_gs in xy_ndc_gs:
                xy_gs_predicted = ndc_to_grid_sample_coords(
                    xy_ndc.flip(dims=(-1,)) if flip_axes else xy_ndc,
                    list(reversed(image_size)) if flip_axes else image_size,
                )
                self.assertClose(
                    xy_gs_predicted,
                    xy_gs.flip(dims=(-1,)) if flip_axes else xy_gs,
                )

            # batched version
            xy_ndc, xy_gs = xy_ndc_gs[:, 0], xy_ndc_gs[:, 1]
            xy_gs_predicted = ndc_to_grid_sample_coords(
                xy_ndc.flip(dims=(-1,)) if flip_axes else xy_ndc,
                list(reversed(image_size)) if flip_axes else image_size,
            )
            self.assertClose(
                xy_gs_predicted,
                xy_gs.flip(dims=(-1,)) if flip_axes else xy_gs,
            )


class NearestNeighborPointsRenderer(PointsRenderer):
    """
    A class for rendering a batch of points by a trivial nearest
    neighbor assignment.
    """

    def forward(self, point_clouds, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(point_clouds, **kwargs)
        # set all weights trivially to one
        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = torch.ones_like(dists2)
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )
        return images
