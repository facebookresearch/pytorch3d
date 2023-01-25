# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from pytorch3d.renderer import HeterogeneousRayBundle, PerspectiveCameras, RayBundle
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.transforms import random_rotations

# Some of these imports are only needed for testing code coverage
from pytorch3d.vis import (  # noqa: F401
    get_camera_wireframe,  # noqa: F401
    plot_batch_individually,  # noqa: F401
    plot_scene,
    texturesuv_image_PIL,  # noqa: F401
)


class TestPlotlyVis(unittest.TestCase):
    def test_plot_scene(
        self,
        B: int = 3,
        n_rays: int = 128,
        n_pts_per_ray: int = 32,
        n_verts: int = 32,
        n_edges: int = 64,
        n_pts: int = 256,
    ):
        """
        Tests plotting of all supported structures using plot_scene.
        """
        for device in ["cpu", "cuda:0"]:
            plot_scene(
                {
                    "scene": {
                        "ray_bundle": RayBundle(
                            origins=torch.randn(B, n_rays, 3, device=device),
                            xys=torch.randn(B, n_rays, 2, device=device),
                            directions=torch.randn(B, n_rays, 3, device=device),
                            lengths=torch.randn(
                                B, n_rays, n_pts_per_ray, device=device
                            ),
                        ),
                        "heterogeneous_ray_bundle": HeterogeneousRayBundle(
                            origins=torch.randn(B * n_rays, 3, device=device),
                            xys=torch.randn(B * n_rays, 2, device=device),
                            directions=torch.randn(B * n_rays, 3, device=device),
                            lengths=torch.randn(
                                B * n_rays, n_pts_per_ray, device=device
                            ),
                            camera_ids=torch.randint(
                                low=0, high=B, size=(B * n_rays,), device=device
                            ),
                        ),
                        "camera": PerspectiveCameras(
                            R=random_rotations(B, device=device),
                            T=torch.randn(B, 3, device=device),
                        ),
                        "mesh": Meshes(
                            verts=torch.randn(B, n_verts, 3, device=device),
                            faces=torch.randint(
                                low=0, high=n_verts, size=(B, n_edges, 3), device=device
                            ),
                        ),
                        "point_clouds": Pointclouds(
                            points=torch.randn(B, n_pts, 3, device=device),
                        ),
                    }
                }
            )
