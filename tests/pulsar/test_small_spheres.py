#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test right hand/left hand system compatibility."""
import sys
import unittest
from os import path

import numpy as np
import torch
from torch import nn


sys.path.insert(0, path.join(path.dirname(__file__), ".."))
devices = [torch.device("cuda"), torch.device("cpu")]


n_points = 10
width = 1_000
height = 1_000


class SceneModel(nn.Module):
    """A simple model to demonstrate use in Modules."""

    def __init__(self):
        super(SceneModel, self).__init__()
        from pytorch3d.renderer.points.pulsar import Renderer

        self.gamma = 1.0
        # Points.
        torch.manual_seed(1)
        vert_pos = torch.rand((1, n_points, 3), dtype=torch.float32) * 10.0
        vert_pos[:, :, 2] += 25.0
        vert_pos[:, :, :2] -= 5.0
        self.register_parameter("vert_pos", nn.Parameter(vert_pos, requires_grad=False))
        self.register_parameter(
            "vert_col",
            nn.Parameter(
                torch.zeros(1, n_points, 3, dtype=torch.float32), requires_grad=True
            ),
        )
        self.register_parameter(
            "vert_rad",
            nn.Parameter(
                torch.ones(1, n_points, dtype=torch.float32) * 0.001,
                requires_grad=False,
            ),
        )
        self.register_parameter(
            "vert_opy",
            nn.Parameter(
                torch.ones(1, n_points, dtype=torch.float32), requires_grad=False
            ),
        )
        self.register_buffer(
            "cam_params",
            torch.tensor(
                [
                    [
                        np.sin(angle) * 35.0,
                        0.0,
                        30.0 - np.cos(angle) * 35.0,
                        0.0,
                        -angle,
                        0.0,
                        5.0,
                        2.0,
                    ]
                    for angle in [-1.5, -0.8, -0.4, -0.1, 0.1, 0.4, 0.8, 1.5]
                ],
                dtype=torch.float32,
            ),
        )
        self.renderer = Renderer(width, height, n_points)

    def forward(self, cam=None):
        if cam is None:
            cam = self.cam_params
            n_views = 8
        else:
            n_views = 1
        return self.renderer.forward(
            self.vert_pos.expand(n_views, -1, -1),
            self.vert_col.expand(n_views, -1, -1),
            self.vert_rad.expand(n_views, -1),
            cam,
            self.gamma,
            45.0,
            return_forward_info=True,
        )


class TestSmallSpheres(unittest.TestCase):
    """Test small sphere rendering and gradients."""

    def test_basic(self):
        for device in devices:
            # Set up model.
            model = SceneModel().to(device)
            angle = 0.0
            for _ in range(50):
                cam_control = torch.tensor(
                    [
                        [
                            np.sin(angle) * 35.0,
                            0.0,
                            30.0 - np.cos(angle) * 35.0,
                            0.0,
                            -angle,
                            0.0,
                            5.0,
                            2.0,
                        ]
                    ],
                    dtype=torch.float32,
                ).to(device)
                result, forw_info = model(cam=cam_control)
                sphere_ids = model.renderer.sphere_ids_from_result_info_nograd(
                    forw_info
                )
                # Assert all spheres are rendered.
                for idx in range(n_points):
                    self.assertTrue(
                        (sphere_ids == idx).sum() > 0, "Sphere ID %d missing!" % (idx)
                    )
                # Visualization code. Activate for debugging.
                # result_im = (result.cpu().detach().numpy() * 255).astype(np.uint8)
                # cv2.imshow("res", result_im[0, :, :, ::-1])
                # cv2.waitKey(0)
                # Back-propagate some dummy gradients.
                loss = ((result - torch.ones_like(result)).abs()).sum()
                loss.backward()
                # Now check whether the gradient arrives at every sphere.
                self.assertTrue(torch.all(model.vert_col.grad[:, :, 0].abs() > 0.0))
                angle += 0.15


if __name__ == "__main__":
    unittest.main()
