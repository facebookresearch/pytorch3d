# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test number of channels."""
import logging
import sys
import unittest
from os import path

import torch

from ..common_testing import TestCaseMixin


sys.path.insert(0, path.join(path.dirname(__file__), "..", ".."))
devices = [torch.device("cuda"), torch.device("cpu")]


class TestChannels(TestCaseMixin, unittest.TestCase):
    """Test different numbers of channels."""

    def test_basic(self):
        """Basic forward test."""
        import torch
        from pytorch3d.renderer.points.pulsar import Renderer

        n_points = 10
        width = 1_000
        height = 1_000
        renderer_1 = Renderer(width, height, n_points, n_channels=1)
        renderer_3 = Renderer(width, height, n_points, n_channels=3)
        renderer_8 = Renderer(width, height, n_points, n_channels=8)
        # Generate sample data.
        torch.manual_seed(1)
        vert_pos = torch.rand(n_points, 3, dtype=torch.float32) * 10.0
        vert_pos[:, 2] += 25.0
        vert_pos[:, :2] -= 5.0
        vert_col = torch.rand(n_points, 8, dtype=torch.float32)
        vert_rad = torch.rand(n_points, dtype=torch.float32)
        cam_params = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 2.0], dtype=torch.float32
        )
        for device in devices:
            vert_pos = vert_pos.to(device)
            vert_col = vert_col.to(device)
            vert_rad = vert_rad.to(device)
            cam_params = cam_params.to(device)
            renderer_1 = renderer_1.to(device)
            renderer_3 = renderer_3.to(device)
            renderer_8 = renderer_8.to(device)
            result_1 = (
                renderer_1.forward(
                    vert_pos,
                    vert_col[:, :1],
                    vert_rad,
                    cam_params,
                    1.0e-1,
                    45.0,
                    percent_allowed_difference=0.01,
                )
                .cpu()
                .detach()
                .numpy()
            )
            hits_1 = (
                renderer_1.forward(
                    vert_pos,
                    vert_col[:, :1],
                    vert_rad,
                    cam_params,
                    1.0e-1,
                    45.0,
                    percent_allowed_difference=0.01,
                    mode=1,
                )
                .cpu()
                .detach()
                .numpy()
            )
            result_3 = (
                renderer_3.forward(
                    vert_pos,
                    vert_col[:, :3],
                    vert_rad,
                    cam_params,
                    1.0e-1,
                    45.0,
                    percent_allowed_difference=0.01,
                )
                .cpu()
                .detach()
                .numpy()
            )
            hits_3 = (
                renderer_3.forward(
                    vert_pos,
                    vert_col[:, :3],
                    vert_rad,
                    cam_params,
                    1.0e-1,
                    45.0,
                    percent_allowed_difference=0.01,
                    mode=1,
                )
                .cpu()
                .detach()
                .numpy()
            )
            result_8 = (
                renderer_8.forward(
                    vert_pos,
                    vert_col,
                    vert_rad,
                    cam_params,
                    1.0e-1,
                    45.0,
                    percent_allowed_difference=0.01,
                )
                .cpu()
                .detach()
                .numpy()
            )
            hits_8 = (
                renderer_8.forward(
                    vert_pos,
                    vert_col,
                    vert_rad,
                    cam_params,
                    1.0e-1,
                    45.0,
                    percent_allowed_difference=0.01,
                    mode=1,
                )
                .cpu()
                .detach()
                .numpy()
            )
            self.assertClose(result_1, result_3[:, :, :1])
            self.assertClose(result_3, result_8[:, :, :3])
            self.assertClose(hits_1, hits_3)
            self.assertClose(hits_8, hits_3)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
