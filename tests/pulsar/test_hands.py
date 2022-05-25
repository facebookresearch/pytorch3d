# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test right hand/left hand system compatibility."""
import logging
import sys
import unittest
from os import path

import torch

from ..common_testing import TestCaseMixin


# Making sure you can run this, even if pulsar hasn't been installed yet.
sys.path.insert(0, path.join(path.dirname(__file__), "..", ".."))
devices = [torch.device("cuda"), torch.device("cpu")]


class TestHands(TestCaseMixin, unittest.TestCase):
    """Test right hand/left hand system compatibility."""

    def test_basic(self):
        """Basic forward test."""
        from pytorch3d.renderer.points.pulsar import Renderer

        n_points = 10
        width = 1000
        height = 1000
        renderer_left = Renderer(width, height, n_points, right_handed_system=False)
        renderer_right = Renderer(width, height, n_points, right_handed_system=True)
        # Generate sample data.
        torch.manual_seed(1)
        vert_pos = torch.rand(n_points, 3, dtype=torch.float32) * 10.0
        vert_pos[:, 2] += 25.0
        vert_pos[:, :2] -= 5.0
        vert_pos_neg = vert_pos.clone()
        vert_pos_neg[:, 2] *= -1.0
        vert_col = torch.rand(n_points, 3, dtype=torch.float32)
        vert_rad = torch.rand(n_points, dtype=torch.float32)
        cam_params = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 2.0], dtype=torch.float32
        )
        for device in devices:
            vert_pos = vert_pos.to(device)
            vert_pos_neg = vert_pos_neg.to(device)
            vert_col = vert_col.to(device)
            vert_rad = vert_rad.to(device)
            cam_params = cam_params.to(device)
            renderer_left = renderer_left.to(device)
            renderer_right = renderer_right.to(device)
            result_left = (
                renderer_left.forward(
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
            hits_left = (
                renderer_left.forward(
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
            result_right = (
                renderer_right.forward(
                    vert_pos_neg,
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
            hits_right = (
                renderer_right.forward(
                    vert_pos_neg,
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
            self.assertClose(result_left, result_right)
            self.assertClose(hits_left, hits_right)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("pulsar.renderer").setLevel(logging.WARN)
    unittest.main()
