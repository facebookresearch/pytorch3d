# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Basic rendering test."""
import logging
import os
import sys
import unittest
from os import path

import imageio
import numpy as np
import torch


# Making sure you can run this, even if pulsar hasn't been installed yet.
sys.path.insert(0, path.join(path.dirname(__file__), "..", ".."))
LOGGER = logging.getLogger(__name__)
devices = [torch.device("cuda"), torch.device("cpu")]


class TestForward(unittest.TestCase):
    """Rendering tests."""

    def test_bg_weight(self):
        """Test background reweighting."""
        from pytorch3d.renderer.points.pulsar import Renderer

        LOGGER.info("Setting up rendering test for 3 channels...")
        n_points = 1
        width = 1_000
        height = 1_000
        renderer = Renderer(width, height, n_points, background_normalized_depth=0.999)
        vert_pos = torch.tensor([[0.0, 0.0, 25.0]], dtype=torch.float32)
        vert_col = torch.tensor([[0.3, 0.5, 0.7]], dtype=torch.float32)
        vert_rad = torch.tensor([1.0], dtype=torch.float32)
        cam_params = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 2.0], dtype=torch.float32
        )
        for device in devices:
            vert_pos = vert_pos.to(device)
            vert_col = vert_col.to(device)
            vert_rad = vert_rad.to(device)
            cam_params = cam_params.to(device)
            renderer = renderer.to(device)
            LOGGER.info("Rendering...")
            # Measurements.
            result = renderer.forward(
                vert_pos, vert_col, vert_rad, cam_params, 1.0e-1, 45.0
            )
            hits = renderer.forward(
                vert_pos,
                vert_col,
                vert_rad,
                cam_params,
                1.0e-1,
                45.0,
                percent_allowed_difference=0.01,
                mode=1,
            )
            if not os.environ.get("FB_TEST", False):
                imageio.imsave(
                    path.join(
                        path.dirname(__file__),
                        "test_out",
                        "test_forward_TestForward_test_bg_weight.png",
                    ),
                    (result * 255.0).cpu().to(torch.uint8).numpy(),
                )
                imageio.imsave(
                    path.join(
                        path.dirname(__file__),
                        "test_out",
                        "test_forward_TestForward_test_bg_weight_hits.png",
                    ),
                    (hits * 255.0).cpu().to(torch.uint8).numpy(),
                )
            self.assertEqual(hits[500, 500, 0].item(), 1.0)
            self.assertTrue(
                np.allclose(
                    result[500, 500, :].cpu().numpy(),
                    [1.0, 1.0, 1.0],
                    rtol=1e-2,
                    atol=1e-2,
                )
            )

    def test_basic_3chan(self):
        """Test rendering one image with one sphere, 3 channels."""
        from pytorch3d.renderer.points.pulsar import Renderer

        LOGGER.info("Setting up rendering test for 3 channels...")
        n_points = 1
        width = 1_000
        height = 1_000
        renderer = Renderer(width, height, n_points)
        vert_pos = torch.tensor([[0.0, 0.0, 25.0]], dtype=torch.float32)
        vert_col = torch.tensor([[0.3, 0.5, 0.7]], dtype=torch.float32)
        vert_rad = torch.tensor([1.0], dtype=torch.float32)
        cam_params = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 2.0], dtype=torch.float32
        )
        for device in devices:
            vert_pos = vert_pos.to(device)
            vert_col = vert_col.to(device)
            vert_rad = vert_rad.to(device)
            cam_params = cam_params.to(device)
            renderer = renderer.to(device)
            LOGGER.info("Rendering...")
            # Measurements.
            result = renderer.forward(
                vert_pos, vert_col, vert_rad, cam_params, 1.0e-1, 45.0
            )
            hits = renderer.forward(
                vert_pos,
                vert_col,
                vert_rad,
                cam_params,
                1.0e-1,
                45.0,
                percent_allowed_difference=0.01,
                mode=1,
            )
            if not os.environ.get("FB_TEST", False):
                imageio.imsave(
                    path.join(
                        path.dirname(__file__),
                        "test_out",
                        "test_forward_TestForward_test_basic_3chan.png",
                    ),
                    (result * 255.0).cpu().to(torch.uint8).numpy(),
                )
                imageio.imsave(
                    path.join(
                        path.dirname(__file__),
                        "test_out",
                        "test_forward_TestForward_test_basic_3chan_hits.png",
                    ),
                    (hits * 255.0).cpu().to(torch.uint8).numpy(),
                )
            self.assertEqual(hits[500, 500, 0].item(), 1.0)
            self.assertTrue(
                np.allclose(
                    result[500, 500, :].cpu().numpy(),
                    [0.3, 0.5, 0.7],
                    rtol=1e-2,
                    atol=1e-2,
                )
            )

    def test_basic_1chan(self):
        """Test rendering one image with one sphere, 1 channel."""
        from pytorch3d.renderer.points.pulsar import Renderer

        LOGGER.info("Setting up rendering test for 1 channel...")
        n_points = 1
        width = 1_000
        height = 1_000
        renderer = Renderer(width, height, n_points, n_channels=1)
        vert_pos = torch.tensor([[0.0, 0.0, 25.0]], dtype=torch.float32)
        vert_col = torch.tensor([[0.3]], dtype=torch.float32)
        vert_rad = torch.tensor([1.0], dtype=torch.float32)
        cam_params = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 2.0], dtype=torch.float32
        )
        for device in devices:
            vert_pos = vert_pos.to(device)
            vert_col = vert_col.to(device)
            vert_rad = vert_rad.to(device)
            cam_params = cam_params.to(device)
            renderer = renderer.to(device)
            LOGGER.info("Rendering...")
            # Measurements.
            result = renderer.forward(
                vert_pos, vert_col, vert_rad, cam_params, 1.0e-1, 45.0
            )
            hits = renderer.forward(
                vert_pos,
                vert_col,
                vert_rad,
                cam_params,
                1.0e-1,
                45.0,
                percent_allowed_difference=0.01,
                mode=1,
            )
            if not os.environ.get("FB_TEST", False):
                imageio.imsave(
                    path.join(
                        path.dirname(__file__),
                        "test_out",
                        "test_forward_TestForward_test_basic_1chan.png",
                    ),
                    (result * 255.0).cpu().to(torch.uint8).numpy(),
                )
                imageio.imsave(
                    path.join(
                        path.dirname(__file__),
                        "test_out",
                        "test_forward_TestForward_test_basic_1chan_hits.png",
                    ),
                    (hits * 255.0).cpu().to(torch.uint8).numpy(),
                )
            self.assertEqual(hits[500, 500, 0].item(), 1.0)
            self.assertTrue(
                np.allclose(
                    result[500, 500, :].cpu().numpy(), [0.3], rtol=1e-2, atol=1e-2
                )
            )

    def test_basic_8chan(self):
        """Test rendering one image with one sphere, 8 channels."""
        from pytorch3d.renderer.points.pulsar import Renderer

        LOGGER.info("Setting up rendering test for 8 channels...")
        n_points = 1
        width = 1_000
        height = 1_000
        renderer = Renderer(width, height, n_points, n_channels=8)
        vert_pos = torch.tensor([[0.0, 0.0, 25.0]], dtype=torch.float32)
        vert_col = torch.tensor(
            [[1.0, 1.0, 1.0, 1.0, 1.0, 0.3, 0.5, 0.7]], dtype=torch.float32
        )
        vert_rad = torch.tensor([1.0], dtype=torch.float32)
        cam_params = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 2.0], dtype=torch.float32
        )
        for device in devices:
            vert_pos = vert_pos.to(device)
            vert_col = vert_col.to(device)
            vert_rad = vert_rad.to(device)
            cam_params = cam_params.to(device)
            renderer = renderer.to(device)
            LOGGER.info("Rendering...")
            # Measurements.
            result = renderer.forward(
                vert_pos, vert_col, vert_rad, cam_params, 1.0e-1, 45.0
            )
            hits = renderer.forward(
                vert_pos,
                vert_col,
                vert_rad,
                cam_params,
                1.0e-1,
                45.0,
                percent_allowed_difference=0.01,
                mode=1,
            )
            if not os.environ.get("FB_TEST", False):
                imageio.imsave(
                    path.join(
                        path.dirname(__file__),
                        "test_out",
                        "test_forward_TestForward_test_basic_8chan.png",
                    ),
                    (result[:, :, 5:8] * 255.0).cpu().to(torch.uint8).numpy(),
                )
                imageio.imsave(
                    path.join(
                        path.dirname(__file__),
                        "test_out",
                        "test_forward_TestForward_test_basic_8chan_hits.png",
                    ),
                    (hits * 255.0).cpu().to(torch.uint8).numpy(),
                )
            self.assertEqual(hits[500, 500, 0].item(), 1.0)
            self.assertTrue(
                np.allclose(
                    result[500, 500, 5:8].cpu().numpy(),
                    [0.3, 0.5, 0.7],
                    rtol=1e-2,
                    atol=1e-2,
                )
            )
            self.assertTrue(
                np.allclose(
                    result[500, 500, :5].cpu().numpy(), 1.0, rtol=1e-2, atol=1e-2
                )
            )

    def test_principal_point(self):
        """Test shifting the principal point."""
        from pytorch3d.renderer.points.pulsar import Renderer

        LOGGER.info("Setting up rendering test for shifted principal point...")
        n_points = 1
        width = 1_000
        height = 1_000
        renderer = Renderer(width, height, n_points, n_channels=1)
        vert_pos = torch.tensor([[0.0, 0.0, 25.0]], dtype=torch.float32)
        vert_col = torch.tensor([[0.0]], dtype=torch.float32)
        vert_rad = torch.tensor([1.0], dtype=torch.float32)
        cam_params = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 2.0, 0.0, 0.0], dtype=torch.float32
        )
        for device in devices:
            vert_pos = vert_pos.to(device)
            vert_col = vert_col.to(device)
            vert_rad = vert_rad.to(device)
            cam_params = cam_params.to(device)
            cam_params[-2] = -250.0
            cam_params[-1] = -250.0
            renderer = renderer.to(device)
            LOGGER.info("Rendering...")
            # Measurements.
            result = renderer.forward(
                vert_pos, vert_col, vert_rad, cam_params, 1.0e-1, 45.0
            )
            if not os.environ.get("FB_TEST", False):
                imageio.imsave(
                    path.join(
                        path.dirname(__file__),
                        "test_out",
                        "test_forward_TestForward_test_principal_point.png",
                    ),
                    (result * 255.0).cpu().to(torch.uint8).numpy(),
                )
            self.assertTrue(
                np.allclose(
                    result[750, 750, :].cpu().numpy(), [0.0], rtol=1e-2, atol=1e-2
                )
            )
        for device in devices:
            vert_pos = vert_pos.to(device)
            vert_col = vert_col.to(device)
            vert_rad = vert_rad.to(device)
            cam_params = cam_params.to(device)
            cam_params[-2] = 250.0
            cam_params[-1] = 250.0
            renderer = renderer.to(device)
            LOGGER.info("Rendering...")
            # Measurements.
            result = renderer.forward(
                vert_pos, vert_col, vert_rad, cam_params, 1.0e-1, 45.0
            )
            if not os.environ.get("FB_TEST", False):
                imageio.imsave(
                    path.join(
                        path.dirname(__file__),
                        "test_out",
                        "test_forward_TestForward_test_principal_point.png",
                    ),
                    (result * 255.0).cpu().to(torch.uint8).numpy(),
                )
            self.assertTrue(
                np.allclose(
                    result[250, 250, :].cpu().numpy(), [0.0], rtol=1e-2, atol=1e-2
                )
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("pulsar.renderer").setLevel(logging.WARN)
    unittest.main()
