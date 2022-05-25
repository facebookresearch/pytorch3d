# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test the sorting of the closest spheres."""
import logging
import os
import sys
import unittest
from os import path

import imageio
import numpy as np
import torch

from ..common_testing import TestCaseMixin

# Making sure you can run this, even if pulsar hasn't been installed yet.
sys.path.insert(0, path.join(path.dirname(__file__), "..", ".."))

devices = [torch.device("cuda"), torch.device("cpu")]
IN_REF_FP = path.join(path.dirname(__file__), "reference", "nr0000-in.pth")
OUT_REF_FP = path.join(path.dirname(__file__), "reference", "nr0000-out.pth")


class TestDepth(TestCaseMixin, unittest.TestCase):
    """Test different numbers of channels."""

    def test_basic(self):
        from pytorch3d.renderer.points.pulsar import Renderer

        for device in devices:
            gamma = 1e-5
            max_depth = 15.0
            min_depth = 5.0
            renderer = Renderer(
                256,
                256,
                10000,
                orthogonal_projection=True,
                right_handed_system=False,
                n_channels=1,
            ).to(device)
            data = torch.load(IN_REF_FP, map_location="cpu")
            # For creating the reference files.
            # Use in case of updates.
            # data["pos"] = torch.rand_like(data["pos"])
            # data["pos"][:, 0] = data["pos"][:, 0] * 2. - 1.
            # data["pos"][:, 1] = data["pos"][:, 1] * 2. - 1.
            # data["pos"][:, 2] = data["pos"][:, 2] + 9.5
            result, result_info = renderer.forward(
                data["pos"].to(device),
                data["col"].to(device),
                data["rad"].to(device),
                data["cam_params"].to(device),
                gamma,
                min_depth=min_depth,
                max_depth=max_depth,
                return_forward_info=True,
                bg_col=torch.zeros(1, device=device, dtype=torch.float32),
                percent_allowed_difference=0.01,
            )
            depth_map = Renderer.depth_map_from_result_info_nograd(result_info)
            depth_vis = (depth_map - depth_map[depth_map > 0].min()) * 200 / (
                depth_map.max() - depth_map[depth_map > 0.0].min()
            ) + 50
            if not os.environ.get("FB_TEST", False):
                imageio.imwrite(
                    path.join(
                        path.dirname(__file__),
                        "test_out",
                        "test_depth_test_basic_depth.png",
                    ),
                    depth_vis.cpu().numpy().astype(np.uint8),
                )
            # For creating the reference files.
            # Use in case of updates.
            # torch.save(
            #     data, path.join(path.dirname(__file__), "reference", "nr0000-in.pth")
            # )
            # torch.save(
            #     {"sphere_ids": sphere_ids, "depth_map": depth_map},
            #     path.join(path.dirname(__file__), "reference", "nr0000-out.pth"),
            # )
            # sys.exit(0)
            reference = torch.load(OUT_REF_FP, map_location="cpu")
            self.assertClose(reference["depth_map"].to(device), depth_map)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
