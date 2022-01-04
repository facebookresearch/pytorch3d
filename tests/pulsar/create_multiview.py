#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Create multiview data."""
import sys
from os import path


# Making sure you can run this, even if pulsar hasn't been installed yet.
sys.path.insert(0, path.join(path.dirname(__file__), "..", ".."))


def create_multiview():
    """Test multiview optimization."""
    import imageio

    # import cv2
    # import skvideo.io
    import numpy as np
    import torch
    from pytorch3d.renderer.points.pulsar import Renderer
    from torch import nn
    from torch.autograd import Variable

    # Constructor.
    n_points = 10
    width = 1000
    height = 1000

    class Model(nn.Module):
        """A dummy model to test the integration into a stacked model."""

        def __init__(self):
            super(Model, self).__init__()
            self.gamma = 0.1
            self.renderer = Renderer(width, height, n_points)

        def forward(self, vp, vc, vr, cam_params):
            # self.gamma *= 0.995
            # print("gamma: ", self.gamma)
            return self.renderer.forward(vp, vc, vr, cam_params, self.gamma, 45.0)

    # Generate sample data.
    torch.manual_seed(1)
    vert_pos = torch.rand(n_points, 3, dtype=torch.float32) * 10.0
    vert_pos[:, 2] += 25.0
    vert_pos[:, :2] -= 5.0
    # print(vert_pos[0])
    vert_col = torch.rand(n_points, 3, dtype=torch.float32)
    vert_rad = torch.rand(n_points, dtype=torch.float32)

    # Distortion.
    # vert_pos[:, 1] += 0.5
    vert_col *= 0.5
    # vert_rad *= 0.7

    for device in [torch.device("cuda")]:
        model = Model().to(device)
        vert_pos = vert_pos.to(device)
        vert_col = vert_col.to(device)
        vert_rad = vert_rad.to(device)
        for angle_idx, angle in enumerate([-1.5, -0.8, -0.4, -0.1, 0.1, 0.4, 0.8, 1.5]):
            vert_pos_v = Variable(vert_pos, requires_grad=False)
            vert_col_v = Variable(vert_col, requires_grad=False)
            vert_rad_v = Variable(vert_rad, requires_grad=False)
            cam_params = torch.tensor(
                [
                    np.sin(angle) * 35.0,
                    0.0,
                    30.0 - np.cos(angle) * 35.0,
                    0.0,
                    -angle,
                    0.0,
                    5.0,
                    2.0,
                ],
                dtype=torch.float32,
            ).to(device)
            cam_params_v = Variable(cam_params, requires_grad=False)
            result = model.forward(vert_pos_v, vert_col_v, vert_rad_v, cam_params_v)
            result_im = (result.cpu().detach().numpy() * 255).astype(np.uint8)
            imageio.imsave(
                "reference/examples_TestRenderer_test_multiview_%d.png" % (angle_idx),
                result_im,
            )


if __name__ == "__main__":
    create_multiview()
