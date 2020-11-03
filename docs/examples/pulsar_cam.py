#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
"""
This example demonstrates camera parameter optimization with the plain
pulsar interface. For this, a reference image has been pre-generated
(you can find it at `../../tests/pulsar/reference/examples_TestRenderer_test_cam.png`).
The same scene parameterization is loaded and the camera parameters
distorted. Gradient-based optimization is used to converge towards the
original camera parameters.
"""
from os import path

import cv2
import imageio
import numpy as np
import torch
from pytorch3d.renderer.points.pulsar import Renderer
from torch import nn, optim


n_points = 20
width = 1_000
height = 1_000
device = torch.device("cuda")


class SceneModel(nn.Module):
    """
    A simple scene model to demonstrate use of pulsar in PyTorch modules.

    The scene model is parameterized with sphere locations (vert_pos),
    channel content (vert_col), radiuses (vert_rad), camera position (cam_pos),
    camera rotation (cam_rot) and sensor focal length and width (cam_sensor).

    The forward method of the model renders this scene description. Any
    of these parameters could instead be passed as inputs to the forward
    method and come from a different model.
    """

    def __init__(self):
        super(SceneModel, self).__init__()
        self.gamma = 0.1
        # Points.
        torch.manual_seed(1)
        vert_pos = torch.rand(n_points, 3, dtype=torch.float32) * 10.0
        vert_pos[:, 2] += 25.0
        vert_pos[:, :2] -= 5.0
        self.register_parameter("vert_pos", nn.Parameter(vert_pos, requires_grad=False))
        self.register_parameter(
            "vert_col",
            nn.Parameter(
                torch.rand(n_points, 3, dtype=torch.float32), requires_grad=False
            ),
        )
        self.register_parameter(
            "vert_rad",
            nn.Parameter(
                torch.rand(n_points, dtype=torch.float32), requires_grad=False
            ),
        )
        self.register_parameter(
            "cam_pos",
            nn.Parameter(
                torch.tensor([0.1, 0.1, 0.0], dtype=torch.float32), requires_grad=True
            ),
        )
        self.register_parameter(
            "cam_rot",
            nn.Parameter(
                torch.tensor(
                    [
                        # We're using the 6D rot. representation for better gradients.
                        0.9995,
                        0.0300445,
                        -0.0098482,
                        -0.0299445,
                        0.9995,
                        0.0101482,
                    ],
                    dtype=torch.float32,
                ),
                requires_grad=True,
            ),
        )
        self.register_parameter(
            "cam_sensor",
            nn.Parameter(
                torch.tensor([4.8, 1.8], dtype=torch.float32), requires_grad=True
            ),
        )
        self.renderer = Renderer(width, height, n_points)

    def forward(self):
        return self.renderer.forward(
            self.vert_pos,
            self.vert_col,
            self.vert_rad,
            torch.cat([self.cam_pos, self.cam_rot, self.cam_sensor]),
            self.gamma,
            45.0,
        )


# Load reference.
ref = (
    torch.from_numpy(
        imageio.imread(
            "../../tests/pulsar/reference/examples_TestRenderer_test_cam.png"
        )
    ).to(torch.float32)
    / 255.0
).to(device)
# Set up model.
model = SceneModel().to(device)
# Optimizer.
optimizer = optim.SGD(
    [
        {"params": [model.cam_pos], "lr": 1e-4},  # 1e-3
        {"params": [model.cam_rot], "lr": 5e-6},
        {"params": [model.cam_sensor], "lr": 1e-4},
    ]
)

print("Writing video to `%s`." % (path.abspath("cam.gif")))
writer = imageio.get_writer("cam.gif", format="gif", fps=25)

# Optimize.
for i in range(300):
    optimizer.zero_grad()
    result = model()
    # Visualize.
    result_im = (result.cpu().detach().numpy() * 255).astype(np.uint8)
    cv2.imshow("opt", result_im[:, :, ::-1])
    writer.append_data(result_im)
    overlay_img = np.ascontiguousarray(
        ((result * 0.5 + ref * 0.5).cpu().detach().numpy() * 255).astype(np.uint8)[
            :, :, ::-1
        ]
    )
    overlay_img = cv2.putText(
        overlay_img,
        "Step %d" % (i),
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
        False,
    )
    cv2.imshow("overlay", overlay_img)
    cv2.waitKey(1)
    # Update.
    loss = ((result - ref) ** 2).sum()
    print("loss {}: {}".format(i, loss.item()))
    loss.backward()
    optimizer.step()
writer.close()
