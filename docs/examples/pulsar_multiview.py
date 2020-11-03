#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
"""
This example demonstrates multiview 3D reconstruction using the plain
pulsar interface. For this, reference images have been pre-generated
(you can find them at `../../tests/pulsar/reference/examples_TestRenderer_test_multiview_%d.png`).
The camera parameters are assumed given. The scene is initialized with
random spheres. Gradient-based optimization is used to optimize sphere
parameters and prune spheres to converge to a 3D representation.
"""
from os import path

import cv2
import imageio
import numpy as np
import torch
from pytorch3d.renderer.points.pulsar import Renderer
from torch import nn, optim


n_points = 400_000
width = 1_000
height = 1_000
visualize_ids = [0, 1]
device = torch.device("cuda")


class SceneModel(nn.Module):
    """
    A simple scene model to demonstrate use of pulsar in PyTorch modules.

    The scene model is parameterized with sphere locations (vert_pos),
    channel content (vert_col), radiuses (vert_rad), camera position (cam_pos),
    camera rotation (cam_rot) and sensor focal length and width (cam_sensor).

    The forward method of the model renders this scene description. Any
    of these parameters could instead be passed as inputs to the forward
    method and come from a different model. Optionally, camera parameters can
    be provided to the forward method in which case the scene is rendered
    using those parameters.
    """

    def __init__(self):
        super(SceneModel, self).__init__()
        self.gamma = 1.0
        # Points.
        torch.manual_seed(1)
        vert_pos = torch.rand((1, n_points, 3), dtype=torch.float32) * 10.0
        vert_pos[:, :, 2] += 25.0
        vert_pos[:, :, :2] -= 5.0
        self.register_parameter("vert_pos", nn.Parameter(vert_pos, requires_grad=True))
        self.register_parameter(
            "vert_col",
            nn.Parameter(
                torch.ones(1, n_points, 3, dtype=torch.float32) * 0.5,
                requires_grad=True,
            ),
        )
        self.register_parameter(
            "vert_rad",
            nn.Parameter(
                torch.ones(1, n_points, dtype=torch.float32) * 0.05, requires_grad=True
            ),
        )
        self.register_parameter(
            "vert_opy",
            nn.Parameter(
                torch.ones(1, n_points, dtype=torch.float32), requires_grad=True
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
        )


# Load reference.
ref = torch.stack(
    [
        torch.from_numpy(
            imageio.imread(
                "../../tests/pulsar/reference/examples_TestRenderer_test_multiview_%d.png"
                % idx
            )
        ).to(torch.float32)
        / 255.0
        for idx in range(8)
    ]
).to(device)
# Set up model.
model = SceneModel().to(device)
# Optimizer.
optimizer = optim.SGD(
    [
        {"params": [model.vert_col], "lr": 1e-1},
        {"params": [model.vert_rad], "lr": 1e-3},
        {"params": [model.vert_pos], "lr": 1e-3},
    ]
)

# For visualization.
angle = 0.0
print("Writing video to `%s`." % (path.abspath("multiview.avi")))
writer = imageio.get_writer("multiview.gif", format="gif", fps=25)

# Optimize.
for i in range(300):
    optimizer.zero_grad()
    result = model()
    # Visualize.
    result_im = (result.cpu().detach().numpy() * 255).astype(np.uint8)
    cv2.imshow("opt", result_im[0, :, :, ::-1])
    overlay_img = np.ascontiguousarray(
        ((result * 0.5 + ref * 0.5).cpu().detach().numpy() * 255).astype(np.uint8)[
            0, :, :, ::-1
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
    # Cleanup.
    with torch.no_grad():
        model.vert_col.data = torch.clamp(model.vert_col.data, 0.0, 1.0)
        # Remove points.
        model.vert_pos.data[model.vert_rad < 0.001, :] = -1000.0
        model.vert_rad.data[model.vert_rad < 0.001] = 0.0001
        vd = (
            (model.vert_col - torch.ones(1, 1, 3, dtype=torch.float32).to(device))
            .abs()
            .sum(dim=2)
        )
        model.vert_pos.data[vd <= 0.2] = -1000.0
    # Rotating visualization.
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
    with torch.no_grad():
        result = model.forward(cam=cam_control)[0]
        result_im = (result.cpu().detach().numpy() * 255).astype(np.uint8)
        cv2.imshow("vis", result_im[:, :, ::-1])
        writer.append_data(result_im)
        angle += 0.05
writer.close()
