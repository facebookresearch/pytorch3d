#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
"""
This example demonstrates scene optimization with the PyTorch3D
pulsar interface. For this, a reference image has been pre-generated
(you can find it at `../../tests/pulsar/reference/examples_TestRenderer_test_smallopt.png`).
The scene is initialized with random spheres. Gradient-based
optimization is used to converge towards a faithful
scene representation.
"""
import math

import cv2
import imageio
import numpy as np
import torch
from pytorch3d.renderer.cameras import PerspectiveCameras  # , look_at_view_transform
from pytorch3d.renderer.points import (
    PointsRasterizationSettings,
    PointsRasterizer,
    PulsarPointsRenderer,
)
from pytorch3d.structures.pointclouds import Pointclouds
from torch import nn, optim


n_points = 10_000
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
        self.gamma = 1.0
        # Points.
        torch.manual_seed(1)
        vert_pos = torch.rand(n_points, 3, dtype=torch.float32, device=device) * 10.0
        vert_pos[:, 2] += 25.0
        vert_pos[:, :2] -= 5.0
        self.register_parameter("vert_pos", nn.Parameter(vert_pos, requires_grad=True))
        self.register_parameter(
            "vert_col",
            nn.Parameter(
                torch.ones(n_points, 3, dtype=torch.float32, device=device) * 0.5,
                requires_grad=True,
            ),
        )
        self.register_parameter(
            "vert_rad",
            nn.Parameter(
                torch.ones(n_points, dtype=torch.float32) * 0.3, requires_grad=True
            ),
        )
        self.register_buffer(
            "cam_params",
            torch.tensor(
                [0.0, 0.0, 0.0, 0.0, math.pi, 0.0, 5.0, 2.0], dtype=torch.float32
            ),
        )
        self.cameras = PerspectiveCameras(
            # The focal length must be double the size for PyTorch3D because of the NDC
            # coordinates spanning a range of two - and they must be normalized by the
            # sensor width (see the pulsar example). This means we need here
            # 5.0 * 2.0 / 2.0 to get the equivalent results as in pulsar.
            focal_length=5.0,
            R=torch.eye(3, dtype=torch.float32, device=device)[None, ...],
            T=torch.zeros((1, 3), dtype=torch.float32, device=device),
            image_size=((width, height),),
            device=device,
        )
        raster_settings = PointsRasterizationSettings(
            image_size=(width, height),
            radius=self.vert_rad,
        )
        rasterizer = PointsRasterizer(
            cameras=self.cameras, raster_settings=raster_settings
        )
        self.renderer = PulsarPointsRenderer(rasterizer=rasterizer, n_track=32)

    def forward(self):
        # The Pointclouds object creates copies of it's arguments - that's why
        # we have to create a new object in every forward step.
        pcl = Pointclouds(
            points=self.vert_pos[None, ...], features=self.vert_col[None, ...]
        )
        return self.renderer(
            pcl,
            gamma=(self.gamma,),
            zfar=(45.0,),
            znear=(1.0,),
            radius_world=True,
            bg_col=torch.ones((3,), dtype=torch.float32, device=device),
        )[0]


# Load reference.
ref = (
    torch.from_numpy(
        imageio.imread(
            "../../tests/pulsar/reference/examples_TestRenderer_test_smallopt.png"
        )[:, ::-1, :].copy()
    ).to(torch.float32)
    / 255.0
).to(device)
# Set up model.
model = SceneModel().to(device)
# Optimizer.
optimizer = optim.SGD(
    [
        {"params": [model.vert_col], "lr": 1e0},
        {"params": [model.vert_rad], "lr": 5e-3},
        {"params": [model.vert_pos], "lr": 1e-2},
    ]
)

# Optimize.
for i in range(500):
    optimizer.zero_grad()
    result = model()
    # Visualize.
    result_im = (result.cpu().detach().numpy() * 255).astype(np.uint8)
    cv2.imshow("opt", result_im[:, :, ::-1])
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
    # Cleanup.
    with torch.no_grad():
        model.vert_col.data = torch.clamp(model.vert_col.data, 0.0, 1.0)
        # Remove points.
        model.vert_pos.data[model.vert_rad < 0.001, :] = -1000.0
        model.vert_rad.data[model.vert_rad < 0.001] = 0.0001
        vd = (
            (model.vert_col - torch.ones(3, dtype=torch.float32).to(device))
            .abs()
            .sum(dim=1)
        )
        model.vert_pos.data[vd <= 0.2] = -1000.0
