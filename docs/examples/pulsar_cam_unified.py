#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This example demonstrates camera parameter optimization with the pulsar
PyTorch3D interface. For this, a reference image has been pre-generated
(you can find it at `../../tests/pulsar/reference/examples_TestRenderer_test_cam.png`).
The same scene parameterization is loaded and the camera parameters
distorted. Gradient-based optimization is used to converge towards the
original camera parameters.
Output: cam-pt3d.gif
"""
import logging
from os import path

import cv2
import imageio
import numpy as np
import torch
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.renderer.points import (
    PointsRasterizationSettings,
    PointsRasterizer,
    PulsarPointsRenderer,
)
from pytorch3d.structures.pointclouds import Pointclouds
from pytorch3d.transforms import axis_angle_to_matrix
from torch import nn, optim


LOGGER = logging.getLogger(__name__)
N_POINTS = 20
WIDTH = 1_000
HEIGHT = 1_000
DEVICE = torch.device("cuda")


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
        vert_pos = torch.rand(N_POINTS, 3, dtype=torch.float32) * 10.0
        vert_pos[:, 2] += 25.0
        vert_pos[:, :2] -= 5.0
        self.register_parameter("vert_pos", nn.Parameter(vert_pos, requires_grad=False))
        self.register_parameter(
            "vert_col",
            nn.Parameter(
                torch.rand(N_POINTS, 3, dtype=torch.float32),
                requires_grad=False,
            ),
        )
        self.register_parameter(
            "vert_rad",
            nn.Parameter(
                torch.rand(N_POINTS, dtype=torch.float32),
                requires_grad=False,
            ),
        )
        self.register_parameter(
            "cam_pos",
            nn.Parameter(
                torch.tensor([0.1, 0.1, 0.0], dtype=torch.float32),
                requires_grad=True,
            ),
        )
        self.register_parameter(
            "cam_rot",
            # We're using the 6D rot. representation for better gradients.
            nn.Parameter(
                axis_angle_to_matrix(
                    torch.tensor(
                        [
                            [0.02, 0.02, 0.01],
                        ],
                        dtype=torch.float32,
                    )
                )[0],
                requires_grad=True,
            ),
        )
        self.register_parameter(
            "focal_length",
            nn.Parameter(
                torch.tensor(
                    [
                        4.8 * 2.0 / 2.0,
                    ],
                    dtype=torch.float32,
                ),
                requires_grad=True,
            ),
        )
        self.cameras = PerspectiveCameras(
            # The focal length must be double the size for PyTorch3D because of the NDC
            # coordinates spanning a range of two - and they must be normalized by the
            # sensor width (see the pulsar example). This means we need here
            # 5.0 * 2.0 / 2.0 to get the equivalent results as in pulsar.
            #
            # R, T and f are provided here, but will be provided again
            # at every call to the forward method. The reason are problems
            # with PyTorch which makes device placement for gradients problematic
            # for tensors which are themselves on a 'gradient path' but not
            # leafs in the calculation tree. This will be addressed by an architectural
            # change in PyTorch3D in the future. Until then, this workaround is
            # recommended.
            focal_length=self.focal_length,
            R=self.cam_rot[None, ...],
            T=self.cam_pos[None, ...],
            image_size=((HEIGHT, WIDTH),),
            device=DEVICE,
        )
        raster_settings = PointsRasterizationSettings(
            image_size=(HEIGHT, WIDTH),
            radius=self.vert_rad,
        )
        rasterizer = PointsRasterizer(
            cameras=self.cameras, raster_settings=raster_settings
        )
        self.renderer = PulsarPointsRenderer(rasterizer=rasterizer)

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
            bg_col=torch.ones((3,), dtype=torch.float32, device=DEVICE),
            # As mentioned above: workaround for device placement of gradients for
            # camera parameters.
            focal_length=self.focal_length,
            R=self.cam_rot[None, ...],
            T=self.cam_pos[None, ...],
        )[0]


def cli():
    """
    Camera optimization example using pulsar.

    Writes to `cam.gif`.
    """
    LOGGER.info("Loading reference...")
    # Load reference.
    ref = (
        torch.from_numpy(
            imageio.imread(
                "../../tests/pulsar/reference/examples_TestRenderer_test_cam.png"
            )[:, ::-1, :].copy()
        ).to(torch.float32)
        / 255.0
    ).to(DEVICE)
    # Set up model.
    model = SceneModel().to(DEVICE)
    # Optimizer.
    optimizer = optim.SGD(
        [
            {"params": [model.cam_pos], "lr": 1e-4},
            {"params": [model.cam_rot], "lr": 5e-6},
            # Using a higher lr for the focal length here, because
            # the sensor width can not be optimized directly.
            {"params": [model.focal_length], "lr": 1e-3},
        ]
    )

    LOGGER.info("Writing video to `%s`.", path.abspath("cam-pt3d.gif"))
    writer = imageio.get_writer("cam-pt3d.gif", format="gif", fps=25)

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
        LOGGER.info("loss %d: %f", i, loss.item())
        loss.backward()
        optimizer.step()
    writer.close()
    LOGGER.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()
