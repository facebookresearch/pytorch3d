#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
"""
This example demonstrates OpenCV camera parameter with the plain
pulsar interface.
"""
import logging

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytorch3d.renderer.points.pulsar as pulsar
import torch
from pytorch3d.renderer.points.pulsar.camera import opencv2pulsar


matplotlib.use("Agg")


def cli():
    """
    Basic example for the OpenCV-to-Pulsar conversion.

    Writes to `opencv2pulsar.png`.
    """
    # ~~~~~~~~~~~~~~~~~
    # sample 3d points
    points3d = np.array(
        [
            [0.128826, -0.347764, 1.62346],
            [0.136779, -0.197784, 1.833705],
            [-0.038932, -0.189967, 1.830946],
            [-0.084399, 0.105825, 1.878489],
            [-0.082497, 0.358484, 1.809373],
            [0.310953, -0.203041, 1.828439],
            [0.363599, 0.086033, 1.858132],
            [0.347989, 0.34087, 1.802693],
            [0.136886, 0.3853, 1.835586],
        ]
    )
    n_spheres = len(points3d)

    # ~~~~~~~~~~~~~~~~~
    # camera params
    zfar = 10.0
    znear = 0.1
    h = 1024
    w = 1024
    f = 1127.64
    cx = 516.12
    cy = 510.58

    K = np.eye(3)
    K[0, 2] = cx
    K[1, 2] = cy
    K[0, 0] = f
    K[1, 1] = f

    rvec = np.array(
        [[-0.051111404817219305, -2.6377198366878027, -0.28602826581257784]]
    )
    C = np.array([[-0.482771, -0.400003, 3.479192]]).transpose()

    R = cv2.Rodrigues(rvec)[0]
    tvec = -R @ C

    # ~~~~~~~~~~~~~~~~~
    # OpenCV projection
    distCoef = np.zeros((5,))
    points2d_opencv, _ = cv2.projectPoints(points3d, rvec, tvec, K, distCoef)
    points2d_opencv = np.squeeze(points2d_opencv)

    # ~~~~~~~~~~~~~~~~~
    # Pulsar projection
    cam_params = opencv2pulsar(K, R, tvec, h, w)

    # We're working with a default left handed system here.
    renderer = pulsar.Renderer(w, h, n_spheres, right_handed_system=False)

    pos = torch.from_numpy(points3d).float().cpu()

    col = torch.zeros((n_spheres, 3)).cpu().float()
    col[:, 0] = 1.0
    rad = torch.ones((n_spheres,)).cpu().float() * 0.02
    image_pulsar = renderer(
        pos,
        col,
        rad,
        cam_params,
        1.0e-1,  # Renderer blending parameter gamma, in [1., 1e-5].
        max_depth=zfar,  # Maximum depth.
        min_depth=znear,
    )
    image_pulsar = (image_pulsar.cpu().numpy() * 255).astype("uint8")

    # Flip the image horizontal
    image_pulsar = image_pulsar[::-1, :, :]

    # ~~~~~~~~~~~~~~~~~
    # Plotting to Figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim([0, w])
    ax.set_ylim([h, 0])
    ax.imshow(image_pulsar)
    ax.scatter(points2d_opencv[:, 0], points2d_opencv[:, 1], color="blue", alpha=0.5)

    plt.tight_layout()
    plt.savefig("opencv2pulsar.png")
    plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()
