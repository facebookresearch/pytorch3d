# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch

from ..renderer import PerspectiveCameras
from ..transforms import so3_exp_map, so3_log_map


def cameras_from_opencv_projection(
    rvec: torch.Tensor,
    tvec: torch.Tensor,
    camera_matrix: torch.Tensor,
    image_size: torch.Tensor,
) -> PerspectiveCameras:
    """
    Converts a batch of OpenCV-conventioned cameras parametrized with the
    axis-angle rotation vectors `rvec`, translation vectors `tvec`, and the camera
    calibration matrices `camera_matrix` to `PerspectiveCameras` in PyTorch3D
    convention.

    More specifically, the conversion is carried out such that a projection
    of a 3D shape to the OpenCV-conventioned screen of size `image_size` results
    in the same image as a projection with the corresponding PyTorch3D camera
    to the NDC screen convention of PyTorch3D.

    More specifically, the OpenCV convention projects points to the OpenCV screen
    space as follows:
        ```
        x_screen_opencv = camera_matrix @ (exp(rvec) @ x_world + tvec)
        ```
    followed by the homogenization of `x_screen_opencv`.

    Note:
        The parameters `rvec, tvec, camera_matrix` correspond, e.g., to the inputs
        of `cv2.projectPoints`, or to the ouputs of `cv2.calibrateCamera`.

    Args:
        rvec: A batch of axis-angle rotation vectors of shape `(N, 3)`.
        tvec: A batch of translation vectors of shape `(N, 3)`.
        camera_matrix: A batch of camera calibration matrices of shape `(N, 3, 3)`.
        image_size: A tensor of shape `(N, 2)` containing the sizes of the images
            (height, width) attached to each camera.

    Returns:
        cameras_pytorch3d: A batch of `N` cameras in the PyTorch3D convention.
    """

    R = so3_exp_map(rvec)
    focal_length = torch.stack([camera_matrix[:, 0, 0], camera_matrix[:, 1, 1]], dim=-1)
    principal_point = camera_matrix[:, :2, 2]

    # Retype the image_size correctly and flip to width, height.
    image_size_wh = image_size.to(R).flip(dims=(1,))

    # Get the PyTorch3D focal length and principal point.
    focal_pytorch3d = focal_length / (0.5 * image_size_wh)
    p0_pytorch3d = -(principal_point / (0.5 * image_size_wh) - 1)

    # For R, T we flip x, y axes (opencv screen space has an opposite
    # orientation of screen axes).
    # We also transpose R (opencv multiplies points from the opposite=left side).
    R_pytorch3d = R.permute(0, 2, 1)
    T_pytorch3d = tvec.clone()
    R_pytorch3d[:, :, :2] *= -1
    T_pytorch3d[:, :2] *= -1

    return PerspectiveCameras(
        R=R_pytorch3d,
        T=T_pytorch3d,
        focal_length=focal_pytorch3d,
        principal_point=p0_pytorch3d,
    )


def opencv_from_cameras_projection(
    cameras: PerspectiveCameras,
    image_size: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts a batch of `PerspectiveCameras` into OpenCV-convention
    axis-angle rotation vectors `rvec`, translation vectors `tvec`, and the camera
    calibration matrices `camera_matrix`. This operation is exactly the inverse
    of `cameras_from_opencv_projection`.

    Note:
        The parameters `rvec, tvec, camera_matrix` correspond, e.g., to the inputs
        of `cv2.projectPoints`, or to the ouputs of `cv2.calibrateCamera`.

    Args:
        cameras: A batch of `N` cameras in the PyTorch3D convention.
        image_size: A tensor of shape `(N, 2)` containing the sizes of the images
            (height, width) attached to each camera.

    Returns:
        rvec: A batch of axis-angle rotation vectors of shape `(N, 3)`.
        tvec: A batch of translation vectors of shape `(N, 3)`.
        camera_matrix: A batch of camera calibration matrices of shape `(N, 3, 3)`.
    """
    R_pytorch3d = cameras.R
    T_pytorch3d = cameras.T
    focal_pytorch3d = cameras.focal_length
    p0_pytorch3d = cameras.principal_point
    T_pytorch3d[:, :2] *= -1  # pyre-ignore
    R_pytorch3d[:, :, :2] *= -1  # pyre-ignore
    tvec = T_pytorch3d.clone()  # pyre-ignore
    R = R_pytorch3d.permute(0, 2, 1)  # pyre-ignore

    # Retype the image_size correctly and flip to width, height.
    image_size_wh = image_size.to(R).flip(dims=(1,))

    principal_point = (-p0_pytorch3d + 1.0) * (0.5 * image_size_wh)  # pyre-ignore
    focal_length = focal_pytorch3d * (0.5 * image_size_wh)

    camera_matrix = torch.zeros_like(R)
    camera_matrix[:, :2, 2] = principal_point
    camera_matrix[:, 2, 2] = 1.0
    camera_matrix[:, 0, 0] = focal_length[:, 0]
    camera_matrix[:, 1, 1] = focal_length[:, 1]
    rvec = so3_log_map(R)
    return rvec, tvec, camera_matrix
