# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Tuple

import torch

from ..renderer import PerspectiveCameras
from ..renderer.camera_conversions import (
    _cameras_from_opencv_projection,
    _opencv_from_cameras_projection,
    _pulsar_from_cameras_projection,
    _pulsar_from_opencv_projection,
)


def cameras_from_opencv_projection(
    R: torch.Tensor,
    tvec: torch.Tensor,
    camera_matrix: torch.Tensor,
    image_size: torch.Tensor,
) -> PerspectiveCameras:
    """
    Converts a batch of OpenCV-conventioned cameras parametrized with the
    rotation matrices `R`, translation vectors `tvec`, and the camera
    calibration matrices `camera_matrix` to `PerspectiveCameras` in PyTorch3D
    convention.

    More specifically, the conversion is carried out such that a projection
    of a 3D shape to the OpenCV-conventioned screen of size `image_size` results
    in the same image as a projection with the corresponding PyTorch3D camera
    to the NDC screen convention of PyTorch3D.

    More specifically, the OpenCV convention projects points to the OpenCV screen
    space as follows::

        x_screen_opencv = camera_matrix @ (R @ x_world + tvec)

    followed by the homogenization of `x_screen_opencv`.

    Note:
        The parameters `R, tvec, camera_matrix` correspond to the inputs of
        `cv2.projectPoints(x_world, rvec, tvec, camera_matrix, [])`,
        where `rvec` is an axis-angle vector that can be obtained from
        the rotation matrix `R` expected here by calling the `so3_log_map` function.
        Correspondingly, `R` can be obtained from `rvec` by calling `so3_exp_map`.

    Args:
        R: A batch of rotation matrices of shape `(N, 3, 3)`.
        tvec: A batch of translation vectors of shape `(N, 3)`.
        camera_matrix: A batch of camera calibration matrices of shape `(N, 3, 3)`.
        image_size: A tensor of shape `(N, 2)` containing the sizes of the images
            (height, width) attached to each camera.

    Returns:
        cameras_pytorch3d: A batch of `N` cameras in the PyTorch3D convention.
    """
    return _cameras_from_opencv_projection(R, tvec, camera_matrix, image_size)


def opencv_from_cameras_projection(
    cameras: PerspectiveCameras,
    image_size: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts a batch of `PerspectiveCameras` into OpenCV-convention
    rotation matrices `R`, translation vectors `tvec`, and the camera
    calibration matrices `camera_matrix`. This operation is exactly the inverse
    of `cameras_from_opencv_projection`.

    Note:
        The outputs `R, tvec, camera_matrix` correspond to the inputs of
        `cv2.projectPoints(x_world, rvec, tvec, camera_matrix, [])`,
        where `rvec` is an axis-angle vector that can be obtained from
        the rotation matrix `R` output here by calling the `so3_log_map` function.
        Correspondingly, `R` can be obtained from `rvec` by calling `so3_exp_map`.

    Args:
        cameras: A batch of `N` cameras in the PyTorch3D convention.
        image_size: A tensor of shape `(N, 2)` containing the sizes of the images
            (height, width) attached to each camera.
        return_as_rotmat (bool): If set to True, return the full 3x3 rotation
            matrices. Otherwise, return an axis-angle vector (default).

    Returns:
        R: A batch of rotation matrices of shape `(N, 3, 3)`.
        tvec: A batch of translation vectors of shape `(N, 3)`.
        camera_matrix: A batch of camera calibration matrices of shape `(N, 3, 3)`.
    """
    return _opencv_from_cameras_projection(cameras, image_size)


def pulsar_from_opencv_projection(
    R: torch.Tensor,
    tvec: torch.Tensor,
    camera_matrix: torch.Tensor,
    image_size: torch.Tensor,
    znear: float = 0.1,
) -> torch.Tensor:
    """
    Convert OpenCV style camera parameters to Pulsar style camera parameters.

    Note:
        * Pulsar does NOT support different focal lengths for x and y.
          For conversion, we use the average of fx and fy.
        * The Pulsar renderer MUST use a left-handed coordinate system for this
          mapping to work.
        * The resulting image will be vertically flipped - which has to be
          addressed AFTER rendering by the user.
        * The parameters `R, tvec, camera_matrix` correspond to the outputs
          of `cv2.decomposeProjectionMatrix`.

    Args:
        R: A batch of rotation matrices of shape `(N, 3, 3)`.
        tvec: A batch of translation vectors of shape `(N, 3)`.
        camera_matrix: A batch of camera calibration matrices of shape `(N, 3, 3)`.
        image_size: A tensor of shape `(N, 2)` containing the sizes of the images
            (height, width) attached to each camera.
        znear (float): The near clipping value to use for Pulsar.

    Returns:
        cameras_pulsar: A batch of `N` Pulsar camera vectors in the Pulsar
            convention `(N, 13)` (3 translation, 6 rotation, focal_length, sensor_width,
            c_x, c_y).
    """
    return _pulsar_from_opencv_projection(R, tvec, camera_matrix, image_size, znear)


def pulsar_from_cameras_projection(
    cameras: PerspectiveCameras,
    image_size: torch.Tensor,
) -> torch.Tensor:
    """
    Convert PyTorch3D `PerspectiveCameras` to Pulsar style camera parameters.

    Note:
        * Pulsar does NOT support different focal lengths for x and y.
          For conversion, we use the average of fx and fy.
        * The Pulsar renderer MUST use a left-handed coordinate system for this
          mapping to work.
        * The resulting image will be vertically flipped - which has to be
          addressed AFTER rendering by the user.

    Args:
        cameras: A batch of `N` cameras in the PyTorch3D convention.
        image_size: A tensor of shape `(N, 2)` containing the sizes of the images
            (height, width) attached to each camera.

    Returns:
        cameras_pulsar: A batch of `N` Pulsar camera vectors in the Pulsar
            convention `(N, 13)` (3 translation, 6 rotation, focal_length, sensor_width,
            c_x, c_y).
    """
    return _pulsar_from_cameras_projection(cameras, image_size)
