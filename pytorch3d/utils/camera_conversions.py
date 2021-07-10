# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Tuple

import torch

from ..renderer import PerspectiveCameras
from ..transforms import matrix_to_rotation_6d


LOGGER = logging.getLogger(__name__)


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
    space as follows:
        ```
        x_screen_opencv = camera_matrix @ (R @ x_world + tvec)
        ```
    followed by the homogenization of `x_screen_opencv`.

    Note:
        The parameters `R, tvec, camera_matrix` correspond to the outputs of
        `cv2.decomposeProjectionMatrix`.

        The `rvec` parameter of the `cv2.projectPoints` is an axis-angle vector
        that can be converted to the rotation matrix `R` expected here by
        calling the `so3_exp_map` function.

    Args:
        R: A batch of rotation matrices of shape `(N, 3, 3)`.
        tvec: A batch of translation vectors of shape `(N, 3)`.
        camera_matrix: A batch of camera calibration matrices of shape `(N, 3, 3)`.
        image_size: A tensor of shape `(N, 2)` containing the sizes of the images
            (height, width) attached to each camera.

    Returns:
        cameras_pytorch3d: A batch of `N` cameras in the PyTorch3D convention.
    """
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
    R_pytorch3d = R.clone().permute(0, 2, 1)
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
    rotation matrices `R`, translation vectors `tvec`, and the camera
    calibration matrices `camera_matrix`. This operation is exactly the inverse
    of `cameras_from_opencv_projection`.

    Note:
        The outputs `R, tvec, camera_matrix` correspond to the outputs of
        `cv2.decomposeProjectionMatrix`.

        The `rvec` parameter of the `cv2.projectPoints` is an axis-angle vector
        that can be converted from the returned rotation matrix `R` here by
        calling the `so3_log_map` function.

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
    R_pytorch3d = cameras.R.clone()  # pyre-ignore
    T_pytorch3d = cameras.T.clone()  # pyre-ignore
    focal_pytorch3d = cameras.focal_length
    p0_pytorch3d = cameras.principal_point
    T_pytorch3d[:, :2] *= -1
    R_pytorch3d[:, :, :2] *= -1
    tvec = T_pytorch3d
    R = R_pytorch3d.permute(0, 2, 1)

    # Retype the image_size correctly and flip to width, height.
    image_size_wh = image_size.to(R).flip(dims=(1,))

    principal_point = (-p0_pytorch3d + 1.0) * (0.5 * image_size_wh)  # pyre-ignore
    focal_length = focal_pytorch3d * (0.5 * image_size_wh)

    camera_matrix = torch.zeros_like(R)
    camera_matrix[:, :2, 2] = principal_point
    camera_matrix[:, 2, 2] = 1.0
    camera_matrix[:, 0, 0] = focal_length[:, 0]
    camera_matrix[:, 1, 1] = focal_length[:, 1]
    return R, tvec, camera_matrix


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
    assert len(camera_matrix.size()) == 3, "This function requires batched inputs!"
    assert len(R.size()) == 3, "This function requires batched inputs!"
    assert len(tvec.size()) in (2, 3), "This function reuqires batched inputs!"

    # Validate parameters.
    image_size_wh = image_size.to(R).flip(dims=(1,))
    assert torch.all(
        image_size_wh > 0
    ), "height and width must be positive but min is: %s" % (
        str(image_size_wh.min().item())
    )
    assert (
        camera_matrix.size(1) == 3 and camera_matrix.size(2) == 3
    ), "Incorrect camera matrix shape: expected 3x3 but got %dx%d" % (
        camera_matrix.size(1),
        camera_matrix.size(2),
    )
    assert (
        R.size(1) == 3 and R.size(2) == 3
    ), "Incorrect R shape: expected 3x3 but got %dx%d" % (
        R.size(1),
        R.size(2),
    )
    if len(tvec.size()) == 2:
        tvec = tvec.unsqueeze(2)
    assert (
        tvec.size(1) == 3 and tvec.size(2) == 1
    ), "Incorrect tvec shape: expected 3x1 but got %dx%d" % (
        tvec.size(1),
        tvec.size(2),
    )
    # Check batch size.
    batch_size = camera_matrix.size(0)
    assert R.size(0) == batch_size, "Expected R to have batch size %d. Has size %d." % (
        batch_size,
        R.size(0),
    )
    assert (
        tvec.size(0) == batch_size
    ), "Expected tvec to have batch size %d. Has size %d." % (
        batch_size,
        tvec.size(0),
    )
    # Check image sizes.
    image_w = image_size_wh[0, 0]
    image_h = image_size_wh[0, 1]
    assert torch.all(
        image_size_wh[:, 0] == image_w
    ), "All images in a batch must have the same width!"
    assert torch.all(
        image_size_wh[:, 1] == image_h
    ), "All images in a batch must have the same height!"
    # Focal length.
    fx = camera_matrix[:, 0, 0].unsqueeze(1)
    fy = camera_matrix[:, 1, 1].unsqueeze(1)
    # Check that we introduce less than 1% error by averaging the focal lengths.
    fx_y = fx / fy
    if torch.any(fx_y > 1.01) or torch.any(fx_y < 0.99):
        LOGGER.warning(
            "Pulsar only supports a single focal lengths. For converting OpenCV "
            "focal lengths, we average them for x and y directions. "
            "The focal lengths for x and y you provided differ by more than 1%, "
            "which means this could introduce a noticeable error."
        )
    f = (fx + fy) / 2
    # Normalize f into normalized device coordinates.
    focal_length_px = f / image_w
    # Transfer into focal_length and sensor_width.
    focal_length = torch.tensor([znear - 1e-5], dtype=torch.float32, device=R.device)
    focal_length = focal_length[None, :].repeat(batch_size, 1)
    sensor_width = focal_length / focal_length_px
    # Principal point.
    cx = camera_matrix[:, 0, 2].unsqueeze(1)
    cy = camera_matrix[:, 1, 2].unsqueeze(1)
    # Transfer principal point offset into centered offset.
    cx = -(cx - image_w / 2)
    cy = cy - image_h / 2
    # Concatenate to final vector.
    param = torch.cat([focal_length, sensor_width, cx, cy], dim=1)
    R_trans = R.permute(0, 2, 1)
    cam_pos = -torch.bmm(R_trans, tvec).squeeze(2)
    cam_rot = matrix_to_rotation_6d(R_trans)
    cam_params = torch.cat([cam_pos, cam_rot, param], dim=1)
    return cam_params


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
    opencv_R, opencv_T, opencv_K = opencv_from_cameras_projection(cameras, image_size)
    return pulsar_from_opencv_projection(opencv_R, opencv_T, opencv_K, image_size)
