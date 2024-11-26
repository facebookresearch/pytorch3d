# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import logging
from typing import Tuple

import torch

from ..transforms import matrix_to_rotation_6d
from .cameras import PerspectiveCameras


LOGGER = logging.getLogger(__name__)


def _cameras_from_opencv_projection(
    R: torch.Tensor,
    tvec: torch.Tensor,
    camera_matrix: torch.Tensor,
    image_size: torch.Tensor,
) -> PerspectiveCameras:
    focal_length = torch.stack([camera_matrix[:, 0, 0], camera_matrix[:, 1, 1]], dim=-1)
    principal_point = camera_matrix[:, :2, 2]

    # Retype the image_size correctly and flip to width, height.
    image_size_wh = image_size.to(R).flip(dims=(1,))

    # Screen to NDC conversion:
    # For non square images, we scale the points such that smallest side
    # has range [-1, 1] and the largest side has range [-u, u], with u > 1.
    # This convention is consistent with the PyTorch3D renderer, as well as
    # the transformation function `get_ndc_to_screen_transform`.
    scale = image_size_wh.to(R).min(dim=1, keepdim=True)[0] / 2.0
    scale = scale.expand(-1, 2)
    c0 = image_size_wh / 2.0

    # Get the PyTorch3D focal length and principal point.
    focal_pytorch3d = focal_length / scale
    p0_pytorch3d = -(principal_point - c0) / scale

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
        image_size=image_size,
        device=R.device,
    )


def _opencv_from_cameras_projection(
    cameras: PerspectiveCameras,
    image_size: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # pyre-fixme[29]: `Union[(self: TensorBase, memory_format:
    #  Optional[memory_format] = ...) -> Tensor, Tensor, Module]` is not a function.
    R_pytorch3d = cameras.R.clone()
    # pyre-fixme[29]: `Union[(self: TensorBase, memory_format:
    #  Optional[memory_format] = ...) -> Tensor, Tensor, Module]` is not a function.
    T_pytorch3d = cameras.T.clone()
    focal_pytorch3d = cameras.focal_length
    p0_pytorch3d = cameras.principal_point
    T_pytorch3d[:, :2] *= -1
    R_pytorch3d[:, :, :2] *= -1
    tvec = T_pytorch3d
    R = R_pytorch3d.permute(0, 2, 1)

    # Retype the image_size correctly and flip to width, height.
    image_size_wh = image_size.to(R).flip(dims=(1,))

    # NDC to screen conversion.
    scale = image_size_wh.to(R).min(dim=1, keepdim=True)[0] / 2.0
    scale = scale.expand(-1, 2)
    c0 = image_size_wh / 2.0

    principal_point = -p0_pytorch3d * scale + c0
    focal_length = focal_pytorch3d * scale

    camera_matrix = torch.zeros_like(R)
    camera_matrix[:, :2, 2] = principal_point
    camera_matrix[:, 2, 2] = 1.0
    camera_matrix[:, 0, 0] = focal_length[:, 0]
    camera_matrix[:, 1, 1] = focal_length[:, 1]
    return R, tvec, camera_matrix


def _pulsar_from_opencv_projection(
    R: torch.Tensor,
    tvec: torch.Tensor,
    camera_matrix: torch.Tensor,
    image_size: torch.Tensor,
    znear: float = 0.1,
) -> torch.Tensor:
    assert len(camera_matrix.size()) == 3, "This function requires batched inputs!"
    assert len(R.size()) == 3, "This function requires batched inputs!"
    assert len(tvec.size()) in (2, 3), "This function reuqires batched inputs!"

    # Validate parameters.
    image_size_wh = image_size.to(R).flip(dims=(1,))
    assert torch.all(image_size_wh > 0), (
        "height and width must be positive but min is: %s"
        % (str(image_size_wh.min().item()))
    )
    assert camera_matrix.size(1) == 3 and camera_matrix.size(2) == 3, (
        "Incorrect camera matrix shape: expected 3x3 but got %dx%d"
        % (
            camera_matrix.size(1),
            camera_matrix.size(2),
        )
    )
    assert R.size(1) == 3 and R.size(2) == 3, (
        "Incorrect R shape: expected 3x3 but got %dx%d"
        % (
            R.size(1),
            R.size(2),
        )
    )
    if len(tvec.size()) == 2:
        tvec = tvec.unsqueeze(2)
    assert tvec.size(1) == 3 and tvec.size(2) == 1, (
        "Incorrect tvec shape: expected 3x1 but got %dx%d"
        % (
            tvec.size(1),
            tvec.size(2),
        )
    )
    # Check batch size.
    batch_size = camera_matrix.size(0)
    assert R.size(0) == batch_size, "Expected R to have batch size %d. Has size %d." % (
        batch_size,
        R.size(0),
    )
    assert tvec.size(0) == batch_size, (
        "Expected tvec to have batch size %d. Has size %d."
        % (
            batch_size,
            tvec.size(0),
        )
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


def _pulsar_from_cameras_projection(
    cameras: PerspectiveCameras,
    image_size: torch.Tensor,
) -> torch.Tensor:
    opencv_R, opencv_T, opencv_K = _opencv_from_cameras_projection(cameras, image_size)
    return _pulsar_from_opencv_projection(opencv_R, opencv_T, opencv_K, image_size)
