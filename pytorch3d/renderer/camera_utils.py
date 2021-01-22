# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Tuple

import torch
from pytorch3d.transforms import Transform3d


def camera_to_eye_at_up(
    world_to_view_transform: Transform3d,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Given a world to view transform, return the eye, at and up vectors which
    represent its position.

    For example, if cam is a camera object, then after running

    .. code-block::

        from cameras import look_at_view_transform
        eye, at, up = camera_to_eye_at_up(cam.get_world_to_view_transform())
        R, T = look_at_view_transform(eye=eye, at=at, up=up)

    any other camera created from R and T will have the same world to view
    transform as cam.

    Also, given a camera position R and T, then after running:

    .. code-block::

        from cameras import get_world_to_view_transform, look_at_view_transform
        eye, at, up = camera_to_eye_at_up(get_world_to_view_transform(R=R, T=T))
        R2, T2 = look_at_view_transform(eye=eye, at=at, up=up)

    R2 will equal R and T2 will equal T.

    Args:
        world_to_view_transform: Transform3d representing the extrinsic
            transformation of N cameras.

    Returns:
        eye: FloatTensor of shape [N, 3] representing the camera centers in world space.
        at: FloatTensor of shape [N, 3] representing points in world space directly in
            front of the cameras e.g. the positions of objects to be viewed by the
            cameras.
        up: FloatTensor of shape [N, 3] representing vectors in world space which
            when projected on to the camera plane point upwards.
    """
    cam_trans = world_to_view_transform.inverse()
    # In the PyTorch3D right handed coordinate system, the camera in view space
    # is always at the origin looking along the +z axis.

    # The up vector is not a position so cannot be transformed with
    # transform_points. However the position eye+up above the camera
    # (whose position vector in the camera coordinate frame is an up vector)
    # can be transformed with transform_points.
    eye_at_up_view = torch.tensor(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=torch.float32, device=cam_trans.device
    )
    eye_at_up_world = cam_trans.transform_points(eye_at_up_view).reshape(-1, 3, 3)

    eye, at, up_plus_eye = eye_at_up_world.unbind(1)
    up = up_plus_eye - eye
    return eye, at, up
