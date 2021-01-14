# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
"""pulsar renderer Camera utils.
"""
import numpy as np
import torch
from pytorch3d.transforms.so3 import so3_log_map


def opencv2pulsar(
    K: torch.Tensor,
    R: torch.Tensor,
    T: torch.Tensor,
    h: int,
    w: int,
    znear: float = 0.1,
) -> torch.Tensor:
    """
    Convert OpenCV style camera parameters to Pulsar style cameras.

    !!IMPORTANT!!
    Pulsar does NOT support different focal lengths for x and y yet so
    we simply take the average of fx and fy.

    Args:
        * K: intrinsic camera parameters. [Bx]3x3.
            [[fx, 0, cx],
             [0, fy, cy],
             [0,  0,  1]]
        * R: camera rotation in world coors. [Bx]3x3.
        * T: camera translation in world coords. [bx]3x1
        * h: image height
        * w: image width
        * znear: defines near clipping plane
    """
    # users may pass numpy arrays rather than torch tensors
    if isinstance(K, np.ndarray):
        K = torch.from_numpy(K).float()
    if isinstance(R, np.ndarray):
        R = torch.from_numpy(R).float()
    if isinstance(T, np.ndarray):
        T = torch.from_numpy(T).float()

    device = K.device

    # test if the data is batched or not using `K`
    # assume that all passed parameters are either
    # all batched or NOT batched at all.
    input_is_not_batched = len(K.size()) == 2
    if input_is_not_batched:
        K = K.unsqueeze(0)
        R = R.unsqueeze(0)
        T = T.unsqueeze(0)
    if len(T.size()) == 2:
        T = T.unsqueeze(2)  # make T a col vector

    # verify parameters
    assert h > 0 and w > 0, "height and width must be positive but are: %d, %d" % (h, w)
    assert (
        K.size(1) == 3 and K.size(2) == 3
    ), "Incorrect intrinsic shape: expected 3x3 but got %dx%d" % (K.size(1), K.size(2))
    assert (
        R.size(1) == 3 and R.size(2) == 3
    ), "Incorrect R shape: expected 3x3 but got %dx%d" % (R.size(1), R.size(2))
    assert (
        T.size(1) == 3 and T.size(2) == 1
    ), "Incorrect T shape: expected 3x1 but got %dx%d" % (T.size(1), T.size(2))

    batch_size = K.size(0)

    fx = K[:, 0, 0].unsqueeze(1)
    fy = K[:, 1, 1].unsqueeze(1)
    f = (fx + fy) / 2

    # Normalize f into normalized device coordinates (NDC).
    focal_length_px = f / w

    # Transfer into focal_length and sensor_width.
    focal_length = torch.tensor(
        [
            [
                znear - 1e-5,
            ]
        ],
        dtype=torch.float32,
        device=device,
    ).repeat(batch_size, 1)
    sensor_width = focal_length / focal_length_px

    cx = K[:, 0, 2].unsqueeze(1)
    cy = K[:, 1, 2].unsqueeze(1)

    # transfer principal point offset into centered offset
    cx = -(cx - w / 2)
    cy = cy - h / 2

    param = torch.cat([focal_length, sensor_width, cx, cy], dim=1)

    R_trans = R.permute(0, 2, 1)

    cam_pos = -torch.bmm(R_trans, T).squeeze(2)

    cam_rot = so3_log_map(R_trans)

    cam_params = torch.cat([cam_pos, cam_rot, param], dim=1)

    if input_is_not_batched:
        # un-batch params
        cam_params = cam_params[0]

    return cam_params
