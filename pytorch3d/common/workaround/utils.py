# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch


def _safe_det_3x3(t: torch.Tensor):
    """
    Fast determinant calculation for a batch of 3x3 matrices.

    Note, result of this function might not be the same as `torch.det()`.
    The differences might be in the last significant digit.

    Args:
        t: Tensor of shape (N, 3, 3).

    Returns:
        Tensor of shape (N) with determinants.
    """

    det = (
        t[..., 0, 0] * (t[..., 1, 1] * t[..., 2, 2] - t[..., 1, 2] * t[..., 2, 1])
        - t[..., 0, 1] * (t[..., 1, 0] * t[..., 2, 2] - t[..., 2, 0] * t[..., 1, 2])
        + t[..., 0, 2] * (t[..., 1, 0] * t[..., 2, 1] - t[..., 2, 0] * t[..., 1, 1])
    )

    return det
