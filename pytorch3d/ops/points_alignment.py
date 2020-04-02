#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import warnings
from typing import Tuple, Union
import torch

from pytorch3d.structures.pointclouds import Pointclouds


def corresponding_points_alignment(
    X: Union[torch.Tensor, Pointclouds],
    Y: Union[torch.Tensor, Pointclouds],
    estimate_scale: bool = False,
    allow_reflection: bool = False,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Finds a similarity transformation (rotation `R`, translation `T`
    and optionally scale `s`)  between two given sets of corresponding
    `d`-dimensional points `X` and `Y` such that:

    `s[i] X[i] R[i] + T[i] = Y[i]`,

    for all batch indexes `i` in the least squares sense.

    The algorithm is also known as Umeyama [1].

    Args:
        X: Batch of `d`-dimensional points of shape `(minibatch, num_point, d)`
           or a `Pointclouds` object.
        Y: Batch of `d`-dimensional points of shape `(minibatch, num_point, d)`
           or a `Pointclouds` object.
        estimate_scale: If `True`, also estimates a scaling component `s`
            of the transformation. Otherwise assumes an identity
            scale and returns a tensor of ones.
        allow_reflection: If `True`, allows the algorithm to return `R`
            which is orthonormal but has determinant==-1.
        eps: A scalar for clamping to avoid dividing by zero. Active for the
            code that estimates the output scale `s`.

    Returns:
        3-element tuple containing
        - **R**: Batch of orthonormal matrices of shape `(minibatch, d, d)`.
        - **T**: Batch of translations of shape `(minibatch, d)`.
        - **s**: batch of scaling factors of shape `(minibatch, )`.

    References:
        [1] Shinji Umeyama: Least-Suqares Estimation of
        Transformation Parameters Between Two Point Patterns
    """

    # make sure we convert input Pointclouds structures to tensors
    Xt, num_points = _convert_point_cloud_to_tensor(X)
    Yt, num_points_Y = _convert_point_cloud_to_tensor(Y)

    if (Xt.shape != Yt.shape) or (num_points != num_points_Y).any():
        raise ValueError(
            "Point sets X and Y have to have the same \
            number of batches, points and dimensions."
        )

    b, n, dim = Xt.shape

    # compute the centroids of the point sets
    Xmu = Xt.sum(1) / torch.clamp(num_points[:, None], 1)
    Ymu = Yt.sum(1) / torch.clamp(num_points[:, None], 1)

    # mean-center the point sets
    Xc = Xt - Xmu[:, None]
    Yc = Yt - Ymu[:, None]

    if (num_points < Xt.shape[1]).any() or (num_points < Yt.shape[1]).any():
        # in case we got Pointclouds as input, mask the unused entries in Xc, Yc
        mask = (
            torch.arange(n, dtype=torch.int64, device=Xc.device)[None]
            < num_points[:, None]
        ).type_as(Xc)
        Xc *= mask[:, :, None]
        Yc *= mask[:, :, None]

    if (num_points < (dim + 1)).any():
        warnings.warn(
            "The size of one of the point clouds is <= dim+1. "
            + "corresponding_points_alignment can't return a unique solution."
        )

    # compute the covariance XYcov between the point sets Xc, Yc
    XYcov = torch.bmm(Xc.transpose(2, 1), Yc)
    XYcov = XYcov / torch.clamp(num_points[:, None, None], 1)

    # decompose the covariance matrix XYcov
    U, S, V = torch.svd(XYcov)

    # identity matrix used for fixing reflections
    E = torch.eye(dim, dtype=XYcov.dtype, device=XYcov.device)[None].repeat(
        b, 1, 1
    )

    if not allow_reflection:
        # reflection test:
        #   checks whether the estimated rotation has det==1,
        #   if not, finds the nearest rotation s.t. det==1 by
        #   flipping the sign of the last singular vector U
        R_test = torch.bmm(U, V.transpose(2, 1))
        E[:, -1, -1] = torch.det(R_test)

    # find the rotation matrix by composing U and V again
    R = torch.bmm(torch.bmm(U, E), V.transpose(2, 1))

    if estimate_scale:
        # estimate the scaling component of the transformation
        trace_ES = (torch.diagonal(E, dim1=1, dim2=2) * S).sum(1)
        Xcov = (Xc * Xc).sum((1, 2)) / torch.clamp(num_points, 1)

        # the scaling component
        s = trace_ES / torch.clamp(Xcov, eps)

        # translation component
        T = Ymu - s[:, None] * torch.bmm(Xmu[:, None], R)[:, 0, :]

    else:
        # translation component
        T = Ymu - torch.bmm(Xmu[:, None], R)[:, 0]

        # unit scaling since we do not estimate scale
        s = T.new_ones(b)

    return R, T, s


def _convert_point_cloud_to_tensor(pcl: Union[torch.Tensor, Pointclouds]):
    """
    If `type(pcl)==Pointclouds`, converts a `pcl` object to a
    padded representation and returns it together with the number of points
    per batch. Otherwise, returns the input itself with the number of points
    set to the size of the second dimension of `pcl`.
    """
    if isinstance(pcl, Pointclouds):
        X = pcl.points_padded()
        num_points = pcl.num_points_per_cloud()
    elif torch.is_tensor(pcl):
        X = pcl
        num_points = X.shape[1] * torch.ones(
            X.shape[0], device=X.device, dtype=torch.int64
        )
    else:
        raise ValueError(
            "The inputs X, Y should be either Pointclouds objects or tensors."
        )
    return X, num_points
