#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import torch.nn.functional as F

from pytorch3d.ops.nearest_neighbor_points import nn_points_idx


def _validate_chamfer_reduction_inputs(
    batch_reduction: str, point_reduction: str
):
    """Check the requested reductions are valid.

    Args:
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["none", "mean", "sum"].
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["none", "mean", "sum"].
    """
    if batch_reduction not in ["none", "mean", "sum"]:
        raise ValueError(
            'batch_reduction must be one of ["none", "mean", "sum"]'
        )
    if point_reduction not in ["none", "mean", "sum"]:
        raise ValueError(
            'point_reduction must be one of ["none", "mean", "sum"]'
        )
    if batch_reduction == "none" and point_reduction == "none":
        raise ValueError(
            'batch_reduction and point_reduction cannot both be "none".'
        )


def chamfer_distance(
    x,
    y,
    x_normals=None,
    y_normals=None,
    weights=None,
    batch_reduction: str = "mean",
    point_reduction: str = "mean",
):
    """
    Chamfer distance between two pointclouds x and y.

    Args:
        x: FloatTensor of shape (N, P1, D) representing a batch of point clouds
            with P1 points in each batch element, batch size N and feature
            dimension D.
        y: FloatTensor of shape (N, P2, D) representing a batch of point clouds
            with P2 points in each batch element, batch size N and feature
            dimension D.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["none", "mean", "sum"].
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["none", "mean", "sum"].

    Returns:
        2-element tuple containing

        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None.
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    N, P1, D = x.shape
    P2 = y.shape[1]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights can not be nonnegative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return (
                (x.sum((1, 2)) * weights) * 0.0,
                (x.sum((1, 2)) * weights) * 0.0,
            )

    return_normals = x_normals is not None and y_normals is not None
    cham_norm_x = x.new_zeros(())
    cham_norm_y = x.new_zeros(())

    x_near, xidx_near, x_normals_near = nn_points_idx(x, y, y_normals)
    y_near, yidx_near, y_normals_near = nn_points_idx(y, x, x_normals)

    cham_x = (x - x_near).norm(dim=2, p=2) ** 2.0  # (N, P1)
    cham_y = (y - y_near).norm(dim=2, p=2) ** 2.0  # (N, P2)

    if weights is not None:
        cham_x *= weights.view(N, 1)
        cham_y *= weights.view(N, 1)

    if return_normals:
        cham_norm_x = 1 - torch.abs(
            F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
        )
        cham_norm_y = 1 - torch.abs(
            F.cosine_similarity(y_normals, y_normals_near, dim=2, eps=1e-6)
        )
        if weights is not None:
            cham_norm_x *= weights.view(N, 1)
            cham_norm_y *= weights.view(N, 1)

    if point_reduction != "none":
        # If not 'none' then either 'sum' or 'mean'.
        cham_x = cham_x.sum(1)  # (N,)
        cham_y = cham_y.sum(1)  # (N,)
        if return_normals:
            cham_norm_x = cham_norm_x.sum(1)  # (N,)
            cham_norm_y = cham_norm_y.sum(1)  # (N,)
        if point_reduction == "mean":
            cham_x /= P1
            cham_y /= P2
            if return_normals:
                cham_norm_x /= P1
                cham_norm_y /= P2

    if batch_reduction != "none":
        cham_x = cham_x.sum()
        cham_y = cham_y.sum()
        if return_normals:
            cham_norm_x = cham_norm_x.sum()
            cham_norm_y = cham_norm_y.sum()
        if batch_reduction == "mean":
            div = weights.sum() if weights is not None else N
            cham_x /= div
            cham_y /= div
            if return_normals:
                cham_norm_x /= div
                cham_norm_y /= div

    cham_dist = cham_x + cham_y
    cham_normals = cham_norm_x + cham_norm_y if return_normals else None

    return cham_dist, cham_normals
