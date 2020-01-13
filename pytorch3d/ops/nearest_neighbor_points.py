#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import torch

from pytorch3d import _C


def nn_points_idx(p1, p2, p2_normals=None) -> torch.Tensor:
    """
    Compute the coordinates of nearest neighbors in pointcloud p2 to points in p1.
    Args:
        p1: FloatTensor of shape (N, P1, D) giving a batch of pointclouds each
            containing P1 points of dimension D.
        p2: FloatTensor of shape (N, P2, D) giving a batch of pointclouds each
            containing P2 points of dimension D.
        p2_normals: [optional] FloatTensor of shape (N, P2, D) giving
                   normals for p2. Default: None.

    Returns:
        3-element tuple containing

        - **p1_nn_points**: FloatTensor of shape (N, P1, D) where
          p1_neighbors[n, i] is the point in p2[n] which is
          the nearest neighbor to p1[n, i].
        - **p1_nn_idx**: LongTensor of shape (N, P1) giving the indices of
          the neighbors.
        - **p1_nn_normals**: Normal vectors for each point in p1_neighbors;
          only returned if p2_normals is passed
          else return [].
    """
    N, P1, D = p1.shape
    with torch.no_grad():
        p1_nn_idx = _C.nn_points_idx(
            p1.contiguous(), p2.contiguous()
        )  # (N, P1)
    p1_nn_idx_expanded = p1_nn_idx.view(N, P1, 1).expand(N, P1, D)
    p1_nn_points = p2.gather(1, p1_nn_idx_expanded)
    if p2_normals is None:
        p1_nn_normals = []
    else:
        if p2_normals.shape != p2.shape:
            raise ValueError("p2_normals has incorrect shape.")
        p1_nn_normals = p2_normals.gather(1, p1_nn_idx_expanded)

    return p1_nn_points, p1_nn_idx, p1_nn_normals
