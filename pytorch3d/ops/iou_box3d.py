# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
import torch.nn.functional as F
from pytorch3d import _C
from torch.autograd import Function


# -------------------------------------------------- #
#                  CONSTANTS                         #
# -------------------------------------------------- #
"""
_box_planes and _box_triangles define the 4- and 3-connectivity
of the 8 box corners.
_box_planes gives the quad faces of the 3D box
_box_triangles gives the triangle faces of the 3D box
"""
_box_planes = [
    [0, 1, 2, 3],
    [3, 2, 6, 7],
    [0, 1, 5, 4],
    [0, 3, 7, 4],
    [1, 2, 6, 5],
    [4, 5, 6, 7],
]
_box_triangles = [
    [0, 1, 2],
    [0, 3, 2],
    [4, 5, 6],
    [4, 6, 7],
    [1, 5, 6],
    [1, 6, 2],
    [0, 4, 7],
    [0, 7, 3],
    [3, 2, 6],
    [3, 6, 7],
    [0, 1, 5],
    [0, 4, 5],
]


def _check_coplanar(boxes: torch.Tensor, eps: float = 1e-4) -> None:
    faces = torch.tensor(_box_planes, dtype=torch.int64, device=boxes.device)
    verts = boxes.index_select(index=faces.view(-1), dim=1)
    B = boxes.shape[0]
    P, V = faces.shape
    # (B, P, 4, 3) -> (B, P, 3)
    v0, v1, v2, v3 = verts.reshape(B, P, V, 3).unbind(2)

    # Compute the normal
    e0 = F.normalize(v1 - v0, dim=-1)
    e1 = F.normalize(v2 - v0, dim=-1)
    normal = F.normalize(torch.cross(e0, e1, dim=-1), dim=-1)

    # Check the fourth vertex is also on the same plane
    mat1 = (v3 - v0).view(B, 1, -1)  # (B, 1, P*3)
    mat2 = normal.view(B, -1, 1)  # (B, P*3, 1)
    if not (mat1.bmm(mat2).abs() < eps).all().item():
        msg = "Plane vertices are not coplanar"
        raise ValueError(msg)

    return


def _check_nonzero(boxes: torch.Tensor, eps: float = 1e-4) -> None:
    """
    Checks that the sides of the box have a non zero area
    """
    faces = torch.tensor(_box_triangles, dtype=torch.int64, device=boxes.device)
    verts = boxes.index_select(index=faces.view(-1), dim=1)
    B = boxes.shape[0]
    T, V = faces.shape
    # (B, T, 3, 3) -> (B, T, 3)
    v0, v1, v2 = verts.reshape(B, T, V, 3).unbind(2)

    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (B, T, 3)
    face_areas = normals.norm(dim=-1) / 2

    if (face_areas < eps).any().item():
        msg = "Planes have zero areas"
        raise ValueError(msg)

    return


class _box3d_overlap(Function):
    """
    Torch autograd Function wrapper for box3d_overlap C++/CUDA implementations.
    Backward is not supported.
    """

    @staticmethod
    def forward(ctx, boxes1, boxes2):
        """
        Arguments defintions the same as in the box3d_overlap function
        """
        vol, iou = _C.iou_box3d(boxes1, boxes2)
        return vol, iou

    @staticmethod
    def backward(ctx, grad_vol, grad_iou):
        raise ValueError("box3d_overlap backward is not supported")


def box3d_overlap(
    boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the intersection of 3D boxes1 and boxes2.

    Inputs boxes1, boxes2 are tensors of shape (B, 8, 3)
    (where B doesn't have to be the same for boxes1 and boxes2),
    containing the 8 corners of the boxes, as follows:

        (4) +---------+. (5)
            | ` .     |  ` .
            | (0) +---+-----+ (1)
            |     |   |     |
        (7) +-----+---+. (6)|
            ` .   |     ` . |
            (3) ` +---------+ (2)


    NOTE: Throughout this implementation, we assume that boxes
    are defined by their 8 corners exactly in the order specified in the
    diagram above for the function to give correct results. In addition
    the vertices on each plane must be coplanar.
    As an alternative to the diagram, this is a unit bounding
    box which has the correct vertex ordering:

    box_corner_vertices = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ]

    Args:
        boxes1: tensor of shape (N, 8, 3) of the coordinates of the 1st boxes
        boxes2: tensor of shape (M, 8, 3) of the coordinates of the 2nd boxes
    Returns:
        vol: (N, M) tensor of the volume of the intersecting convex shapes
        iou: (N, M) tensor of the intersection over union which is
            defined as: `iou = vol / (vol1 + vol2 - vol)`
    """
    if not all((8, 3) == box.shape[1:] for box in [boxes1, boxes2]):
        raise ValueError("Each box in the batch must be of shape (8, 3)")

    _check_coplanar(boxes1, eps)
    _check_coplanar(boxes2, eps)
    _check_nonzero(boxes1, eps)
    _check_nonzero(boxes2, eps)

    vol, iou = _box3d_overlap.apply(boxes1, boxes2)

    return vol, iou
