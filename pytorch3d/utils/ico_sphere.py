# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from pytorch3d.ops.subdivide_meshes import SubdivideMeshes
from pytorch3d.structures.meshes import Meshes


# Vertex coordinates for a level 0 ico-sphere.
_ico_verts0 = [
    [-0.5257, 0.8507, 0.0000],
    [0.5257, 0.8507, 0.0000],
    [-0.5257, -0.8507, 0.0000],
    [0.5257, -0.8507, 0.0000],
    [0.0000, -0.5257, 0.8507],
    [0.0000, 0.5257, 0.8507],
    [0.0000, -0.5257, -0.8507],
    [0.0000, 0.5257, -0.8507],
    [0.8507, 0.0000, -0.5257],
    [0.8507, 0.0000, 0.5257],
    [-0.8507, 0.0000, -0.5257],
    [-0.8507, 0.0000, 0.5257],
]


# Faces for level 0 ico-sphere
_ico_faces0 = [
    [0, 11, 5],
    [0, 5, 1],
    [0, 1, 7],
    [0, 7, 10],
    [0, 10, 11],
    [1, 5, 9],
    [5, 11, 4],
    [11, 10, 2],
    [10, 7, 6],
    [7, 1, 8],
    [3, 9, 4],
    [3, 4, 2],
    [3, 2, 6],
    [3, 6, 8],
    [3, 8, 9],
    [4, 9, 5],
    [2, 4, 11],
    [6, 2, 10],
    [8, 6, 7],
    [9, 8, 1],
]


def ico_sphere(level: int = 0, device=None):
    """
    Create verts and faces for a unit ico-sphere, with all faces oriented
    consistently.

    Args:
        level: integer specifying the number of iterations for subdivision
               of the mesh faces. Each additional level will result in four new
               faces per face.
        device: A torch.device object on which the outputs will be allocated.

    Returns:
        Meshes object with verts and faces.
    """
    if device is None:
        device = torch.device("cpu")
    if level < 0:
        raise ValueError("level must be >= 0.")
    if level == 0:
        verts = torch.tensor(_ico_verts0, dtype=torch.float32, device=device)
        faces = torch.tensor(_ico_faces0, dtype=torch.int64, device=device)

    else:
        mesh = ico_sphere(level - 1, device)
        subdivide = SubdivideMeshes()
        mesh = subdivide(mesh)
        verts = mesh.verts_list()[0]
        verts /= verts.norm(p=2, dim=1, keepdim=True)
        faces = mesh.faces_list()[0]
    return Meshes(verts=[verts], faces=[faces])
