# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from pytorch3d.ops.subdivide_meshes import SubdivideMeshes
from pytorch3d.structures.meshes import Meshes


# Vertex coordinates for a level 0 plane.
_plane_verts0 = [
    [-0.5000, -0.5000, 0.0000],  # TL
    [+0.5000, -0.5000, 0.0000],  # TR
    [+0.5000, +0.5000, 0.0000],  # BR
    [-0.5000, +0.5000, 0.0000],  # BL
]

# Faces for level 0 plane
_plane_faces0 = [[2, 1, 0], [0, 3, 2]]


def plane(level: int = 0, device=None):
    """
    Create verts and faces for a unit plane, with all faces oriented
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
        verts = torch.tensor(_plane_verts0, dtype=torch.float32, device=device)
        faces = torch.tensor(_plane_faces0, dtype=torch.int64, device=device)
    else:
        mesh = plane(level - 1, device)
        subdivide = SubdivideMeshes()
        mesh = subdivide(mesh)
        verts = mesh.verts_list()[0]
        faces = mesh.faces_list()[0]
    return Meshes(verts=[verts], faces=[faces])
