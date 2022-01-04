# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import tee
from math import cos, pi, sin
from typing import Iterator, Optional, Tuple

import torch
from pytorch3d.structures.meshes import Meshes


# Make an iterator over the adjacent pairs: (-1, 0), (0, 1), ..., (N - 2, N - 1)
def _make_pair_range(N: int) -> Iterator[Tuple[int, int]]:
    i, j = tee(range(-1, N))
    next(j, None)
    return zip(i, j)


def torus(
    r: float, R: float, sides: int, rings: int, device: Optional[torch.device] = None
) -> Meshes:
    """
    Create vertices and faces for a torus.

    Args:
        r: Inner radius of the torus.
        R: Outer radius of the torus.
        sides: Number of inner divisions.
        rings: Number of outer divisions.
        device: Device on which the outputs will be allocated.

    Returns:
        Meshes object with the generated vertices and faces.
    """
    if not (sides > 0):
        raise ValueError("sides must be > 0.")
    if not (rings > 0):
        raise ValueError("rings must be > 0.")
    device = device if device else torch.device("cpu")

    verts = []
    for i in range(rings):
        # phi ranges from 0 to 2 pi (rings - 1) / rings
        phi = 2 * pi * i / rings
        for j in range(sides):
            # theta ranges from 0 to 2 pi (sides - 1) / sides
            theta = 2 * pi * j / sides
            x = (R + r * cos(theta)) * cos(phi)
            y = (R + r * cos(theta)) * sin(phi)
            z = r * sin(theta)
            # This vertex has index i * sides + j
            verts.append([x, y, z])

    faces = []
    for i0, i1 in _make_pair_range(rings):
        index0 = (i0 % rings) * sides
        index1 = (i1 % rings) * sides
        for j0, j1 in _make_pair_range(sides):
            index00 = index0 + (j0 % sides)
            index01 = index0 + (j1 % sides)
            index10 = index1 + (j0 % sides)
            index11 = index1 + (j1 % sides)
            faces.append([index00, index10, index11])
            faces.append([index11, index01, index00])

    verts_list = [torch.tensor(verts, dtype=torch.float32, device=device)]
    faces_list = [torch.tensor(faces, dtype=torch.int64, device=device)]
    return Meshes(verts_list, faces_list)
