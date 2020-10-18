# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


from typing import List, NamedTuple, Tuple

import torch
from pytorch3d.ops import interpolate_face_attributes


def _clip_barycentric_coordinates(bary) -> torch.Tensor:
    """
    Args:
        bary: barycentric coordinates of shape (...., 3) where `...` represents
            an arbitrary number of dimensions

    Returns:
        bary: Barycentric coordinates clipped (i.e any values < 0 are set to 0)
        and renormalized. We only clip  the negative values. Values > 1 will fall
        into the [0, 1] range after renormalization.
        The output is the same shape as the input.
    """
    if bary.shape[-1] != 3:
        msg = "Expected barycentric coords to have last dim = 3; got %r"
        raise ValueError(msg % (bary.shape,))
    ndims = bary.ndim - 1
    mask = bary.eq(-1).all(dim=-1, keepdim=True).expand(*((-1,) * ndims + (3,)))
    clipped = bary.clamp(min=0.0)
    clipped[mask] = 0.0
    clipped_sum = torch.clamp(clipped.sum(dim=-1, keepdim=True), min=1e-5)
    clipped = clipped / clipped_sum
    clipped[mask] = -1.0
    return clipped


def _interpolate_zbuf(
    pix_to_face: torch.Tensor, barycentric_coords: torch.Tensor, meshes
) -> torch.Tensor:
    """
    A helper function to calculate the z buffer for each pixel in the
    rasterized output.

    Args:
        pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
            of the faces (in the packed representation) which
            overlap each pixel in the image.
        barycentric_coords: FloatTensor of shape (N, H, W, K, 3) specifying
            the barycentric coordianates of each pixel
            relative to the faces (in the packed
            representation) which overlap the pixel.
        meshes: Meshes object representing a batch of meshes.

    Returns:
        zbuffer: (N, H, W, K) FloatTensor
    """
    verts = meshes.verts_packed()
    faces = meshes.faces_packed()
    faces_verts_z = verts[faces][..., 2][..., None]  # (F, 3, 1)
    zbuf = interpolate_face_attributes(pix_to_face, barycentric_coords, faces_verts_z)[
        ..., 0
    ]  # (1, H, W, K)
    zbuf[pix_to_face == -1] = -1
    return zbuf


# -----------  Rectangle Packing  -------------------- #

# Note the order of members matters here because it determines the queue order.
# We want to place longer rectangles first.
class _UnplacedRectangle(NamedTuple):
    size: Tuple[int, int]
    ind: int
    flipped: bool


def _try_place_rectangle(
    rect: _UnplacedRectangle,
    placed_so_far: List[Tuple[int, int, bool]],
    occupied: List[Tuple[int, int]],
) -> bool:
    """
    Try to place rect within the current bounding box.
    Part of the implementation of pack_rectangles.

    Note that the arguments `placed_so_far` and `occupied` are modified.

    Args:
        rect: rectangle to place
        placed_so_far: the locations decided upon so far - a list of
                    (x, y, whether flipped). The nth element is the
                    location of the nth rectangle if it has been decided.
                    (modified in place)
        occupied: the nodes of the graph of extents of rightmost placed
                    rectangles - (modified in place)

    Returns:
        True on success.

    Example:
    (We always have placed the first rectangle horizontally and other
    rectangles above it.)
    Let's say the placed boxes 1-4 are layed out like this.
    The coordinates of the points marked X are stored in occupied.
    It is to the right of the X's that we seek to place rect.

        +-----------------------X
        |2                      |
        |                       +---X
        |                       |4  |
        |                       |   |
        |                       +---+X
        |                       |3   |
        |                       |    |
        +-----------------------+----+------X
    y    |1                                  |
    ^    |     --->x                         |
    |    +-----------------------------------+

    We want to place this rectangle.

                +-+
                |5|
                | |
                | |   = rect
                | |
                | |
                | |
                +-+

    The call will succeed, returning True, leaving us with

        +-----------------------X
        |2                      |    +-X
        |                       +---+|5|
        |                       |4  || |
        |                       |   || |
        |                       +---++ |
        |                       |3   | |
        |                       |    | |
        +-----------------------+----+-+----X
        |1                                  |
        |                                   |
        +-----------------------------------+ .

    """
    total_width = occupied[0][0]
    needed_height = rect.size[1]
    current_start_idx = None
    current_max_width = 0
    previous_height = 0
    currently_packed = 0
    for idx, interval in enumerate(occupied):
        if interval[0] <= total_width - rect.size[0]:
            currently_packed += interval[1] - previous_height
            current_max_width = max(interval[0], current_max_width)
            if current_start_idx is None:
                current_start_idx = idx
            if currently_packed >= needed_height:
                current_max_width = max(interval[0], current_max_width)
                placed_so_far[rect.ind] = (
                    current_max_width,
                    occupied[current_start_idx - 1][1],
                    rect.flipped,
                )
                new_occupied = (
                    current_max_width + rect.size[0],
                    occupied[current_start_idx - 1][1] + needed_height,
                )
                if currently_packed == needed_height:
                    occupied[idx] = new_occupied
                    del occupied[current_start_idx:idx]
                elif idx > current_start_idx:
                    occupied[idx - 1] = new_occupied
                    del occupied[current_start_idx : (idx - 1)]
                else:
                    occupied.insert(idx, new_occupied)
                return True
        else:
            current_start_idx = None
            current_max_width = 0
            currently_packed = 0
        previous_height = interval[1]
    return False


class PackedRectangles(NamedTuple):
    total_size: Tuple[int, int]
    locations: List[Tuple[int, int, bool]]  # (x,y) and whether flipped


def pack_rectangles(sizes: List[Tuple[int, int]]) -> PackedRectangles:
    """
    Naive rectangle packing in to a large rectangle. Flipping (i.e. rotating
    a rectangle by 90 degrees) is allowed.

    This is used to join several uv maps into a single scene, see
    TexturesUV.join_scene.

    Args:
        sizes: List of sizes of rectangles to pack

    Returns:
        total_size: size of total large rectangle
        rectangles: location for each of the input rectangles
    """

    if len(sizes) < 2:
        raise ValueError("Cannot pack less than two boxes")

    queue = []
    for i, size in enumerate(sizes):
        if size[0] < size[1]:
            queue.append(_UnplacedRectangle((size[1], size[0]), i, True))
        else:
            queue.append(_UnplacedRectangle((size[0], size[1]), i, False))
    queue.sort()
    placed_so_far = [(-1, -1, False)] * len(sizes)

    biggest = queue.pop()
    total_width, current_height = biggest.size
    placed_so_far[biggest.ind] = (0, 0, biggest.flipped)

    second = queue.pop()
    placed_so_far[second.ind] = (0, current_height, second.flipped)
    current_height += second.size[1]
    occupied = [biggest.size, (second.size[0], current_height)]

    for rect in reversed(queue):
        if _try_place_rectangle(rect, placed_so_far, occupied):
            continue

        rotated = _UnplacedRectangle(
            (rect.size[1], rect.size[0]), rect.ind, not rect.flipped
        )
        if _try_place_rectangle(rotated, placed_so_far, occupied):
            continue

        # rect wasn't placed in the current bounding box,
        # so we add extra space to fit it in.
        placed_so_far[rect.ind] = (0, current_height, rect.flipped)
        current_height += rect.size[1]
        occupied.append((rect.size[0], current_height))

    return PackedRectangles((total_width, current_height), placed_so_far)
