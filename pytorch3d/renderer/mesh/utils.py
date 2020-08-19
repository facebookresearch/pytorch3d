# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


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
