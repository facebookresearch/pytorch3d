# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import torch


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
    clipped = bary.clamp(min=0.0)
    clipped_sum = torch.clamp(clipped.sum(dim=-1, keepdim=True), min=1e-5)
    clipped = clipped / clipped_sum
    return clipped


def interpolate_face_attributes(
    pix_to_face: torch.Tensor,
    barycentric_coords: torch.Tensor,
    face_attributes: torch.Tensor,
) -> torch.Tensor:
    """
    Interpolate arbitrary face attributes using the barycentric coordinates
    for each pixel in the rasterized output.

    Args:
        pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
            of the faces (in the packed representation) which
            overlap each pixel in the image.
        barycentric_coords: FloatTensor of shape (N, H, W, K, 3) specifying
            the barycentric coordianates of each pixel
            relative to the faces (in the packed
            representation) which overlap the pixel.
        face_attributes: packed attributes of shape (total_faces, 3, D),
            specifying the value of the attribute for each
            vertex in the face.

    Returns:
        pixel_vals: tensor of shape (N, H, W, K, D) giving the interpolated
        value of the face attribute for each pixel.
    """
    F, FV, D = face_attributes.shape
    if FV != 3:
        raise ValueError("Faces can only have three vertices; got %r" % FV)
    N, H, W, K, _ = barycentric_coords.shape
    if pix_to_face.shape != (N, H, W, K):
        msg = "pix_to_face must have shape (batch_size, H, W, K); got %r"
        raise ValueError(msg % (pix_to_face.shape,))

    # Replace empty pixels in pix_to_face with 0 in order to interpolate.
    mask = pix_to_face == -1
    pix_to_face = pix_to_face.clone()
    pix_to_face[mask] = 0
    idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
    pixel_face_vals = face_attributes.gather(0, idx).view(N, H, W, K, 3, D)
    pixel_vals = (barycentric_coords[..., None] * pixel_face_vals).sum(dim=-2)
    pixel_vals[mask] = 0  # Replace masked values in output.
    return pixel_vals


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
    return interpolate_face_attributes(pix_to_face, barycentric_coords, faces_verts_z)[
        ..., 0
    ]  # (1, H, W, K)
