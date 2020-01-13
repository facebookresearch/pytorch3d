#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import torch
import torch.nn.functional as F

from pytorch3d.structures.textures import Textures


def _clip_barycentric_coordinates(bary) -> torch.Tensor:
    """
    Args:
        bary: barycentric coordinates of shape (...., 3) where `...` represents
            an arbitrary number of dimensions

    Returns:
        bary: All barycentric coordinate values clipped to the range [0, 1]
        and renormalized. The output is the same shape as the input.
    """
    if bary.shape[-1] != 3:
        msg = "Expected barycentric coords to have last dim = 3; got %r"
        raise ValueError(msg % bary.shape)
    clipped = bary.clamp(min=0, max=1)
    clipped_sum = torch.clamp(clipped.sum(dim=-1, keepdim=True), min=1e-5)
    clipped = clipped / clipped_sum
    return clipped


def interpolate_face_attributes(
    fragments, face_attributes: torch.Tensor, bary_clip: bool = False
) -> torch.Tensor:
    """
    Interpolate arbitrary face attributes using the barycentric coordinates
    for each pixel in the rasterized output.

    Args:
        fragments:
            The outputs of rasterization. From this we use

            - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
              of the faces (in the packed representation) which
              overlap each pixel in the image.
            - barycentric_coords: FloatTensor of shape (N, H, W, K, 3) specifying
              the barycentric coordianates of each pixel
              relative to the faces (in the packed
              representation) which overlap the pixel.
        face_attributes: packed attributes of shape (total_faces, 3, D),
            specifying the value of the attribute for each
            vertex in the face.
        bary_clip: Bool to indicate if barycentric_coords should be clipped
            before being used for interpolation.

    Returns:
        pixel_vals: tensor of shape (N, H, W, K, D) giving the interpolated
        value of the face attribute for each pixel.
    """
    pix_to_face = fragments.pix_to_face
    barycentric_coords = fragments.bary_coords
    F, FV, D = face_attributes.shape
    if FV != 3:
        raise ValueError("Faces can only have three vertices; got %r" % FV)
    N, H, W, K, _ = barycentric_coords.shape
    if pix_to_face.shape != (N, H, W, K):
        msg = "pix_to_face must have shape (batch_size, H, W, K); got %r"
        raise ValueError(msg % pix_to_face.shape)
    if bary_clip:
        barycentric_coords = _clip_barycentric_coordinates(barycentric_coords)

    # Replace empty pixels in pix_to_face with 0 in order to interpolate.
    mask = pix_to_face == -1
    pix_to_face = pix_to_face.clone()
    pix_to_face[mask] = 0
    idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
    pixel_face_vals = face_attributes.gather(0, idx).view(N, H, W, K, 3, D)
    pixel_vals = (barycentric_coords[..., None] * pixel_face_vals).sum(dim=-2)
    pixel_vals[mask] = 0  # Replace masked values in output.
    return pixel_vals


def interpolate_texture_map(fragments, meshes) -> torch.Tensor:
    """
    Interpolate a 2D texture map using uv vertex texture coordinates for each
    face in the mesh. First interpolate the vertex uvs using barycentric coordinates
    for each pixel in the rasterized output. Then interpolate the texture map
    using the uv coordinate for each pixel.

    Args:
        fragments:
            The outputs of rasterization. From this we use

            - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
              of the faces (in the packed representation) which
              overlap each pixel in the image.
            - barycentric_coords: FloatTensor of shape (N, H, W, K, 3) specifying
              the barycentric coordianates of each pixel
              relative to the faces (in the packed
              representation) which overlap the pixel.
        meshes: Meshes representing a batch of meshes. It is expected that
                meshes has a textures attribute which is an instance of the
                Textures class.

    Returns:
        texels: tensor of shape (N, H, W, K, C) giving the interpolated
        texture for each pixel in the rasterized image.
    """
    if not isinstance(meshes.textures, Textures):
        msg = "Expected meshes.textures to be an instance of Textures; got %r"
        raise ValueError(msg % type(meshes.textures))

    faces_uvs = meshes.textures.faces_uvs_packed()
    verts_uvs = meshes.textures.verts_uvs_packed()
    faces_verts_uvs = verts_uvs[faces_uvs]
    texture_maps = meshes.textures.maps_padded()

    # pixel_uvs: (N, H, W, K, 2)
    pixel_uvs = interpolate_face_attributes(fragments, faces_verts_uvs)

    N, H_out, W_out, K = fragments.pix_to_face.shape
    N, H_in, W_in, C = texture_maps.shape  # 3 for RGB

    # pixel_uvs: (N, H, W, K, 2) -> (N, K, H, W, 2) -> (NK, H, W, 2)
    pixel_uvs = pixel_uvs.permute(0, 3, 1, 2, 4).view(N * K, H_out, W_out, 2)

    # textures.map:
    #   (N, H, W, C) -> (N, C, H, W) -> (1, N, C, H, W)
    #   -> expand (K, N, C, H, W) -> reshape (N*K, C, H, W)
    texture_maps = (
        texture_maps.permute(0, 3, 1, 2)[None, ...]
        .expand(K, -1, -1, -1, -1)
        .transpose(0, 1)
        .reshape(N * K, C, H_in, W_in)
    )

    # Textures: (N*K, C, H, W), pixel_uvs: (N*K, H, W, 2)
    # Now need to format the pixel uvs and the texture map correctly!
    # From pytorch docs, grid_sample takes `grid` and `input`:
    #   grid specifies the sampling pixel locations normalized by
    #   the input spatial dimensions It should have most
    #   values in the range of [-1, 1]. Values x = -1, y = -1
    #   is the left-top pixel of input, and values x = 1, y = 1 is the
    #   right-bottom pixel of input.

    pixel_uvs = pixel_uvs * 2.0 - 1.0
    texture_maps = torch.flip(
        texture_maps, [2]
    )  # flip y axis of the texture map
    if texture_maps.device != pixel_uvs.device:
        texture_maps = texture_maps.to(pixel_uvs.device)
    texels = F.grid_sample(texture_maps, pixel_uvs, align_corners=False)
    texels = texels.view(N, K, C, H_out, W_out).permute(0, 3, 4, 1, 2)
    return texels


def interpolate_vertex_colors(fragments, meshes) -> torch.Tensor:
    """
    Detemine the color for each rasterized face. Interpolate the colors for
    vertices which form the face using the barycentric coordinates.
    Args:
        meshes: A Meshes class representing a batch of meshes.
        fragments:
            The outputs of rasterization. From this we use

            - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
              of the faces (in the packed representation) which
              overlap each pixel in the image.
            - barycentric_coords: FloatTensor of shape (N, H, W, K, 3) specifying
              the barycentric coordianates of each pixel
              relative to the faces (in the packed
              representation) which overlap the pixel.

    Returns:
        texels: An texture per pixel of shape (N, H, W, K, C).
        There will be one C dimensional value for each element in
        fragments.pix_to_face.
    """
    vertex_textures = meshes.textures.verts_rgb_padded().view(-1, 3)  # (V, C)
    vertex_textures = vertex_textures[meshes.verts_padded_to_packed_idx(), :]
    faces_packed = meshes.faces_packed()
    faces_textures = vertex_textures[faces_packed]  # (F, 3, C)
    texels = interpolate_face_attributes(fragments, faces_textures)
    return texels
