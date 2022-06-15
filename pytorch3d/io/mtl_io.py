# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""This module implements utility functions for loading .mtl files and textures."""
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from iopath.common.file_io import PathManager
from pytorch3d.common.compat import meshgrid_ij
from pytorch3d.common.datatypes import Device
from pytorch3d.io.utils import _open_file, _read_image


def make_mesh_texture_atlas(
    material_properties: Dict,
    texture_images: Dict,
    face_material_names,
    faces_uvs: torch.Tensor,
    verts_uvs: torch.Tensor,
    texture_size: int,
    texture_wrap: Optional[str],
) -> torch.Tensor:
    """
    Given properties for materials defined in the .mtl file, and the face texture uv
    coordinates, construct an (F, R, R, 3) texture atlas where R is the texture_size
    and F is the number of faces in the mesh.

    Args:
        material_properties: dict of properties for each material. If a material
                does not have any properties it will have an empty dict.
        texture_images: dict of material names and texture images
        face_material_names: numpy array of the material name corresponding to each
            face. Faces which don't have an associated material will be an empty string.
            For these faces, a uniform white texture is assigned.
        faces_uvs: LongTensor of shape (F, 3,) giving the index into the verts_uvs for
            each face in the mesh.
        verts_uvs: FloatTensor of shape (V, 2) giving the uv coordinates for each vertex.
        texture_size: the resolution of the per face texture map returned by this function.
            Each face will have a texture map of shape (texture_size, texture_size, 3).
        texture_wrap: string, one of ["repeat", "clamp", None]
            If `texture_wrap="repeat"` for uv values outside the range [0, 1] the integer part
            is ignored and a repeating pattern is formed.
            If `texture_wrap="clamp"` the values are clamped to the range [0, 1].
            If None, do nothing.

    Returns:
        atlas: FloatTensor of shape (F, texture_size, texture_size, 3) giving the per
        face texture map.
    """
    # Create an R x R texture map per face in the mesh
    R = texture_size
    F = faces_uvs.shape[0]

    # Initialize the per face texture map to a white color.
    # TODO: allow customization of this base color?
    atlas = torch.ones(size=(F, R, R, 3), dtype=torch.float32, device=faces_uvs.device)

    # Check for empty materials.
    if not material_properties and not texture_images:
        return atlas

    # Iterate through the material properties - not
    # all materials have texture images so this is
    # done first separately to the texture interpolation.
    for material_name, props in material_properties.items():
        # Bool to indicate which faces use this texture map.
        faces_material_ind = torch.from_numpy(face_material_names == material_name).to(
            faces_uvs.device
        )
        if faces_material_ind.sum() > 0:
            # For these faces, update the base color to the
            # diffuse material color.
            if "diffuse_color" not in props:
                continue
            atlas[faces_material_ind, ...] = props["diffuse_color"][None, :]

    # If there are vertex texture coordinates, create an (F, 3, 2)
    # tensor of the vertex textures per face.
    faces_verts_uvs = verts_uvs[faces_uvs] if len(verts_uvs) > 0 else None

    # Some meshes only have material properties and no texture image.
    # In this case, return the atlas here.
    if faces_verts_uvs is None:
        return atlas

    if texture_wrap == "repeat":
        # If texture uv coordinates are outside the range [0, 1] follow
        # the convention GL_REPEAT in OpenGL i.e the integer part of the coordinate
        # will be ignored and a repeating pattern is formed.
        # Shapenet data uses this format see:
        # https://shapenet.org/qaforum/index.php?qa=15&qa_1=why-is-the-texture-coordinate-in-the-obj-file-not-in-the-range # noqa: B950
        if (faces_verts_uvs > 1).any() or (faces_verts_uvs < 0).any():
            msg = "Texture UV coordinates outside the range [0, 1]. \
                The integer part will be ignored to form a repeating pattern."
            warnings.warn(msg)
            faces_verts_uvs = faces_verts_uvs % 1
    elif texture_wrap == "clamp":
        # Clamp uv coordinates to the [0, 1] range.
        faces_verts_uvs = faces_verts_uvs.clamp(0.0, 1.0)

    # Iterate through the materials used in this mesh. Update the
    # texture atlas for the faces which use this material.
    # Faces without texture are white.
    for material_name, image in list(texture_images.items()):
        # Only use the RGB colors
        if image.shape[2] == 4:
            image = image[:, :, :3]

        # Reverse the image y direction
        image = torch.flip(image, [0]).type_as(faces_verts_uvs)

        # Bool to indicate which faces use this texture map.
        faces_material_ind = torch.from_numpy(face_material_names == material_name).to(
            faces_verts_uvs.device
        )

        # Find the subset of faces which use this texture with this texture image
        uvs_subset = faces_verts_uvs[faces_material_ind, :, :]

        # Update the texture atlas for the faces which use this texture.
        # TODO: should the texture map values be multiplied
        # by the diffuse material color (i.e. use *= as the atlas has
        # been initialized to the diffuse color)?. This is
        # not being done in SoftRas.
        atlas[faces_material_ind, :, :] = make_material_atlas(image, uvs_subset, R)

    return atlas


def make_material_atlas(
    image: torch.Tensor, faces_verts_uvs: torch.Tensor, texture_size: int
) -> torch.Tensor:
    r"""
    Given a single texture image and the uv coordinates for all the
    face vertices, create a square texture map per face using
    the formulation from [1].

    For a triangle with vertices (v0, v1, v2) we can create a barycentric coordinate system
    with the x axis being the vector (v0 - v2) and the y axis being the vector (v1 - v2).
    The barycentric coordinates range from [0, 1] in the +x and +y direction so this creates
    a triangular texture space with vertices at (0, 1), (0, 0) and (1, 0).

    The per face texture map is of shape (texture_size, texture_size, 3)
    which is a square. To map a triangular texture to a square grid, each
    triangle is parametrized as follows (e.g. R = texture_size = 3):

    The triangle texture is first divided into RxR = 9 subtriangles which each
    map to one grid cell. The numbers in the grid cells and triangles show the mapping.

    ..code-block::python

        Triangular Texture Space:

              1
                |\
                |6 \
                |____\
                |\  7 |\
                |3 \  |4 \
                |____\|____\
                |\ 8  |\  5 |\
                |0 \  |1 \  |2 \
                |____\|____\|____\
               0                   1

        Square per face texture map:

               R ____________________
                |      |      |      |
                |  6   |  7   |  8   |
                |______|______|______|
                |      |      |      |
                |  3   |  4   |  5   |
                |______|______|______|
                |      |      |      |
                |  0   |  1   |  2   |
                |______|______|______|
               0                      R


    The barycentric coordinates of each grid cell are calculated using the
    xy coordinates:

    ..code-block::python

            The cartesian coordinates are:

            Grid 1:

               R ____________________
                |      |      |      |
                |  20  |  21  |  22  |
                |______|______|______|
                |      |      |      |
                |  10  |  11  |  12  |
                |______|______|______|
                |      |      |      |
                |  00  |  01  |  02  |
                |______|______|______|
               0                      R

            where 02 means y = 0, x = 2

        Now consider this subset of the triangle which corresponds to
        grid cells 0 and 8:

        ..code-block::python

            1/R  ________
                |\    8  |
                |  \     |
                | 0   \  |
                |_______\|
               0          1/R

        The centroids of the triangles are:
            0: (1/3, 1/3) * 1/R
            8: (2/3, 2/3) * 1/R

    For each grid cell we can now calculate the centroid `(c_y, c_x)`
    of the corresponding texture triangle:
        - if `(x + y) < R`, then offset the centroid of
            triangle 0 by `(y, x) * (1/R)`
        - if `(x + y) > R`, then offset the centroid of
            triangle 8 by `((R-1-y), (R-1-x)) * (1/R)`.

    This is equivalent to updating the portion of Grid 1
    above the diagonal, replacing `(y, x)` with `((R-1-y), (R-1-x))`:

    ..code-block::python

              R _____________________
                |      |      |      |
                |  20  |  01  |  00  |
                |______|______|______|
                |      |      |      |
                |  10  |  11  |  10  |
                |______|______|______|
                |      |      |      |
                |  00  |  01  |  02  |
                |______|______|______|
               0                      R

    The barycentric coordinates (w0, w1, w2) are then given by:

    ..code-block::python

        w0 = c_x
        w1 = c_y
        w2 = 1- w0 - w1

    Args:
        image: FloatTensor of shape (H, W, 3)
        faces_verts_uvs: uv coordinates for each vertex in each face  (F, 3, 2)
        texture_size: int

    Returns:
        atlas: a FloatTensor of shape (F, texture_size, texture_size, 3) giving a
            per face texture map.

    [1] Liu et al, 'Soft Rasterizer: A Differentiable Renderer for Image-based
        3D Reasoning', ICCV 2019
    """
    R = texture_size
    device = faces_verts_uvs.device
    rng = torch.arange(R, device=device)

    # Meshgrid returns (row, column) i.e (Y, X)
    # Change order to (X, Y) to make the grid.
    Y, X = meshgrid_ij(rng, rng)
    # pyre-fixme[28]: Unexpected keyword argument `axis`.
    grid = torch.stack([X, Y], axis=-1)  # (R, R, 2)

    # Grid cells below the diagonal: x + y < R.
    below_diag = grid.sum(-1) < R

    # map a [0, R] grid -> to a [0, 1] barycentric coordinates of
    # the texture triangle centroids.
    bary = torch.zeros((R, R, 3), device=device)  # (R, R, 3)
    slc = torch.arange(2, device=device)[:, None]
    # w0, w1
    bary[below_diag, slc] = ((grid[below_diag] + 1.0 / 3.0) / R).T
    # w0, w1 for above diagonal grid cells.
    bary[~below_diag, slc] = (((R - 1.0 - grid[~below_diag]) + 2.0 / 3.0) / R).T
    # w2 = 1. - w0 - w1
    bary[..., -1] = 1 - bary[..., :2].sum(dim=-1)

    # Calculate the uv position in the image for each pixel
    # in the per face texture map
    # (F, 1, 1, 3, 2) * (R, R, 3, 1) -> (F, R, R, 3, 2) -> (F, R, R, 2)
    uv_pos = (faces_verts_uvs[:, None, None] * bary[..., None]).sum(-2)

    # bi-linearly interpolate the textures from the images
    # using the uv coordinates given by uv_pos.
    textures = _bilinear_interpolation_grid_sample(image, uv_pos)

    return textures


def _bilinear_interpolation_vectorized(
    image: torch.Tensor, grid: torch.Tensor
) -> torch.Tensor:
    """
    Bi linearly interpolate the image using the uv positions in the flow-field
    grid (following the naming conventions for torch.nn.functional.grid_sample).

    This implementation uses the same steps as in the SoftRasterizer CUDA kernel
    for loading textures. We are keeping it for reference to make it easy to
    compare if required.

    However it doesn't properly handle the out of bound values in the same way as
    the grid_sample function does with the padding_mode argument.
    This vectorized version requires less memory than
    _bilinear_interpolation_grid_sample but is slightly slower.

    Args:
        image: FloatTensor of shape (H, W, D) a single image/input tensor with D
            channels.
        grid: FloatTensor of shape (N, R, R, 2) giving the pixel locations of the
            points at which to sample a value in the image. The grid values must
            be in the range [0, 1]. u is the x direction and v is the y direction.

    Returns:
        out: FloatTensor of shape (N, H, W, D) giving the interpolated
            D dimensional value from image at each of the pixel locations in grid.

    """
    H, W, _ = image.shape
    # Convert [0, 1] to the range [0, W-1] and [0, H-1]
    grid = grid * torch.tensor([W - 1, H - 1]).type_as(grid)
    weight_1 = grid - grid.int()
    weight_0 = 1.0 - weight_1

    grid_x, grid_y = grid.unbind(-1)
    y0 = grid_y.to(torch.int64)
    y1 = (grid_y + 1).to(torch.int64)
    x0 = grid_x.to(torch.int64)
    x1 = x0 + 1

    weight_x0, weight_y0 = weight_0.unbind(-1)
    weight_x1, weight_y1 = weight_1.unbind(-1)

    # Bi-linear interpolation
    # griditions = [[y,     x], [(y+1),     x]
    #              [y, (x+1)], [(y+1), (x+1)]]
    # weights   = [[wx0*wy0, wx0*wy1],
    #              [wx1*wy0, wx1*wy1]]
    out = (
        image[y0, x0] * (weight_x0 * weight_y0)[..., None]
        + image[y1, x0] * (weight_x0 * weight_y1)[..., None]
        + image[y0, x1] * (weight_x1 * weight_y0)[..., None]
        + image[y1, x1] * (weight_x1 * weight_y1)[..., None]
    )

    return out


def _bilinear_interpolation_grid_sample(
    image: torch.Tensor, grid: torch.Tensor
) -> torch.Tensor:
    """
    Bi linearly interpolate the image using the uv positions in the flow-field
    grid (following the conventions for torch.nn.functional.grid_sample).

    This implementation is faster than _bilinear_interpolation_vectorized but
    requires more memory so can cause OOMs. If speed is an issue try this function
    instead.

    Args:
        image: FloatTensor of shape (H, W, D) a single image/input tensor with D
            channels.
        grid: FloatTensor of shape (N, R, R, 2) giving the pixel locations of the
            points at which to sample a value in the image. The grid values must
            be in the range [0, 1]. u is the x direction and v is the y direction.

    Returns:
        out: FloatTensor of shape (N, H, W, D) giving the interpolated
            D dimensional value from image at each of the pixel locations in grid.
    """

    N = grid.shape[0]
    # convert [0, 1] to the range [-1, 1] expected by grid_sample.
    grid = grid * 2.0 - 1.0
    image = image.permute(2, 0, 1)[None, ...].expand(N, -1, -1, -1)  # (N, 3, H, W)
    # Align_corners has to be set to True to match the output of the SoftRas
    # cuda kernel for bilinear sampling.
    out = F.grid_sample(image, grid, mode="bilinear", align_corners=True)
    return out.permute(0, 2, 3, 1)


MaterialProperties = Dict[str, Dict[str, torch.Tensor]]
TextureFiles = Dict[str, str]
TextureImages = Dict[str, torch.Tensor]


def _parse_mtl(
    f: str, path_manager: PathManager, device: Device = "cpu"
) -> Tuple[MaterialProperties, TextureFiles]:
    material_properties = {}
    texture_files = {}
    material_name = ""

    with _open_file(f, path_manager, "r") as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue
            if tokens[0] == "newmtl":
                material_name = tokens[1]
                material_properties[material_name] = {}
            elif tokens[0] == "map_Kd":
                # Diffuse texture map
                # Account for the case where filenames might have spaces
                filename = line.strip()[7:]
                texture_files[material_name] = filename
            elif tokens[0] == "Kd":
                # RGB diffuse reflectivity
                kd = np.array(tokens[1:4]).astype(np.float32)
                kd = torch.from_numpy(kd).to(device)
                material_properties[material_name]["diffuse_color"] = kd
            elif tokens[0] == "Ka":
                # RGB ambient reflectivity
                ka = np.array(tokens[1:4]).astype(np.float32)
                ka = torch.from_numpy(ka).to(device)
                material_properties[material_name]["ambient_color"] = ka
            elif tokens[0] == "Ks":
                # RGB specular reflectivity
                ks = np.array(tokens[1:4]).astype(np.float32)
                ks = torch.from_numpy(ks).to(device)
                material_properties[material_name]["specular_color"] = ks
            elif tokens[0] == "Ns":
                # Specular exponent
                ns = np.array(tokens[1:4]).astype(np.float32)
                ns = torch.from_numpy(ns).to(device)
                material_properties[material_name]["shininess"] = ns

    return material_properties, texture_files


def _load_texture_images(
    material_names: List[str],
    data_dir: str,
    material_properties: MaterialProperties,
    texture_files: TextureFiles,
    path_manager: PathManager,
) -> Tuple[MaterialProperties, TextureImages]:
    final_material_properties = {}
    texture_images = {}

    used_material_names = list(material_names)
    if not used_material_names and material_properties:
        if len(material_properties) > 1:
            raise ValueError(
                "Multiple materials but no usemtl declarations in the obj file"
            )
        # No materials were specified in obj file and only one is in the
        # specified .mtl file, so we use it.
        used_material_names.append(next(iter(material_properties.keys())))

    # Only keep the materials referenced in the obj.
    for material_name in used_material_names:
        if material_name in texture_files:
            # Load the texture image.
            path = os.path.join(data_dir, texture_files[material_name])
            if path_manager.exists(path):
                image = (
                    _read_image(path, path_manager=path_manager, format="RGB") / 255.0
                )
                image = torch.from_numpy(image)
                texture_images[material_name] = image
            else:
                msg = f"Texture file does not exist: {path}"
                warnings.warn(msg)

        if material_name in material_properties:
            final_material_properties[material_name] = material_properties[
                material_name
            ]

    return final_material_properties, texture_images


def load_mtl(
    f: str,
    *,
    material_names: List[str],
    data_dir: str,
    device: Device = "cpu",
    path_manager: PathManager,
) -> Tuple[MaterialProperties, TextureImages]:
    """
    Load texture images and material reflectivity values for ambient, diffuse
    and specular light (Ka, Kd, Ks, Ns).

    Args:
        f: path to the material information.
        material_names: a list of the material names found in the .obj file.
        data_dir: the directory where the material texture files are located.
        device: Device (as str or torch.tensor) on which to return the new tensors.
        path_manager: PathManager for interpreting both f and material_names.

    Returns:
        material_properties: dict of properties for each material. If a material
                does not have any properties it will have an empty dict.
                {
                    material_name_1:  {
                        "ambient_color": tensor of shape (1, 3),
                        "diffuse_color": tensor of shape (1, 3),
                        "specular_color": tensor of shape (1, 3),
                        "shininess": tensor of shape (1)
                    },
                    material_name_2: {},
                    ...
                }
        texture_images: dict of material names and texture images
                {
                    material_name_1: (H, W, 3) image,
                    ...
                }
    """
    material_properties, texture_files = _parse_mtl(f, path_manager, device)
    return _load_texture_images(
        material_names,
        data_dir,
        material_properties,
        texture_files,
        path_manager=path_manager,
    )
