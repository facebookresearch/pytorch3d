# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import torch
from pytorch3d.ops import interpolate_face_attributes

from .textures import TexturesVertex


def _apply_lighting(
    points, normals, lights, cameras, materials
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        points: torch tensor of shape (N, ..., 3) or (P, 3).
        normals: torch tensor of shape (N, ..., 3) or (P, 3)
        lights: instance of the Lights class.
        cameras: instance of the Cameras class.
        materials: instance of the Materials class.

    Returns:
        ambient_color: same shape as materials.ambient_color
        diffuse_color: same shape as the input points
        specular_color: same shape as the input points
    """
    light_diffuse = lights.diffuse(normals=normals, points=points)
    light_specular = lights.specular(
        normals=normals,
        points=points,
        camera_position=cameras.get_camera_center(),
        shininess=materials.shininess,
    )
    ambient_color = materials.ambient_color * lights.ambient_color
    diffuse_color = materials.diffuse_color * light_diffuse
    specular_color = materials.specular_color * light_specular

    if normals.dim() == 2 and points.dim() == 2:
        # If given packed inputs remove batch dim in output.
        return (
            ambient_color.squeeze(),
            diffuse_color.squeeze(),
            specular_color.squeeze(),
        )

    if ambient_color.ndim != diffuse_color.ndim:
        # Reshape from (N, 3) to have dimensions compatible with
        # diffuse_color which is of shape (N, H, W, K, 3)
        ambient_color = ambient_color[:, None, None, None, :]
    return ambient_color, diffuse_color, specular_color


def _phong_shading_with_pixels(
    meshes, fragments, lights, cameras, materials, texels
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply per pixel shading. First interpolate the vertex normals and
    vertex coordinates using the barycentric coordinates to get the position
    and normal at each pixel. Then compute the illumination for each pixel.
    The pixel color is obtained by multiplying the pixel textures by the ambient
    and diffuse illumination and adding the specular component.

    Args:
        meshes: Batch of meshes
        fragments: Fragments named tuple with the outputs of rasterization
        lights: Lights class containing a batch of lights
        cameras: Cameras class containing a batch of cameras
        materials: Materials class containing a batch of material properties
        texels: texture per pixel of shape (N, H, W, K, 3)

    Returns:
        colors: (N, H, W, K, 3)
        pixel_coords: (N, H, W, K, 3), camera coordinates of each intersection.
    """
    verts = meshes.verts_packed()  # (V, 3)
    faces = meshes.faces_packed()  # (F, 3)
    vertex_normals = meshes.verts_normals_packed()  # (V, 3)
    faces_verts = verts[faces]
    faces_normals = vertex_normals[faces]
    pixel_coords_in_camera = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_verts
    )
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_normals
    )
    ambient, diffuse, specular = _apply_lighting(
        pixel_coords_in_camera, pixel_normals, lights, cameras, materials
    )
    colors = (ambient + diffuse) * texels + specular
    return colors, pixel_coords_in_camera


def phong_shading(
    meshes, fragments, lights, cameras, materials, texels
) -> torch.Tensor:
    """
    Apply per pixel shading. First interpolate the vertex normals and
    vertex coordinates using the barycentric coordinates to get the position
    and normal at each pixel. Then compute the illumination for each pixel.
    The pixel color is obtained by multiplying the pixel textures by the ambient
    and diffuse illumination and adding the specular component.

    Args:
        meshes: Batch of meshes
        fragments: Fragments named tuple with the outputs of rasterization
        lights: Lights class containing a batch of lights
        cameras: Cameras class containing a batch of cameras
        materials: Materials class containing a batch of material properties
        texels: texture per pixel of shape (N, H, W, K, 3)

    Returns:
        colors: (N, H, W, K, 3)
    """
    colors, _ = _phong_shading_with_pixels(
        meshes, fragments, lights, cameras, materials, texels
    )
    return colors


def gouraud_shading(meshes, fragments, lights, cameras, materials) -> torch.Tensor:
    """
    Apply per vertex shading. First compute the vertex illumination by applying
    ambient, diffuse and specular lighting. If vertex color is available,
    combine the ambient and diffuse vertex illumination with the vertex color
    and add the specular component to determine the vertex shaded color.
    Then interpolate the vertex shaded colors using the barycentric coordinates
    to get a color per pixel.

    Gouraud shading is only supported for meshes with texture type `TexturesVertex`.
    This is because the illumination is applied to the vertex colors.

    Args:
        meshes: Batch of meshes
        fragments: Fragments named tuple with the outputs of rasterization
        lights: Lights class containing a batch of lights parameters
        cameras: Cameras class containing a batch of cameras parameters
        materials: Materials class containing a batch of material properties

    Returns:
        colors: (N, H, W, K, 3)
    """
    if not isinstance(meshes.textures, TexturesVertex):
        raise ValueError("Mesh textures must be an instance of TexturesVertex")

    faces = meshes.faces_packed()  # (F, 3)
    verts = meshes.verts_packed()  # (V, 3)
    verts_normals = meshes.verts_normals_packed()  # (V, 3)
    verts_colors = meshes.textures.verts_features_packed()  # (V, D)
    vert_to_mesh_idx = meshes.verts_packed_to_mesh_idx()

    # Format properties of lights and materials so they are compatible
    # with the packed representation of the vertices. This transforms
    # all tensor properties in the class from shape (N, ...) -> (V, ...) where
    # V is the number of packed vertices. If the number of meshes in the
    # batch is one then this is not necessary.
    if len(meshes) > 1:
        lights = lights.clone().gather_props(vert_to_mesh_idx)
        cameras = cameras.clone().gather_props(vert_to_mesh_idx)
        materials = materials.clone().gather_props(vert_to_mesh_idx)

    # Calculate the illumination at each vertex
    ambient, diffuse, specular = _apply_lighting(
        verts, verts_normals, lights, cameras, materials
    )

    verts_colors_shaded = verts_colors * (ambient + diffuse) + specular
    face_colors = verts_colors_shaded[faces]
    colors = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, face_colors
    )
    return colors


def flat_shading(meshes, fragments, lights, cameras, materials, texels) -> torch.Tensor:
    """
    Apply per face shading. Use the average face position and the face normals
    to compute the ambient, diffuse and specular lighting. Apply the ambient
    and diffuse color to the pixel color and add the specular component to
    determine the final pixel color.

    Args:
        meshes: Batch of meshes
        fragments: Fragments named tuple with the outputs of rasterization
        lights: Lights class containing a batch of lights parameters
        cameras: Cameras class containing a batch of cameras parameters
        materials: Materials class containing a batch of material properties
        texels: texture per pixel of shape (N, H, W, K, 3)

    Returns:
        colors: (N, H, W, K, 3)
    """
    verts = meshes.verts_packed()  # (V, 3)
    faces = meshes.faces_packed()  # (F, 3)
    face_normals = meshes.faces_normals_packed()  # (V, 3)
    faces_verts = verts[faces]
    face_coords = faces_verts.mean(dim=-2)  # (F, 3, XYZ) mean xyz across verts

    # Replace empty pixels in pix_to_face with 0 in order to interpolate.
    mask = fragments.pix_to_face == -1
    pix_to_face = fragments.pix_to_face.clone()
    pix_to_face[mask] = 0

    N, H, W, K = pix_to_face.shape
    idx = pix_to_face.view(N * H * W * K, 1).expand(N * H * W * K, 3)

    # gather pixel coords
    pixel_coords = face_coords.gather(0, idx).view(N, H, W, K, 3)
    pixel_coords[mask] = 0.0
    # gather pixel normals
    pixel_normals = face_normals.gather(0, idx).view(N, H, W, K, 3)
    pixel_normals[mask] = 0.0

    # Calculate the illumination at each face
    ambient, diffuse, specular = _apply_lighting(
        pixel_coords, pixel_normals, lights, cameras, materials
    )
    colors = (ambient + diffuse) * texels + specular
    return colors
