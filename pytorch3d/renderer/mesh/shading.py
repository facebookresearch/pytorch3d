# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


from typing import Tuple

import torch

from .texturing import interpolate_face_attributes


def _apply_lighting(
    points, normals, lights, cameras, materials
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        points: torch tensor of shape (N, P, 3) or (P, 3).
        normals: torch tensor of shape (N, P, 3) or (P, 3)
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
    return ambient_color, diffuse_color, specular_color


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
    verts = meshes.verts_packed()  # (V, 3)
    faces = meshes.faces_packed()  # (F, 3)
    vertex_normals = meshes.verts_normals_packed()  # (V, 3)
    faces_verts = verts[faces]
    faces_normals = vertex_normals[faces]
    pixel_coords = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_verts
    )
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_normals
    )
    ambient, diffuse, specular = _apply_lighting(
        pixel_coords, pixel_normals, lights, cameras, materials
    )
    colors = (ambient + diffuse) * texels + specular
    return colors


def gouraud_shading(meshes, fragments, lights, cameras, materials) -> torch.Tensor:
    """
    Apply per vertex shading. First compute the vertex illumination by applying
    ambient, diffuse and specular lighting. If vertex color is available,
    combine the ambient and diffuse vertex illumination with the vertex color
    and add the specular component to determine the vertex shaded color.
    Then interpolate the vertex shaded colors using the barycentric coordinates
    to get a color per pixel.

    Args:
        meshes: Batch of meshes
        fragments: Fragments named tuple with the outputs of rasterization
        lights: Lights class containing a batch of lights parameters
        cameras: Cameras class containing a batch of cameras parameters
        materials: Materials class containing a batch of material properties

    Returns:
        colors: (N, H, W, K, 3)
    """
    faces = meshes.faces_packed()  # (F, 3)
    verts = meshes.verts_packed()
    vertex_normals = meshes.verts_normals_packed()  # (V, 3)
    vertex_colors = meshes.textures.verts_rgb_packed()
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
        verts, vertex_normals, lights, cameras, materials
    )
    verts_colors_shaded = vertex_colors * (ambient + diffuse) + specular
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
    pixel_coords = face_coords[fragments.pix_to_face]
    pixel_normals = face_normals[fragments.pix_to_face]

    # Calculate the illumination at each face
    ambient, diffuse, specular = _apply_lighting(
        pixel_coords, pixel_normals, lights, cameras, materials
    )
    colors = (ambient + diffuse) * texels + specular
    return colors
