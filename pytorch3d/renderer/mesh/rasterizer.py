#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import NamedTuple, Optional
import torch
import torch.nn as nn

from ..cameras import get_world_to_view_transform
from .rasterize_meshes import rasterize_meshes


# Class to store the outputs of mesh rasterization
class Fragments(NamedTuple):
    pix_to_face: torch.Tensor
    zbuf: torch.Tensor
    bary_coords: torch.Tensor
    dists: torch.Tensor


# Class to store the mesh rasterization params with defaults
class RasterizationSettings(NamedTuple):
    image_size: int = 256
    blur_radius: float = 0.0
    faces_per_pixel: int = 1
    bin_size: Optional[int] = 0
    max_faces_per_bin: Optional[int] = 100


class MeshRasterizer(nn.Module):
    """
    This class implements methods for rasterizing a batch of heterogenous
    Meshes.
    """

    def __init__(self, cameras, raster_settings=None):
        """
        Args:
            cameras: A cameras object which has a  `transform_points` method
                which returns the transformed points after applying the
                world-to-view and view-to-screen
                transformations.
            raster_settings: the parameters for rasterization. This should be a
                named tuple.

        All these initial settings can be overridden by passing keyword
        arguments to the forward function.
        """
        super().__init__()
        if raster_settings is None:
            raster_settings = RasterizationSettings()

        self.cameras = cameras
        self.raster_settings = raster_settings

    def transform(self, meshes_world, **kwargs) -> torch.Tensor:
        """
        Args:
            meshes_world: a Meshes object representing a batch of meshes with
                vertex coordinates in world space.

        Returns:
            meshes_screen: a Meshes object with the vertex positions in screen
            space

        NOTE: keeping this as a separate function for readability but it could
        be moved into forward.
        """
        cameras = kwargs.get("cameras", self.cameras)
        verts_world = meshes_world.verts_padded()
        verts_world_packed = meshes_world.verts_packed()
        verts_screen = cameras.transform_points(verts_world, **kwargs)

        # NOTE: Retaining view space z coordinate for now.
        # TODO: Remove this line when the convention for the z coordinate in
        # the rasterizer is decided. i.e. retain z in view space or transform
        # to a different range.
        view_transform = get_world_to_view_transform(R=cameras.R, T=cameras.T)
        verts_view = view_transform.transform_points(verts_world)
        verts_screen[..., 2] = verts_view[..., 2]

        # TODO also transform normals? Only necessary if view space normals
        # are used later in the pipeline.

        # Offset verts of input mesh to reuse cached padded/packed calculations.
        pad_to_packed_idx = meshes_world.verts_padded_to_packed_idx()
        verts_screen_packed = verts_screen.view(-1, 3)[pad_to_packed_idx, :]
        verts_packed_offset = verts_screen_packed - verts_world_packed
        return meshes_world.offset_verts(verts_packed_offset)

    def forward(self, meshes_world, **kwargs) -> Fragments:
        """
        Args:
            meshes_world: a Meshes object representing a batch of meshes with
                          coordinates in world space.
        Returns:
            Fragments: Rasterization outputs as a named tuple.
        """
        meshes_screen = self.transform(meshes_world, **kwargs)
        raster_settings = kwargs.get("raster_settings", self.raster_settings)
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=raster_settings.image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
        )
        return Fragments(
            pix_to_face=pix_to_face,
            zbuf=zbuf,
            bary_coords=bary_coords,
            dists=dists,
        )
