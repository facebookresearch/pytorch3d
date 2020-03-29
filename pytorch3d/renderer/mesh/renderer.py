# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import torch
import torch.nn as nn

from .rasterizer import Fragments
from .utils import _clip_barycentric_coordinates, _interpolate_zbuf


# A renderer class should be initialized with a
# function for rasterization and a function for shading.
# The rasterizer should:
#     - transform inputs from world -> screen space
#     - rasterize inputs
#     - return fragments
# The shader can take fragments as input along with any other properties of
# the scene and generate images.

# E.g. rasterize inputs and then shade
#
# fragments = self.rasterize(meshes)
# images = self.shader(fragments, meshes)
# return images


class MeshRenderer(nn.Module):
    """
    A class for rendering a batch of heterogeneous meshes. The class should
    be initialized with a rasterizer and shader class which each have a forward
    function.
    """

    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        """
        Render a batch of images from a batch of meshes by rasterizing and then
        shading.

        NOTE: If the blur radius for rasterization is > 0.0, some pixels can
        have one or more barycentric coordinates lying outside the range [0, 1].
        For a pixel with out of bounds barycentric coordinates with respect to a
        face f, clipping is required before interpolating the texture uv
        coordinates and z buffer so that the colors and depths are limited to
        the range for the corresponding face.
        """
        fragments = self.rasterizer(meshes_world, **kwargs)
        raster_settings = kwargs.get("raster_settings", self.rasterizer.raster_settings)
        if raster_settings.blur_radius > 0.0:
            # TODO: potentially move barycentric clipping to the rasterizer
            # if no downstream functions requires unclipped values.
            # This will avoid unnecssary re-interpolation of the z buffer.
            clipped_bary_coords = _clip_barycentric_coordinates(fragments.bary_coords)
            clipped_zbuf = _interpolate_zbuf(
                fragments.pix_to_face, clipped_bary_coords, meshes_world
            )
            fragments = Fragments(
                bary_coords=clipped_bary_coords,
                zbuf=clipped_zbuf,
                dists=fragments.dists,
                pix_to_face=fragments.pix_to_face,
            )
        images = self.shader(fragments, meshes_world, **kwargs)

        return images
