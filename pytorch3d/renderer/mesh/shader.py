#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import torch
import torch.nn as nn

from ..blending import (
    BlendParams,
    hard_rgb_blend,
    sigmoid_alpha_blend,
    softmax_rgb_blend,
)
from ..cameras import OpenGLPerspectiveCameras
from ..lighting import PointLights
from ..materials import Materials
from .shading import gourad_shading, phong_shading
from .texturing import interpolate_texture_map, interpolate_vertex_colors

# A Shader should take as input fragments from the output of rasterization
# along with scene params and output images. A shader could perform operations
# such as:
#     - interpolate vertex attributes for all the fragments
#     - sample colors from a texture map
#     - apply per pixel lighting
#     - blend colors across top K faces per pixel.


class PhongShader(nn.Module):
    """
    Per pixel lighting. Apply the lighting model using the interpolated coords
    and normals for each pixel.
    """

    def __init__(self, device="cpu", cameras=None, lights=None, materials=None):
        super().__init__()
        self.lights = (
            lights if lights is not None else PointLights(device=device)
        )
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = (
            cameras
            if cameras is not None
            else OpenGLPerspectiveCameras(device=device)
        )

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        texels = interpolate_vertex_colors(fragments, meshes)
        cameras = kwargs.get("cameras", self.cameras)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        colors = phong_shading(
            meshes=meshes,
            fragments=fragments,
            texels=texels,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        images = hard_rgb_blend(colors, fragments)
        return images


class GouradShader(nn.Module):
    """
    Per vertex lighting. Apply the lighting model to the vertex colors and then
    interpolate using the barycentric coordinates to get colors for each pixel.
    """

    def __init__(self, device="cpu", cameras=None, lights=None, materials=None):
        super().__init__()
        self.lights = (
            lights if lights is not None else PointLights(device=device)
        )
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = (
            cameras
            if cameras is not None
            else OpenGLPerspectiveCameras(device=device)
        )

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        pixel_colors = gourad_shading(
            meshes=meshes,
            fragments=fragments,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        images = hard_rgb_blend(pixel_colors, fragments)
        return images


class TexturedPhongShader(nn.Module):
    def __init__(
        self,
        device="cpu",
        cameras=None,
        lights=None,
        materials=None,
        blend_params=None,
    ):
        super().__init__()
        self.lights = (
            lights if lights is not None else PointLights(device=device)
        )
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = (
            cameras
            if cameras is not None
            else OpenGLPerspectiveCameras(device=device)
        )
        self.blend_params = (
            blend_params if blend_params is not None else BlendParams()
        )

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        texels = interpolate_texture_map(fragments, meshes)
        cameras = kwargs.get("cameras", self.cameras)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        colors = phong_shading(
            meshes=meshes,
            fragments=fragments,
            texels=texels,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        images = softmax_rgb_blend(colors, fragments, self.blend_params)
        return images


class SilhouetteShader(nn.Module):
    def __init__(
        self,
        device="cpu",
        cameras=None,
        lights=None,
        materials=None,
        blend_params=None,
    ):
        super().__init__()
        self.lights = (
            lights if lights is not None else PointLights(device=device)
        )
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = (
            cameras
            if cameras is not None
            else OpenGLPerspectiveCameras(device=device)
        )
        self.blend_params = (
            blend_params if blend_params is None else BlendParams()
        )
        # In the rasterizer set:
        #   blur_radius = np.log(1 / blur - 1.) * blend_params.sigma

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        """"
        Only want to render the silhouette so RGB values can be ones.
        There is no need for lighting or texturing
        """
        colors = torch.ones_like(fragments.bary_coords)
        blend_params = kwargs.get("blend_params", self.blend_params)
        images = sigmoid_alpha_blend(colors, fragments, blend_params)
        return images
