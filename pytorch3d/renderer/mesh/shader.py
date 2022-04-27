# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Optional

import torch
import torch.nn as nn

from ...common.datatypes import Device
from ...structures.meshes import Meshes
from ..blending import (
    BlendParams,
    hard_rgb_blend,
    sigmoid_alpha_blend,
    softmax_rgb_blend,
)
from ..lighting import PointLights
from ..materials import Materials
from ..utils import TensorProperties
from .rasterizer import Fragments
from .shading import flat_shading, gouraud_shading, phong_shading


# A Shader should take as input fragments from the output of rasterization
# along with scene params and output images. A shader could perform operations
# such as:
#     - interpolate vertex attributes for all the fragments
#     - sample colors from a texture map
#     - apply per pixel lighting
#     - blend colors across top K faces per pixel.
class ShaderBase(nn.Module):
    def __init__(
        self,
        device: Device = "cpu",
        cameras: Optional[TensorProperties] = None,
        lights: Optional[TensorProperties] = None,
        materials: Optional[Materials] = None,
        blend_params: Optional[BlendParams] = None,
    ) -> None:
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    # pyre-fixme[14]: `to` overrides method defined in `Module` inconsistently.
    def to(self, device: Device):
        # Manually move to device modules which are not subclasses of nn.Module
        cameras = self.cameras
        if cameras is not None:
            self.cameras = cameras.to(device)
        self.materials = self.materials.to(device)
        self.lights = self.lights.to(device)
        return self


class HardPhongShader(ShaderBase):
    """
    Per pixel lighting - the lighting model is applied using the interpolated
    coordinates and normals for each pixel. The blending function hard assigns
    the color of the closest face for each pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = HardPhongShader(device=torch.device("cuda:0"))
    """

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of HardPhongShader"
            raise ValueError(msg)

        texels = meshes.sample_textures(fragments)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        blend_params = kwargs.get("blend_params", self.blend_params)
        colors = phong_shading(
            meshes=meshes,
            fragments=fragments,
            texels=texels,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        images = hard_rgb_blend(colors, fragments, blend_params)
        return images


class SoftPhongShader(ShaderBase):
    """
    Per pixel lighting - the lighting model is applied using the interpolated
    coordinates and normals for each pixel. The blending function returns the
    soft aggregated color using all the faces per pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = SoftPhongShader(device=torch.device("cuda:0"))
    """

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of SoftPhongShader"
            raise ValueError(msg)

        texels = meshes.sample_textures(fragments)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        blend_params = kwargs.get("blend_params", self.blend_params)
        colors = phong_shading(
            meshes=meshes,
            fragments=fragments,
            texels=texels,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        images = softmax_rgb_blend(
            colors, fragments, blend_params, znear=znear, zfar=zfar
        )
        return images


class HardGouraudShader(ShaderBase):
    """
    Per vertex lighting - the lighting model is applied to the vertex colors and
    the colors are then interpolated using the barycentric coordinates to
    obtain the colors for each pixel. The blending function hard assigns
    the color of the closest face for each pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = HardGouraudShader(device=torch.device("cuda:0"))
    """

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of HardGouraudShader"
            raise ValueError(msg)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        blend_params = kwargs.get("blend_params", self.blend_params)

        # As Gouraud shading applies the illumination to the vertex
        # colors, the interpolated pixel texture is calculated in the
        # shading step. In comparison, for Phong shading, the pixel
        # textures are computed first after which the illumination is
        # applied.
        pixel_colors = gouraud_shading(
            meshes=meshes,
            fragments=fragments,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        images = hard_rgb_blend(pixel_colors, fragments, blend_params)
        return images


class SoftGouraudShader(ShaderBase):
    """
    Per vertex lighting - the lighting model is applied to the vertex colors and
    the colors are then interpolated using the barycentric coordinates to
    obtain the colors for each pixel. The blending function returns the
    soft aggregated color using all the faces per pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = SoftGouraudShader(device=torch.device("cuda:0"))
    """

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of SoftGouraudShader"
            raise ValueError(msg)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        pixel_colors = gouraud_shading(
            meshes=meshes,
            fragments=fragments,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        images = softmax_rgb_blend(
            pixel_colors, fragments, self.blend_params, znear=znear, zfar=zfar
        )
        return images


def TexturedSoftPhongShader(
    device: Device = "cpu",
    cameras: Optional[TensorProperties] = None,
    lights: Optional[TensorProperties] = None,
    materials: Optional[Materials] = None,
    blend_params: Optional[BlendParams] = None,
) -> SoftPhongShader:
    """
    TexturedSoftPhongShader class has been DEPRECATED. Use SoftPhongShader instead.
    Preserving TexturedSoftPhongShader as a function for backwards compatibility.
    """
    warnings.warn(
        """TexturedSoftPhongShader is now deprecated;
            use SoftPhongShader instead.""",
        PendingDeprecationWarning,
    )
    return SoftPhongShader(
        device=device,
        cameras=cameras,
        lights=lights,
        materials=materials,
        blend_params=blend_params,
    )


class HardFlatShader(ShaderBase):
    """
    Per face lighting - the lighting model is applied using the average face
    position and the face normal. The blending function hard assigns
    the color of the closest face for each pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = HardFlatShader(device=torch.device("cuda:0"))
    """

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of HardFlatShader"
            raise ValueError(msg)
        texels = meshes.sample_textures(fragments)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        blend_params = kwargs.get("blend_params", self.blend_params)
        colors = flat_shading(
            meshes=meshes,
            fragments=fragments,
            texels=texels,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        images = hard_rgb_blend(colors, fragments, blend_params)
        return images


class SoftSilhouetteShader(nn.Module):
    """
    Calculate the silhouette by blending the top K faces for each pixel based
    on the 2d euclidean distance of the center of the pixel to the mesh face.

    Use this shader for generating silhouettes similar to SoftRasterizer [0].

    .. note::

        To be consistent with SoftRasterizer, initialize the
        RasterizationSettings for the rasterizer with
        `blur_radius = np.log(1. / 1e-4 - 1.) * blend_params.sigma`

    [0] Liu et al, 'Soft Rasterizer: A Differentiable Renderer for Image-based
        3D Reasoning', ICCV 2019
    """

    def __init__(self, blend_params: Optional[BlendParams] = None) -> None:
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        """
        Only want to render the silhouette so RGB values can be ones.
        There is no need for lighting or texturing
        """
        colors = torch.ones_like(fragments.bary_coords)
        blend_params = kwargs.get("blend_params", self.blend_params)
        images = sigmoid_alpha_blend(colors, fragments, blend_params)
        return images
