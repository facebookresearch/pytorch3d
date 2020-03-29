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
from .shading import flat_shading, gouraud_shading, phong_shading
from .texturing import interpolate_texture_map, interpolate_vertex_colors


# A Shader should take as input fragments from the output of rasterization
# along with scene params and output images. A shader could perform operations
# such as:
#     - interpolate vertex attributes for all the fragments
#     - sample colors from a texture map
#     - apply per pixel lighting
#     - blend colors across top K faces per pixel.


class HardPhongShader(nn.Module):
    """
    Per pixel lighting - the lighting model is applied using the interpolated
    coordinates and normals for each pixel. The blending function hard assigns
    the color of the closest face for each pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = HardPhongShader(device=torch.device("cuda:0"))
    """

    def __init__(self, device="cpu", cameras=None, lights=None, materials=None):
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = (
            cameras if cameras is not None else OpenGLPerspectiveCameras(device=device)
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


class SoftPhongShader(nn.Module):
    """
    Per pixel lighting - the lighting model is applied using the interpolated
    coordinates and normals for each pixel. The blending function returns the
    soft aggregated color using all the faces per pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = SoftPhongShader(device=torch.device("cuda:0"))
    """

    def __init__(
        self, device="cpu", cameras=None, lights=None, materials=None, blend_params=None
    ):
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = (
            cameras if cameras is not None else OpenGLPerspectiveCameras(device=device)
        )
        self.blend_params = blend_params if blend_params is not None else BlendParams()

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
        images = softmax_rgb_blend(colors, fragments, self.blend_params)
        return images


class HardGouraudShader(nn.Module):
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

    def __init__(self, device="cpu", cameras=None, lights=None, materials=None):
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = (
            cameras if cameras is not None else OpenGLPerspectiveCameras(device=device)
        )

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        pixel_colors = gouraud_shading(
            meshes=meshes,
            fragments=fragments,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        images = hard_rgb_blend(pixel_colors, fragments)
        return images


class SoftGouraudShader(nn.Module):
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

    def __init__(
        self, device="cpu", cameras=None, lights=None, materials=None, blend_params=None
    ):
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = (
            cameras if cameras is not None else OpenGLPerspectiveCameras(device=device)
        )
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        pixel_colors = gouraud_shading(
            meshes=meshes,
            fragments=fragments,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        images = softmax_rgb_blend(pixel_colors, fragments, self.blend_params)
        return images


class TexturedSoftPhongShader(nn.Module):
    """
    Per pixel lighting applied to a texture map. First interpolate the vertex
    uv coordinates and sample from a texture map. Then apply the lighting model
    using the interpolated coords and normals for each pixel.

    The blending function returns the soft aggregated color using all
    the faces per pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = TexturedPhongShader(device=torch.device("cuda:0"))
    """

    def __init__(
        self, device="cpu", cameras=None, lights=None, materials=None, blend_params=None
    ):
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = (
            cameras if cameras is not None else OpenGLPerspectiveCameras(device=device)
        )
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        texels = interpolate_texture_map(fragments, meshes)
        cameras = kwargs.get("cameras", self.cameras)
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
        images = softmax_rgb_blend(colors, fragments, blend_params)
        return images


class HardFlatShader(nn.Module):
    """
    Per face lighting - the lighting model is applied using the average face
    position and the face normal. The blending function hard assigns
    the color of the closest face for each pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = HardFlatShader(device=torch.device("cuda:0"))
    """

    def __init__(self, device="cpu", cameras=None, lights=None, materials=None):
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = (
            cameras if cameras is not None else OpenGLPerspectiveCameras(device=device)
        )

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        texels = interpolate_vertex_colors(fragments, meshes)
        cameras = kwargs.get("cameras", self.cameras)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        colors = flat_shading(
            meshes=meshes,
            fragments=fragments,
            texels=texels,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        images = hard_rgb_blend(colors, fragments)
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

    def __init__(self, blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        """"
        Only want to render the silhouette so RGB values can be ones.
        There is no need for lighting or texturing
        """
        colors = torch.ones_like(fragments.bary_coords)
        blend_params = kwargs.get("blend_params", self.blend_params)
        images = sigmoid_alpha_blend(colors, fragments, blend_params)
        return images
