# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

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
from ..splatter_blend import SplatterBlender
from ..utils import TensorProperties
from .rasterizer import Fragments
from .shading import (
    _phong_shading_with_pixels,
    flat_shading,
    gouraud_shading,
    phong_shading,
)


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

    def _get_cameras(self, **kwargs):
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of the shader."
            raise ValueError(msg)

        return cameras

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
        cameras = super()._get_cameras(**kwargs)
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
        cameras = super()._get_cameras(**kwargs)
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
        cameras = super()._get_cameras(**kwargs)
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
        cameras = super()._get_cameras(**kwargs)
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
        cameras = super()._get_cameras(**kwargs)
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


class SplatterPhongShader(ShaderBase):
    """
    Per pixel lighting - the lighting model is applied using the interpolated
    coordinates and normals for each pixel. The blending function returns the
    color aggregated using splats from surrounding pixels (see [0]).

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = SplatterPhongShader(device=torch.device("cuda:0"))

    [0] Cole, F. et al., "Differentiable Surface Rendering via Non-differentiable
        Sampling".
    """

    def __init__(self, **kwargs):
        self.splatter_blender = None
        super().__init__(**kwargs)

    def to(self, device: Device):
        if self.splatter_blender:
            self.splatter_blender.to(device)
        return super().to(device)

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        cameras = super()._get_cameras(**kwargs)
        texels = meshes.sample_textures(fragments)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)

        colors, pixel_coords_cameras = _phong_shading_with_pixels(
            meshes=meshes,
            fragments=fragments.detach(),
            texels=texels,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )

        if not self.splatter_blender:
            # Init only once, to avoid re-computing constants.
            N, H, W, K, _ = colors.shape
            self.splatter_blender = SplatterBlender((N, H, W, K), colors.device)

        blend_params = kwargs.get("blend_params", self.blend_params)
        self.check_blend_params(blend_params)

        images = self.splatter_blender(
            colors,
            pixel_coords_cameras,
            cameras,
            fragments.pix_to_face < 0,
            kwargs.get("blend_params", self.blend_params),
        )

        return images

    def check_blend_params(self, blend_params):
        if blend_params.sigma != 0.5:
            warnings.warn(
                f"SplatterPhongShader received sigma={blend_params.sigma}. sigma is "
                "defined in pixel units, and any value other than 0.5 is highly "
                "unexpected. Only use other values if you know what you are doing. "
            )


class HardDepthShader(ShaderBase):
    """
    Renders the Z distances of the closest face for each pixel. If no face is
    found it returns the zfar value of the camera.

    Output from this shader is [N, H, W, 1] since it's only depth.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = HardDepthShader(device=torch.device("cuda:0"))
    """

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        cameras = super()._get_cameras(**kwargs)

        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        mask = fragments.pix_to_face[..., 0:1] < 0

        zbuf = fragments.zbuf[..., 0:1].clone()
        zbuf[mask] = zfar
        return zbuf


class SoftDepthShader(ShaderBase):
    """
    Renders the Z distances using an aggregate of the distances of each face
    based off of the point distance.  If no face is found it returns the zfar
    value of the camera.

    Output from this shader is [N, H, W, 1] since it's only depth.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = SoftDepthShader(device=torch.device("cuda:0"))
    """

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        if fragments.dists is None:
            raise ValueError("SoftDepthShader requires Fragments.dists to be present.")

        cameras = super()._get_cameras(**kwargs)

        N, H, W, K = fragments.pix_to_face.shape
        device = fragments.zbuf.device
        mask = fragments.pix_to_face >= 0

        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))

        # Sigmoid probability map based on the distance of the pixel to the face.
        prob_map = torch.sigmoid(-fragments.dists / self.blend_params.sigma) * mask

        # append extra face for zfar
        dists = torch.cat(
            (fragments.zbuf, torch.ones((N, H, W, 1), device=device) * zfar), dim=3
        )
        probs = torch.cat((prob_map, torch.ones((N, H, W, 1), device=device)), dim=3)

        # compute weighting based off of probabilities using cumsum
        probs = probs.cumsum(dim=3)
        probs = probs.clamp(max=1)
        probs = probs.diff(dim=3, prepend=torch.zeros((N, H, W, 1), device=device))

        return (probs * dists).sum(dim=3).unsqueeze(3)
