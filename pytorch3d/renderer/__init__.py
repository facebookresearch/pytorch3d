# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .blending import (
    BlendParams,
    hard_rgb_blend,
    sigmoid_alpha_blend,
    softmax_rgb_blend,
)
from .camera_utils import join_cameras_as_batch, rotate_on_spot
from .cameras import (  # deprecated  # deprecated  # deprecated  # deprecated
    camera_position_from_spherical_angles,
    CamerasBase,
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    get_world_to_view_transform,
    look_at_rotation,
    look_at_view_transform,
    OpenGLOrthographicCameras,
    OpenGLPerspectiveCameras,
    OrthographicCameras,
    PerspectiveCameras,
    SfMOrthographicCameras,
    SfMPerspectiveCameras,
)
from .implicit import (
    AbsorptionOnlyRaymarcher,
    EmissionAbsorptionRaymarcher,
    GridRaysampler,
    HarmonicEmbedding,
    HeterogeneousRayBundle,
    ImplicitRenderer,
    MonteCarloRaysampler,
    MultinomialRaysampler,
    NDCGridRaysampler,
    NDCMultinomialRaysampler,
    ray_bundle_to_ray_points,
    ray_bundle_variables_to_ray_points,
    RayBundle,
    VolumeRenderer,
    VolumeSampler,
)
from .lighting import AmbientLights, diffuse, DirectionalLights, PointLights, specular
from .materials import Materials
from .mesh import (
    gouraud_shading,
    HardFlatShader,
    HardGouraudShader,
    HardPhongShader,
    MeshRasterizer,
    MeshRenderer,
    MeshRendererWithFragments,
    phong_shading,
    RasterizationSettings,
    rasterize_meshes,
    SoftGouraudShader,
    SoftPhongShader,
    SoftSilhouetteShader,
    SplatterPhongShader,
    Textures,
    TexturesAtlas,
    TexturesUV,
    TexturesVertex,
)

from .points import (
    AlphaCompositor,
    NormWeightedCompositor,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    PulsarPointsRenderer,
    rasterize_points,
)
from .splatter_blend import SplatterBlender
from .utils import (
    convert_to_tensors_and_broadcast,
    ndc_grid_sample,
    ndc_to_grid_sample_coords,
    TensorProperties,
)


__all__ = [k for k in globals().keys() if not k.startswith("_")]
