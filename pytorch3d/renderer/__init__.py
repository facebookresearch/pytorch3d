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
from .cameras import OpenGLOrthographicCameras  # deprecated
from .cameras import OpenGLPerspectiveCameras  # deprecated
from .cameras import SfMOrthographicCameras  # deprecated
from .cameras import SfMPerspectiveCameras  # deprecated
from .cameras import (
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    OrthographicCameras,
    PerspectiveCameras,
    camera_position_from_spherical_angles,
    get_world_to_view_transform,
    look_at_rotation,
    look_at_view_transform,
)
from .implicit import (
    AbsorptionOnlyRaymarcher,
    EmissionAbsorptionRaymarcher,
    GridRaysampler,
    HarmonicEmbedding,
    ImplicitRenderer,
    MonteCarloRaysampler,
    MultinomialRaysampler,
    NDCGridRaysampler,
    NDCMultinomialRaysampler,
    RayBundle,
    VolumeRenderer,
    VolumeSampler,
    ray_bundle_to_ray_points,
    ray_bundle_variables_to_ray_points,
)
from .lighting import AmbientLights, DirectionalLights, PointLights, diffuse, specular
from .materials import Materials
from .mesh import (
    HardFlatShader,
    HardGouraudShader,
    HardPhongShader,
    MeshRasterizer,
    MeshRenderer,
    MeshRendererWithFragments,
    RasterizationSettings,
    SoftGouraudShader,
    SoftPhongShader,
    SoftSilhouetteShader,
    Textures,
    TexturesAtlas,
    TexturesUV,
    TexturesVertex,
    gouraud_shading,
    phong_shading,
    rasterize_meshes,
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
from .utils import (
    TensorProperties,
    convert_to_tensors_and_broadcast,
    ndc_grid_sample,
    ndc_to_grid_sample_coords,
)


__all__ = [k for k in globals().keys() if not k.startswith("_")]
