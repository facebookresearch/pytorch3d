# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .clip import (
    clip_faces,
    ClipFrustum,
    ClippedFaces,
    convert_clipped_rasterization_to_original_faces,
)
from .rasterize_meshes import rasterize_meshes
from .rasterizer import MeshRasterizer, RasterizationSettings
from .renderer import MeshRenderer, MeshRendererWithFragments
from .shader import (  # DEPRECATED
    BlendParams,
    HardFlatShader,
    HardGouraudShader,
    HardPhongShader,
    SoftGouraudShader,
    SoftPhongShader,
    SoftSilhouetteShader,
    SplatterPhongShader,
    TexturedSoftPhongShader,
)
from .shading import gouraud_shading, phong_shading
from .textures import (  # DEPRECATED
    Textures,
    TexturesAtlas,
    TexturesBase,
    TexturesUV,
    TexturesVertex,
)


__all__ = [k for k in globals().keys() if not k.startswith("_")]
