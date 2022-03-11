# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from .clip import (
    ClipFrustum,
    ClippedFaces,
    clip_faces,
    convert_clipped_rasterization_to_original_faces,
)
from .rasterize_meshes import rasterize_meshes
from .rasterizer import MeshRasterizer, RasterizationSettings
from .renderer import MeshRenderer, MeshRendererWithFragments
from .shader import TexturedSoftPhongShader  # DEPRECATED
from .shader import (
    BlendParams,
    HardFlatShader,
    HardGouraudShader,
    HardPhongShader,
    SoftGouraudShader,
    SoftPhongShader,
    SoftSilhouetteShader,
)
from .shading import gouraud_shading, phong_shading
from .textures import Textures  # DEPRECATED
from .textures import TexturesAtlas, TexturesBase, TexturesUV, TexturesVertex


__all__ = [k for k in globals().keys() if not k.startswith("_")]
