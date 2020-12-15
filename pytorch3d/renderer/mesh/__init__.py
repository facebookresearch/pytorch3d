# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


from .rasterize_meshes import rasterize_meshes
from .rasterizer import MeshRasterizer, RasterizationSettings
from .renderer import MeshRenderer
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
from .textures import TexturesAtlas, TexturesUV, TexturesVertex


__all__ = [k for k in globals().keys() if not k.startswith("_")]
