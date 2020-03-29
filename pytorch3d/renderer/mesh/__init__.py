# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


from .texturing import interpolate_texture_map, interpolate_vertex_colors  # isort:skip
from .rasterize_meshes import rasterize_meshes
from .rasterizer import MeshRasterizer, RasterizationSettings
from .renderer import MeshRenderer
from .shader import (
    HardFlatShader,
    HardGouraudShader,
    HardPhongShader,
    SoftGouraudShader,
    SoftPhongShader,
    SoftSilhouetteShader,
    TexturedSoftPhongShader,
)
from .shading import gouraud_shading, phong_shading
from .utils import interpolate_face_attributes


__all__ = [k for k in globals().keys() if not k.startswith("_")]
