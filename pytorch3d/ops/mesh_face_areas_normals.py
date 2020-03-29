# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
from pytorch3d import _C
from torch.autograd import Function
from torch.autograd.function import once_differentiable


class _MeshFaceAreasNormals(Function):
    """
    Torch autograd Function wrapper for face areas & normals C++/CUDA implementations.
    """

    @staticmethod
    def forward(ctx, verts, faces):
        """
        Args:
            ctx: Context object used to calculate gradients.
            verts: FloatTensor of shape (V, 3), representing the packed
                batch verts tensor.
            faces: LongTensor of shape (F, 3), representing the packed
                batch faces tensor
        Returns:
            areas: FloatTensor of shape (F,) with the areas of each face
            normals: FloatTensor of shape (F,3) with the normals of each face
        """
        if not (verts.dim() == 2):
            raise ValueError("verts need to be of shape Vx3.")
        if not (verts.shape[1] == 3):
            raise ValueError("verts need to be of shape Vx3.")
        if not (faces.dim() == 2):
            raise ValueError("faces need to be of shape Fx3.")
        if not (faces.shape[1] == 3):
            raise ValueError("faces need to be of shape Fx3.")
        if not (faces.dtype == torch.int64):
            raise ValueError("faces need to be of type torch.int64.")
        # TODO(gkioxari) Change cast to floats once we add support for doubles.
        if not (verts.dtype == torch.float32):
            verts = verts.float()

        ctx.save_for_backward(verts, faces)
        areas, normals = _C.face_areas_normals_forward(verts, faces)
        return areas, normals

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_areas, grad_normals):
        grad_areas = grad_areas.contiguous()
        grad_normals = grad_normals.contiguous()
        verts, faces = ctx.saved_tensors
        # TODO(gkioxari) Change cast to floats once we add support for doubles.
        if not (grad_areas.dtype == torch.float32):
            grad_areas = grad_areas.float()
        if not (grad_normals.dtype == torch.float32):
            grad_normals = grad_normals.float()
        grad_verts = _C.face_areas_normals_backward(
            grad_areas, grad_normals, verts, faces
        )
        return grad_verts, None


mesh_face_areas_normals = _MeshFaceAreasNormals.apply
