# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from pytorch3d import _C
from torch.autograd import Function
from torch.autograd.function import once_differentiable


def interpolate_face_attributes(
    pix_to_face: torch.Tensor,
    barycentric_coords: torch.Tensor,
    face_attributes: torch.Tensor,
) -> torch.Tensor:
    """
    Interpolate arbitrary face attributes using the barycentric coordinates
    for each pixel in the rasterized output.

    Args:
        pix_to_face: LongTensor of shape (...) specifying the indices
            of the faces (in the packed representation) which overlap each
            pixel in the image. A value < 0 indicates that the pixel does not
            overlap any face and should be skipped.
        barycentric_coords: FloatTensor of shape (N, H, W, K, 3) specifying
            the barycentric coordinates of each pixel
            relative to the faces (in the packed
            representation) which overlap the pixel.
        face_attributes: packed attributes of shape (total_faces, 3, D),
            specifying the value of the attribute for each
            vertex in the face.

    Returns:
        pixel_vals: tensor of shape (N, H, W, K, D) giving the interpolated
        value of the face attribute for each pixel.
    """
    # Check shapes
    F, FV, D = face_attributes.shape
    if FV != 3:
        raise ValueError("Faces can only have three vertices; got %r" % FV)
    N, H, W, K, _ = barycentric_coords.shape
    if pix_to_face.shape != (N, H, W, K):
        msg = "pix_to_face must have shape (batch_size, H, W, K); got %r"
        raise ValueError(msg % (pix_to_face.shape,))

    # On CPU use the python version
    # TODO: Implement a C++ version of this function
    if not pix_to_face.is_cuda:
        args = (pix_to_face, barycentric_coords, face_attributes)
        return interpolate_face_attributes_python(*args)

    # Otherwise flatten and call the custom autograd function
    N, H, W, K = pix_to_face.shape
    pix_to_face = pix_to_face.view(-1)
    barycentric_coords = barycentric_coords.view(N * H * W * K, 3)
    args = (pix_to_face, barycentric_coords, face_attributes)
    out = _InterpFaceAttrs.apply(*args)
    out = out.view(N, H, W, K, -1)
    return out


class _InterpFaceAttrs(Function):
    @staticmethod
    def forward(ctx, pix_to_face, barycentric_coords, face_attrs):
        args = (pix_to_face, barycentric_coords, face_attrs)
        ctx.save_for_backward(*args)
        return _C.interp_face_attrs_forward(*args)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_pix_attrs):
        args = ctx.saved_tensors
        args = args + (grad_pix_attrs,)
        grads = _C.interp_face_attrs_backward(*args)
        grad_pix_to_face = None
        grad_barycentric_coords = grads[0]
        grad_face_attrs = grads[1]
        return grad_pix_to_face, grad_barycentric_coords, grad_face_attrs


def interpolate_face_attributes_python(
    pix_to_face: torch.Tensor,
    barycentric_coords: torch.Tensor,
    face_attributes: torch.Tensor,
) -> torch.Tensor:
    F, FV, D = face_attributes.shape
    N, H, W, K, _ = barycentric_coords.shape

    # Replace empty pixels in pix_to_face with 0 in order to interpolate.
    mask = pix_to_face < 0
    pix_to_face = pix_to_face.clone()
    pix_to_face[mask] = 0
    idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
    pixel_face_vals = face_attributes.gather(0, idx).view(N, H, W, K, 3, D)
    pixel_vals = (barycentric_coords[..., None] * pixel_face_vals).sum(dim=-2)
    pixel_vals[mask] = 0  # Replace masked values in output.
    return pixel_vals
