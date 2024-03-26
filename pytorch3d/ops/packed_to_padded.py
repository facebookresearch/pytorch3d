# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
from pytorch3d import _C
from torch.autograd import Function
from torch.autograd.function import once_differentiable


class _PackedToPadded(Function):
    """
    Torch autograd Function wrapper for packed_to_padded C++/CUDA implementations.
    """

    @staticmethod
    def forward(ctx, inputs, first_idxs, max_size):
        """
        Args:
            ctx: Context object used to calculate gradients.
            inputs: FloatTensor of shape (F, D), representing the packed batch tensor.
                e.g. areas for faces in a batch of meshes.
            first_idxs: LongTensor of shape (N,) where N is the number of
                elements in the batch and `first_idxs[i] = f`
                means that the inputs for batch element i begin at `inputs[f]`.
            max_size: Max length of an element in the batch.

        Returns:
            inputs_padded: FloatTensor of shape (N, max_size, D) where max_size is max
                of `sizes`. The values for batch element i which start at
                `inputs[first_idxs[i]]` will be copied to `inputs_padded[i, :]`,
                with zeros padding out the extra inputs.
        """
        if not (inputs.dim() == 2):
            raise ValueError("input can only be 2-dimensional.")
        if not (first_idxs.dim() == 1):
            raise ValueError("first_idxs can only be 1-dimensional.")
        if not (inputs.dtype == torch.float32):
            raise ValueError("input has to be of type torch.float32.")
        if not (first_idxs.dtype == torch.int64):
            raise ValueError("first_idxs has to be of type torch.int64.")
        if not isinstance(max_size, int):
            raise ValueError("max_size has to be int.")

        ctx.save_for_backward(first_idxs)
        ctx.num_inputs = int(inputs.shape[0])
        inputs, first_idxs = inputs.contiguous(), first_idxs.contiguous()
        inputs_padded = _C.packed_to_padded(inputs, first_idxs, max_size)
        return inputs_padded

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        first_idxs = ctx.saved_tensors[0]
        num_inputs = ctx.num_inputs
        grad_input = _C.padded_to_packed(grad_output, first_idxs, num_inputs)
        return grad_input, None, None


def packed_to_padded(
    inputs: torch.Tensor, first_idxs: torch.LongTensor, max_size: int
) -> torch.Tensor:
    """
    Torch wrapper that handles allowed input shapes. See description below.

    Args:
        inputs: FloatTensor of shape (F,) or (F, ...), representing the packed
            batch tensor, e.g. areas for faces in a batch of meshes.
        first_idxs: LongTensor of shape (N,) where N is the number of
            elements in the batch and `first_idxs[i] = f`
            means that the inputs for batch element i begin at `inputs[f]`.
        max_size: Max length of an element in the batch.

    Returns:
        inputs_padded: FloatTensor of shape (N, max_size) or (N, max_size, ...)
            where max_size is max of `sizes`. The values for batch element i
            which start at `inputs[first_idxs[i]]` will be copied to
            `inputs_padded[i, :]`, with zeros padding out the extra inputs.

    To handle the allowed input shapes, we convert the inputs tensor of shape
    (F,) to (F, 1). We reshape the output back to (N, max_size) from
    (N, max_size, 1).
    """
    # if inputs is of shape (F,), reshape into (F, 1)
    input_shape = inputs.shape
    n_dims = inputs.dim()
    if n_dims == 1:
        inputs = inputs.unsqueeze(1)
    else:
        inputs = inputs.reshape(input_shape[0], -1)
    inputs_padded = _PackedToPadded.apply(inputs, first_idxs, max_size)
    # if flat is True, reshape output to (N, max_size) from (N, max_size, 1)
    # else reshape output to (N, max_size, ...)
    if n_dims == 1:
        return inputs_padded.squeeze(2)
    if n_dims == 2:
        return inputs_padded
    return inputs_padded.view(*inputs_padded.shape[:2], *input_shape[1:])


class _PaddedToPacked(Function):
    """
    Torch autograd Function wrapper for padded_to_packed C++/CUDA implementations.
    """

    @staticmethod
    def forward(ctx, inputs, first_idxs, num_inputs):
        """
        Args:
            ctx: Context object used to calculate gradients.
            inputs: FloatTensor of shape (N, max_size, D), representing
            the padded tensor, e.g. areas for faces in a batch of meshes.
            first_idxs: LongTensor of shape (N,) where N is the number of
                elements in the batch and `first_idxs[i] = f`
                means that the inputs for batch element i begin at `inputs_packed[f]`.
            num_inputs: Number of packed entries (= F)

        Returns:
            inputs_packed: FloatTensor of shape (F, D) where
                `inputs_packed[first_idx[i]:] = inputs[i, :]`.
        """
        if not (inputs.dim() == 3):
            raise ValueError("input can only be 3-dimensional.")
        if not (first_idxs.dim() == 1):
            raise ValueError("first_idxs can only be 1-dimensional.")
        if not (inputs.dtype == torch.float32):
            raise ValueError("input has to be of type torch.float32.")
        if not (first_idxs.dtype == torch.int64):
            raise ValueError("first_idxs has to be of type torch.int64.")
        if not isinstance(num_inputs, int):
            raise ValueError("max_size has to be int.")

        ctx.save_for_backward(first_idxs)
        ctx.max_size = inputs.shape[1]
        inputs, first_idxs = inputs.contiguous(), first_idxs.contiguous()
        inputs_packed = _C.padded_to_packed(inputs, first_idxs, num_inputs)
        return inputs_packed

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        first_idxs = ctx.saved_tensors[0]
        max_size = ctx.max_size
        grad_input = _C.packed_to_padded(grad_output, first_idxs, max_size)
        return grad_input, None, None


def padded_to_packed(
    inputs: torch.Tensor,
    first_idxs: torch.LongTensor,
    num_inputs: int,
    max_size_dim: int = 1,
) -> torch.Tensor:
    """
    Torch wrapper that handles allowed input shapes. See description below.

    Args:
        inputs: FloatTensor of shape (N, ..., max_size) or (N, ..., max_size, ...),
            representing the padded tensor, e.g. areas for faces in a batch of
            meshes, where max_size occurs on max_size_dim-th position.
        first_idxs: LongTensor of shape (N,) where N is the number of
            elements in the batch and `first_idxs[i] = f`
            means that the inputs for batch element i begin at `inputs_packed[f]`.
        num_inputs: Number of packed entries (= F)
        max_size_dim: the dimension to be packed

    Returns:
        inputs_packed: FloatTensor of shape (F,) or (F, ...) where
            `inputs_packed[first_idx[i]:first_idx[i+1]] = inputs[i, ..., :delta[i]]`,
            where `delta[i] = first_idx[i+1] - first_idx[i]`.

    To handle the allowed input shapes, we convert the inputs tensor of shape
    (N, max_size) to (N, max_size, 1). We reshape the output back to (F,) from
    (F, 1).
    """
    n_dims = inputs.dim()
    # move the variable dim to position 1
    inputs = inputs.movedim(max_size_dim, 1)

    # if inputs is of shape (N, max_size), reshape into (N, max_size, 1))
    input_shape = inputs.shape
    if n_dims == 2:
        inputs = inputs.unsqueeze(2)
    else:
        inputs = inputs.reshape(*input_shape[:2], -1)
    inputs_packed = _PaddedToPacked.apply(inputs, first_idxs, num_inputs)
    # if input is flat, reshape output to (F,) from (F, 1)
    # else reshape output to (F, ...)
    if n_dims == 2:
        return inputs_packed.squeeze(1)

    return inputs_packed.view(-1, *input_shape[2:])
