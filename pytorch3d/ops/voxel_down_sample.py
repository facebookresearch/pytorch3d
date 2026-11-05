# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from numbers import Real

import torch

from pytorch3d.structures import Pointclouds


def voxel_down_sample(pointclouds: Pointclouds, voxel_size: float) -> Pointclouds:
    """
    Downsample each point cloud by averaging points within occupied voxels.

    Voxels are axis-aligned cubes with side length ``voxel_size`` and a fixed
    origin at (0, 0, 0). A point at ``p`` belongs to voxel ``floor(p / voxel_size)``;
    each coordinate interval is half-open, ``[k * voxel_size, (k + 1) * voxel_size)``.
    In particular, negative coordinates use floor, not truncation toward zero.
    Adding points does not move the grid. This differs from implementations which
    choose the grid origin from the point cloud's bounding box.

    Each occupied voxel produces the arithmetic mean of its points. Features and
    normals, when present, are averaged using the same groups. Averaged normals
    are not normalized. Clouds are processed independently, and output voxels
    within each cloud are ordered lexicographically by their integer coordinates.

    Voxel assignment is not differentiable. Within a fixed assignment, gradients
    propagate through the averages to points, features, and normals. No gradient
    is defined for voxel_size or across voxel boundaries. Floating-point reduction
    order can differ across devices, so CPU and CUDA results need not be bitwise
    identical. PyTorch's deterministic-algorithm settings are respected.

    The implementation uses sparse grouping with storage proportional to the
    number of input points and their attributes, rather than allocating a dense
    volume covering the bounding box. Geometry and attributes remain on the input
    device; Pointclouds may read batch-length metadata when converting to lists.

    Args:
        pointclouds: A batch of Pointclouds, including heterogeneous and empty
            clouds. Points and any features or normals must be finite float32 or
            float64 tensors. Each tensor's dtype and device are preserved.
        voxel_size: Positive, finite real scalar specifying the cube side length
            in the same units as the points. Tensor-valued sizes are not supported.

    Returns:
        A new Pointclouds with the same batch size and optional attributes. Its
        number of points per cloud is the number of occupied voxels. Input data
        is not modified. Empty clouds remain empty.

    Raises:
        TypeError: If pointclouds is not a Pointclouds object or voxel_size is
            not a real scalar.
        ValueError: If voxel_size is not positive and finite, input tensors are
            nonfinite or have unsupported dtypes, or quantized coordinates are
            outside the signed 64-bit integer range.
    """
    if not isinstance(pointclouds, Pointclouds):
        raise TypeError("pointclouds must be a Pointclouds object")
    if isinstance(voxel_size, bool) or not isinstance(voxel_size, Real):
        raise TypeError("voxel_size must be a real scalar")
    voxel_size = float(voxel_size)
    if not math.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be positive and finite")

    points_list = pointclouds.points_list()
    features_list = pointclouds.features_list()
    normals_list = pointclouds.normals_list()
    for name, tensors in (
        ("points", points_list),
        ("features", features_list),
        ("normals", normals_list),
    ):
        if tensors is not None:
            for tensor in tensors:
                if tensor.dtype not in (torch.float32, torch.float64):
                    raise ValueError(f"{name} must have dtype float32 or float64")
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"{name} must be finite")

    if pointclouds.isempty():
        return pointclouds.clone()

    output_points = []
    output_features = [] if features_list is not None else None
    output_normals = [] if normals_list is not None else None
    for i, points in enumerate(points_list):
        if len(points) == 0:
            output_points.append(points.clone())
            if features_list is not None:
                output_features.append(features_list[i].clone())
            if normals_list is not None:
                output_normals.append(normals_list[i].clone())
            continue

        # Quantize in float64 to avoid casting small voxel sizes to float32 zero.
        # Check before converting to int64: out-of-range casts can alias faraway
        # points to the same integer, silently merging unrelated voxels.
        coordinates = torch.floor(points.detach().to(torch.float64) / voxel_size)
        if not ((coordinates >= -(2**63)) & (coordinates < 2**63)).all():
            raise ValueError("voxel coordinates are outside the int64 range")
        # Keep all three coordinates instead of a flattened bounding-grid index,
        # whose product could overflow even when each coordinate is representable.
        _, inverse, counts = torch.unique(
            coordinates.to(torch.int64),
            dim=0,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        output_points.append(_mean_per_voxel(points, inverse, counts))
        if features_list is not None:
            output_features.append(_mean_per_voxel(features_list[i], inverse, counts))
        if normals_list is not None:
            output_normals.append(_mean_per_voxel(normals_list[i], inverse, counts))

    return Pointclouds(output_points, normals=output_normals, features=output_features)


def _mean_per_voxel(
    values: torch.Tensor, inverse: torch.Tensor, counts: torch.Tensor
) -> torch.Tensor:
    return _VoxelMean.apply(values, inverse, counts)


class _VoxelMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, inverse, counts):
        ctx.save_for_backward(inverse, counts)
        shape = (len(counts), values.shape[1])
        # Normalize each channel within each voxel before summing. Dividing
        # values directly by counts can underflow subnormals, whereas summing
        # first can overflow even if the mean is representable.
        scale = values.new_zeros(shape).scatter_reduce_(
            0,
            inverse[:, None].expand_as(values),
            values.abs(),
            reduce="amax",
            include_self=True,
        )
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        normalized = values / scale[inverse]
        sums = values.new_zeros(shape).index_add(0, inverse, normalized)
        mean = sums / counts[:, None].to(values.dtype)
        # A normalized mean is in [-1, 1]. Guard rounding at the dtype limit
        # before rescaling, so a finite constant input cannot produce infinity.
        return mean.clamp(-1, 1) * scale

    @staticmethod
    def backward(ctx, grad_output):
        inverse, counts = ctx.saved_tensors
        # The mean is linear within a voxel. Using its analytic derivative also
        # avoids underflow in a chain rule through the numerical scaling above.
        grad_values = grad_output[inverse] / counts[inverse, None].to(grad_output.dtype)
        return grad_values, None, None
