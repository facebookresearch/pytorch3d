# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from pytorch3d import _C

from pytorch3d.renderer.mesh.rasterize_meshes import pix_to_non_square_ndc

from ..utils import parse_image_size


# Maximum number of faces per bins for
# coarse-to-fine rasterization
kMaxPointsPerBin = 22


def rasterize_points(
    pointclouds,
    image_size: Union[int, List[int], Tuple[int, int]] = 256,
    radius: Union[float, List, Tuple, torch.Tensor] = 0.01,
    points_per_pixel: int = 8,
    bin_size: Optional[int] = None,
    max_points_per_bin: Optional[int] = None,
):
    """
    Each pointcloud is rasterized onto a separate image of shape
    (H, W) if `image_size` is a tuple or (image_size, image_size) if it
    is an int.

    If the desired image size is non square (i.e. a tuple of (H, W) where H != W)
    the aspect ratio needs special consideration. There are two aspect ratios
    to be aware of:
        - the aspect ratio of each pixel
        - the aspect ratio of the output image
    The camera can be used to set the pixel aspect ratio. In the rasterizer,
    we assume square pixels, but variable image aspect ratio (i.e rectangle images).

    In most cases you will want to set the camera aspect ratio to
    1.0 (i.e. square pixels) and only vary the
    `image_size` (i.e. the output image dimensions in pix

    Args:
        pointclouds: A Pointclouds object representing a batch of point clouds to be
            rasterized. This is a batch of N pointclouds, where each point cloud
            can have a different number of points; the coordinates of each point
            are (x, y, z). The coordinates are expected to
            be in normalized device coordinates (NDC): [-1, 1]^3 with the camera at
            (0, 0, 0); In the camera coordinate frame the x-axis goes from right-to-left,
            the y-axis goes from bottom-to-top, and the z-axis goes from back-to-front.
        image_size: Size in pixels of the output image to be rasterized.
            Can optionally be a tuple of (H, W) in the case of non square images.
        radius (Optional): The radius (in NDC units) of the disk to
            be rasterized. This can either be a float in which case the same radius is used
            for each point, or a torch.Tensor of shape (N, P) giving a radius per point
            in the batch.
        points_per_pixel (Optional): We will keep track of this many points per
            pixel, returning the nearest points_per_pixel points along the z-axis
        bin_size: Size of bins to use for coarse-to-fine rasterization. Setting
            bin_size=0 uses naive rasterization; setting bin_size=None attempts to
            set it heuristically based on the shape of the input. This should not
            affect the output, but can affect the speed of the forward pass.
        max_points_per_bin: Only applicable when using coarse-to-fine rasterization
            (bin_size > 0); this is the maximum number of points allowed within each
            bin. This should not affect the output values, but can affect
            the memory usage in the forward pass.

    Returns:
        3-element tuple containing

        - **idx**: int32 Tensor of shape (N, image_size, image_size, points_per_pixel)
          giving the indices of the nearest points at each pixel, in ascending
          z-order. Concretely `idx[n, y, x, k] = p` means that `points[p]` is the kth
          closest point (along the z-direction) to pixel (y, x) - note that points
          represents the packed points of shape (P, 3).
          Pixels that are hit by fewer than points_per_pixel are padded with -1.
        - **zbuf**: Tensor of shape (N, image_size, image_size, points_per_pixel)
          giving the z-coordinates of the nearest points at each pixel, sorted in
          z-order. Concretely, if `idx[n, y, x, k] = p` then
          `zbuf[n, y, x, k] = points[n, p, 2]`. Pixels hit by fewer than
          points_per_pixel are padded with -1
        - **dists2**: Tensor of shape (N, image_size, image_size, points_per_pixel)
          giving the squared Euclidean distance (in NDC units) in the x/y plane
          for each point closest to the pixel. Concretely if `idx[n, y, x, k] = p`
          then `dists[n, y, x, k]` is the squared distance between the pixel (y, x)
          and the point `(points[n, p, 0], points[n, p, 1])`. Pixels hit with fewer
          than points_per_pixel are padded with -1.

        In the case that image_size is a tuple of (H, W) then the outputs
        will be of shape `(N, H, W, ...)`.
    """
    points_packed = pointclouds.points_packed()
    cloud_to_packed_first_idx = pointclouds.cloud_to_packed_first_idx()
    num_points_per_cloud = pointclouds.num_points_per_cloud()

    radius = _format_radius(radius, pointclouds)

    # In the case that H != W use the max image size to set the bin_size
    # to accommodate the num bins constraint in the coarse rasterizer.
    # If the ratio of H:W is large this might cause issues as the smaller
    # dimension will have fewer bins.
    # TODO: consider a better way of setting the bin size.
    im_size = parse_image_size(image_size)
    max_image_size = max(*im_size)

    if bin_size is None:
        if not points_packed.is_cuda:
            # Binned CPU rasterization not fully implemented
            bin_size = 0
        else:
            bin_size = int(2 ** max(np.ceil(np.log2(max_image_size)) - 4, 4))

    if bin_size != 0:
        # There is a limit on the number of points per bin in the cuda kernel.
        points_per_bin = 1 + (max_image_size - 1) // bin_size
        if points_per_bin >= kMaxPointsPerBin:
            raise ValueError(
                "bin_size too small, number of points per bin must be less than %d; got %d"
                % (kMaxPointsPerBin, points_per_bin)
            )

    if max_points_per_bin is None:
        max_points_per_bin = int(max(10000, pointclouds._P / 5))

    # Function.apply cannot take keyword args, so we handle defaults in this
    # wrapper and call apply with positional args only
    return _RasterizePoints.apply(
        points_packed,
        cloud_to_packed_first_idx,
        num_points_per_cloud,
        im_size,
        radius,
        points_per_pixel,
        bin_size,
        max_points_per_bin,
    )


def _format_radius(
    radius: Union[float, List, Tuple, torch.Tensor], pointclouds
) -> torch.Tensor:
    """
    Format the radius as a torch tensor of shape (P_packed,)
    where P_packed is the total number of points in the
    batch (i.e. pointclouds.points_packed().shape[0]).

    This will enable support for a different size radius
    for each point in the batch.

    Args:
        radius: can be a float, List, Tuple or tensor of
            shape (N, P_padded) where P_padded is the
            maximum number of points for each pointcloud
            in the batch.

    Returns:
        radius: torch.Tensor of shape (P_packed)
    """
    N, P_padded = pointclouds._N, pointclouds._P
    points_packed = pointclouds.points_packed()
    P_packed = points_packed.shape[0]
    if isinstance(radius, (list, tuple)):
        radius = torch.tensor(radius).type_as(points_packed)
    if isinstance(radius, torch.Tensor):
        if N == 1 and radius.ndim == 1:
            radius = radius[None, ...]
        if radius.shape != (N, P_padded):
            msg = "radius must be of shape (N, P): got %s"
            raise ValueError(msg % (repr(radius.shape)))
        else:
            padded_to_packed_idx = pointclouds.padded_to_packed_idx()
            radius = radius.view(-1)[padded_to_packed_idx]
    elif isinstance(radius, float):
        radius = torch.full((P_packed,), fill_value=radius).type_as(points_packed)
    else:
        msg = "radius must be a float, list, tuple or tensor; got %s"
        raise ValueError(msg % type(radius))
    return radius


class _RasterizePoints(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        points,  # (P, 3)
        cloud_to_packed_first_idx,
        num_points_per_cloud,
        image_size: Union[List[int], Tuple[int, int]] = (256, 256),
        radius: Union[float, torch.Tensor] = 0.01,
        points_per_pixel: int = 8,
        bin_size: int = 0,
        max_points_per_bin: int = 0,
    ):
        # TODO: Add better error handling for when there are more than
        # max_points_per_bin in any bin.
        args = (
            points,
            cloud_to_packed_first_idx,
            num_points_per_cloud,
            image_size,
            radius,
            points_per_pixel,
            bin_size,
            max_points_per_bin,
        )
        # pyre-fixme[16]: Module `pytorch3d` has no attribute `_C`.
        idx, zbuf, dists = _C.rasterize_points(*args)
        ctx.save_for_backward(points, idx)
        ctx.mark_non_differentiable(idx)
        return idx, zbuf, dists

    @staticmethod
    def backward(ctx, grad_idx, grad_zbuf, grad_dists):
        grad_points = None
        grad_cloud_to_packed_first_idx = None
        grad_num_points_per_cloud = None
        grad_image_size = None
        grad_radius = None
        grad_points_per_pixel = None
        grad_bin_size = None
        grad_max_points_per_bin = None
        points, idx = ctx.saved_tensors
        args = (points, idx, grad_zbuf, grad_dists)
        grad_points = _C.rasterize_points_backward(*args)
        grads = (
            grad_points,
            grad_cloud_to_packed_first_idx,
            grad_num_points_per_cloud,
            grad_image_size,
            grad_radius,
            grad_points_per_pixel,
            grad_bin_size,
            grad_max_points_per_bin,
        )
        return grads


def rasterize_points_python(
    pointclouds,
    image_size: Union[int, Tuple[int, int]] = 256,
    radius: Union[float, torch.Tensor] = 0.01,
    points_per_pixel: int = 8,
):
    """
    Naive pure PyTorch implementation of pointcloud rasterization.

    Inputs / Outputs: Same as above
    """
    N = len(pointclouds)
    H, W = (
        image_size
        if isinstance(image_size, (tuple, list))
        else (image_size, image_size)
    )
    K = points_per_pixel
    device = pointclouds.device

    points_packed = pointclouds.points_packed()
    cloud_to_packed_first_idx = pointclouds.cloud_to_packed_first_idx()
    num_points_per_cloud = pointclouds.num_points_per_cloud()

    # Support variable size radius for each point in the batch
    radius = _format_radius(radius, pointclouds)

    # Initialize output tensors.
    point_idxs = torch.full(
        (N, H, W, K), fill_value=-1, dtype=torch.int32, device=device
    )
    zbuf = torch.full((N, H, W, K), fill_value=-1, dtype=torch.float32, device=device)
    pix_dists = torch.full(
        (N, H, W, K), fill_value=-1, dtype=torch.float32, device=device
    )

    # NDC is from [-1, 1]. Get pixel size using specified image size.
    radius2 = radius * radius

    # Iterate through the batch of point clouds.
    for n in range(N):
        point_start_idx = cloud_to_packed_first_idx[n]
        point_stop_idx = point_start_idx + num_points_per_cloud[n]

        # Iterate through the horizontal lines of the image from top to bottom.
        for yi in range(H):
            # Y coordinate of one end of the image. Reverse the ordering
            # of yi so that +Y is pointing up in the image.
            yfix = H - 1 - yi
            yf = pix_to_non_square_ndc(yfix, H, W)

            # Iterate through pixels on this horizontal line, left to right.
            for xi in range(W):
                # X coordinate of one end of the image. Reverse the ordering
                # of xi so that +X is pointing to the left in the image.
                xfix = W - 1 - xi
                xf = pix_to_non_square_ndc(xfix, W, H)

                top_k_points = []
                # Check whether each point in the batch affects this pixel.
                for p in range(point_start_idx, point_stop_idx):
                    px, py, pz = points_packed[p, :]
                    r = radius2[p]
                    if pz < 0:
                        continue
                    dx = px - xf
                    dy = py - yf
                    dist2 = dx * dx + dy * dy
                    if dist2 < r:
                        top_k_points.append((pz, p, dist2))
                        top_k_points.sort()
                        if len(top_k_points) > K:
                            top_k_points = top_k_points[:K]
                for k, (pz, p, dist2) in enumerate(top_k_points):
                    zbuf[n, yi, xi, k] = pz
                    point_idxs[n, yi, xi, k] = p
                    pix_dists[n, yi, xi, k] = dist2
    return point_idxs, zbuf, pix_dists
