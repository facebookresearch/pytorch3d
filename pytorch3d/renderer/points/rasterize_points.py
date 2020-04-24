# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Optional

import torch
from pytorch3d import _C
from pytorch3d.renderer.mesh.rasterize_meshes import pix_to_ndc


# Maxinum number of faces per bins for
# coarse-to-fine rasterization
kMaxPointsPerBin = 22


# TODO(jcjohns): Support non-square images
def rasterize_points(
    pointclouds,
    image_size: int = 256,
    radius: float = 0.01,
    points_per_pixel: int = 8,
    bin_size: Optional[int] = None,
    max_points_per_bin: Optional[int] = None,
):
    """
    Pointcloud rasterization

    Args:
        pointclouds: A Pointclouds object representing a batch of point clouds to be
            rasterized. This is a batch of N pointclouds, where each point cloud
            can have a different number of points; the coordinates of each point
            are (x, y, z). The coordinates are expected to
            be in normalized device coordinates (NDC): [-1, 1]^3 with the camera at
            (0, 0, 0); In the camera coordinate frame the x-axis goes from right-to-left,
            the y-axis goes from bottom-to-top, and the z-axis goes from back-to-front.
        image_size: Integer giving the resolution of the rasterized image
        radius (Optional): Float giving the radius (in NDC units) of the disk to
            be rasterized for each point.
        points_per_pixel (Optional): We will keep track of this many points per
            pixel, returning the nearest points_per_pixel points along the z-axis
        bin_size: Size of bins to use for coarse-to-fine rasterization. Setting
            bin_size=0 uses naive rasterization; setting bin_size=None attempts to
            set it heuristically based on the shape of the input. This should not
            affect the output, but can affect the speed of the forward pass.
        points_per_bin: Only applicable when using coarse-to-fine rasterization
            (bin_size > 0); this is the maxiumum number of points allowed within each
            bin. If more than this many points actually fall into a bin, an error
            will be raised. This should not affect the output values, but can affect
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
    """
    points_packed = pointclouds.points_packed()
    cloud_to_packed_first_idx = pointclouds.cloud_to_packed_first_idx()
    num_points_per_cloud = pointclouds.num_points_per_cloud()

    if bin_size is None:
        if not points_packed.is_cuda:
            # Binned CPU rasterization not fully implemented
            bin_size = 0
        else:
            # TODO: These heuristics are not well-thought out!
            if image_size <= 64:
                bin_size = 8
            elif image_size <= 256:
                bin_size = 16
            elif image_size <= 512:
                bin_size = 32
            elif image_size <= 1024:
                bin_size = 64

    if bin_size != 0:
        # There is a limit on the number of points per bin in the cuda kernel.
        points_per_bin = 1 + (image_size - 1) // bin_size
        if points_per_bin >= kMaxPointsPerBin:
            raise ValueError(
                "bin_size too small, number of points per bin must be less than %d; got %d"
                % (kMaxPointsPerBin, points_per_bin)
            )

    if max_points_per_bin is None:
        max_points_per_bin = int(max(10000, points_packed.shape[0] / 5))

    # Function.apply cannot take keyword args, so we handle defaults in this
    # wrapper and call apply with positional args only
    return _RasterizePoints.apply(
        points_packed,
        cloud_to_packed_first_idx,
        num_points_per_cloud,
        image_size,
        radius,
        points_per_pixel,
        bin_size,
        max_points_per_bin,
    )


class _RasterizePoints(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        points,  # (P, 3)
        cloud_to_packed_first_idx,
        num_points_per_cloud,
        image_size: int = 256,
        radius: float = 0.01,
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
        idx, zbuf, dists = _C.rasterize_points(*args)
        ctx.save_for_backward(points, idx)
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
    pointclouds, image_size: int = 256, radius: float = 0.01, points_per_pixel: int = 8
):
    """
    Naive pure PyTorch implementation of pointcloud rasterization.

    Inputs / Outputs: Same as above
    """
    N = len(pointclouds)
    S, K = image_size, points_per_pixel
    device = pointclouds.device

    points_packed = pointclouds.points_packed()
    cloud_to_packed_first_idx = pointclouds.cloud_to_packed_first_idx()
    num_points_per_cloud = pointclouds.num_points_per_cloud()

    # Intialize output tensors.
    point_idxs = torch.full(
        (N, S, S, K), fill_value=-1, dtype=torch.int32, device=device
    )
    zbuf = torch.full((N, S, S, K), fill_value=-1, dtype=torch.float32, device=device)
    pix_dists = torch.full(
        (N, S, S, K), fill_value=-1, dtype=torch.float32, device=device
    )

    # NDC is from [-1, 1]. Get pixel size using specified image size.
    radius2 = radius * radius

    # Iterate through the batch of point clouds.
    for n in range(N):
        point_start_idx = cloud_to_packed_first_idx[n]
        point_stop_idx = point_start_idx + num_points_per_cloud[n]

        # Iterate through the horizontal lines of the image from top to bottom.
        for yi in range(S):
            # Y coordinate of one end of the image. Reverse the ordering
            # of yi so that +Y is pointing up in the image.
            yfix = S - 1 - yi
            yf = pix_to_ndc(yfix, S)

            # Iterate through pixels on this horizontal line, left to right.
            for xi in range(S):
                # X coordinate of one end of the image. Reverse the ordering
                # of xi so that +X is pointing to the left in the image.
                xfix = S - 1 - xi
                xf = pix_to_ndc(xfix, S)

                top_k_points = []
                # Check whether each point in the batch affects this pixel.
                for p in range(point_start_idx, point_stop_idx):
                    px, py, pz = points_packed[p, :]
                    if pz < 0:
                        continue
                    dx = px - xf
                    dy = py - yf
                    dist2 = dx * dx + dy * dy
                    if dist2 < radius2:
                        top_k_points.append((pz, p, dist2))
                        top_k_points.sort()
                        if len(top_k_points) > K:
                            top_k_points = top_k_points[:K]
                for k, (pz, p, dist2) in enumerate(top_k_points):
                    zbuf[n, yi, xi, k] = pz
                    point_idxs[n, yi, xi, k] = p
                    pix_dists[n, yi, xi, k] = dist2
    return point_idxs, zbuf, pix_dists
