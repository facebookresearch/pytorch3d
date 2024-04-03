#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple, Union

import torch
import torch.nn as nn
from pytorch3d.renderer.cameras import try_get_projection_transform
from pytorch3d.structures import Pointclouds

from .rasterize_points import rasterize_points


class PointFragments(NamedTuple):
    """
    Class to store the outputs of point rasterization

    Members:
        idx: int32 Tensor of shape (N, image_size, image_size, points_per_pixel)
            giving the indices of the nearest points at each pixel, in ascending
            z-order. Concretely `idx[n, y, x, k] = p` means that `points[p]` is the kth
            closest point (along the z-direction) to pixel (y, x) - note that points
            represents the packed points of shape (P, 3).
            Pixels that are hit by fewer than points_per_pixel are padded with -1.
        zbuf: Tensor of shape (N, image_size, image_size, points_per_pixel)
            giving the z-coordinates of the nearest points at each pixel, sorted in
            z-order. Concretely, if `idx[n, y, x, k] = p` then
            `zbuf[n, y, x, k] = points[n, p, 2]`. Pixels hit by fewer than
            points_per_pixel are padded with -1.
        dists: Tensor of shape (N, image_size, image_size, points_per_pixel)
            giving the squared Euclidean distance (in NDC units) in the x/y plane
            for each point closest to the pixel. Concretely if `idx[n, y, x, k] = p`
            then `dists[n, y, x, k]` is the squared distance between the pixel (y, x)
            and the point `(points[n, p, 0], points[n, p, 1])`. Pixels hit with fewer
            than points_per_pixel are padded with -1.
    """

    idx: torch.Tensor
    zbuf: torch.Tensor
    dists: torch.Tensor


@dataclass
class PointsRasterizationSettings:
    """
    Class to store the point rasterization params with defaults

    Members:
        image_size: Either common height and width or (height, width), in pixels.
        radius: The radius (in NDC units) of each disk to be rasterized.
            This can either be a float in which case the same radius is used
            for each point, or a torch.Tensor of shape (N, P) giving a radius
            per point in the batch.
        points_per_pixel: (int) Number of points to keep track of per pixel.
            We return the nearest points_per_pixel points along the z-axis.
        bin_size: Size of bins to use for coarse-to-fine rasterization. Setting
            bin_size=0 uses naive rasterization; setting bin_size=None attempts
            to set it heuristically based on the shape of the input. This should
            not affect the output, but can affect the speed of the forward pass.
        max_points_per_bin: Only applicable when using coarse-to-fine
            rasterization (bin_size != 0); this is the maximum number of points
            allowed within each bin. This should not affect the output values,
            but can affect the memory usage in the forward pass.
            Setting max_points_per_bin=None attempts to set with a heuristic.
    """

    image_size: Union[int, Tuple[int, int]] = 256
    radius: Union[float, torch.Tensor] = 0.01
    points_per_pixel: int = 8
    bin_size: Optional[int] = None
    max_points_per_bin: Optional[int] = None


class PointsRasterizer(nn.Module):
    """
    This class implements methods for rasterizing a batch of pointclouds.
    """

    def __init__(self, cameras=None, raster_settings=None) -> None:
        """
        cameras: A cameras object which has a  `transform_points` method
                which returns the transformed points after applying the
                world-to-view and view-to-ndc transformations.
            raster_settings: the parameters for rasterization. This should be a
                named tuple.

        All these initial settings can be overridden by passing keyword
        arguments to the forward function.
        """
        super().__init__()
        if raster_settings is None:
            raster_settings = PointsRasterizationSettings()

        self.cameras = cameras
        self.raster_settings = raster_settings

    def transform(self, point_clouds, **kwargs) -> Pointclouds:
        """
        Args:
            point_clouds: a set of point clouds

        Returns:
            points_proj: the points with positions projected
            in NDC space

        NOTE: keeping this as a separate function for readability but it could
        be moved into forward.
        """
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of PointsRasterizer"
            raise ValueError(msg)

        pts_world = point_clouds.points_padded()
        # NOTE: Retaining view space z coordinate for now.
        # TODO: Remove this line when the convention for the z coordinate in
        # the rasterizer is decided. i.e. retain z in view space or transform
        # to a different range.
        eps = kwargs.get("eps", None)
        pts_view = cameras.get_world_to_view_transform(**kwargs).transform_points(
            pts_world, eps=eps
        )
        to_ndc_transform = cameras.get_ndc_camera_transform(**kwargs)
        projection_transform = try_get_projection_transform(cameras, kwargs)
        if projection_transform is not None:
            projection_transform = projection_transform.compose(to_ndc_transform)
            pts_ndc = projection_transform.transform_points(pts_view, eps=eps)
        else:
            # Call transform_points instead of explicitly composing transforms to handle
            # the case, where camera class does not have a projection matrix form.
            pts_proj = cameras.transform_points(pts_world, eps=eps)
            pts_ndc = to_ndc_transform.transform_points(pts_proj, eps=eps)

        pts_ndc[..., 2] = pts_view[..., 2]
        point_clouds = point_clouds.update_padded(pts_ndc)
        return point_clouds

    def to(self, device):
        # Manually move to device cameras as it is not a subclass of nn.Module
        if self.cameras is not None:
            self.cameras = self.cameras.to(device)
        return self

    def forward(self, point_clouds, **kwargs) -> PointFragments:
        """
        Args:
            point_clouds: a set of point clouds with coordinates in world space.
        Returns:
            PointFragments: Rasterization outputs as a named tuple.
        """
        points_proj = self.transform(point_clouds, **kwargs)
        raster_settings = kwargs.get("raster_settings", self.raster_settings)
        idx, zbuf, dists2 = rasterize_points(
            points_proj,
            image_size=raster_settings.image_size,
            radius=raster_settings.radius,
            points_per_pixel=raster_settings.points_per_pixel,
            bin_size=raster_settings.bin_size,
            max_points_per_bin=raster_settings.max_points_per_bin,
        )
        return PointFragments(idx=idx, zbuf=zbuf, dists=dists2)
