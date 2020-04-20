#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


from typing import NamedTuple, Optional

import torch
import torch.nn as nn

from ..cameras import get_world_to_view_transform
from .rasterize_points import rasterize_points


# Class to store the outputs of point rasterization
class PointFragments(NamedTuple):
    idx: torch.Tensor
    zbuf: torch.Tensor
    dists: torch.Tensor


# Class to store the point rasterization params with defaults
class PointsRasterizationSettings:
    __slots__ = [
        "image_size",
        "radius",
        "points_per_pixel",
        "bin_size",
        "max_points_per_bin",
    ]

    def __init__(
        self,
        image_size: int = 256,
        radius: float = 0.01,
        points_per_pixel: int = 8,
        bin_size: Optional[int] = None,
        max_points_per_bin: Optional[int] = None,
    ):
        self.image_size = image_size
        self.radius = radius
        self.points_per_pixel = points_per_pixel
        self.bin_size = bin_size
        self.max_points_per_bin = max_points_per_bin


class PointsRasterizer(nn.Module):
    """
    This class implements methods for rasterizing a batch of pointclouds.
    """

    def __init__(self, cameras, raster_settings=None):
        """
        cameras: A cameras object which has a  `transform_points` method
                which returns the transformed points after applying the
                world-to-view and view-to-screen
                transformations.
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

    def transform(self, point_clouds, **kwargs) -> torch.Tensor:
        """
        Args:
            point_clouds: a set of point clouds

        Returns:
            points_screen: the points with the vertex positions in screen
            space

        NOTE: keeping this as a separate function for readability but it could
        be moved into forward.
        """
        cameras = kwargs.get("cameras", self.cameras)

        pts_world = point_clouds.points_padded()
        pts_world_packed = point_clouds.points_packed()
        pts_screen = cameras.transform_points(pts_world, **kwargs)

        # NOTE: Retaining view space z coordinate for now.
        # TODO: Remove this line when the convention for the z coordinate in
        # the rasterizer is decided. i.e. retain z in view space or transform
        # to a different range.
        view_transform = get_world_to_view_transform(R=cameras.R, T=cameras.T)
        verts_view = view_transform.transform_points(pts_world)
        pts_screen[..., 2] = verts_view[..., 2]

        # Offset points of input pointcloud to reuse cached padded/packed calculations.
        pad_to_packed_idx = point_clouds.padded_to_packed_idx()
        pts_screen_packed = pts_screen.view(-1, 3)[pad_to_packed_idx, :]
        pts_packed_offset = pts_screen_packed - pts_world_packed
        point_clouds = point_clouds.offset(pts_packed_offset)
        return point_clouds

    def forward(self, point_clouds, **kwargs) -> PointFragments:
        """
        Args:
            point_clouds: a set of point clouds with coordinates in world space.
        Returns:
            PointFragments: Rasterization outputs as a named tuple.
        """
        points_screen = self.transform(point_clouds, **kwargs)
        raster_settings = kwargs.get("raster_settings", self.raster_settings)
        idx, zbuf, dists2 = rasterize_points(
            points_screen,
            image_size=raster_settings.image_size,
            radius=raster_settings.radius,
            points_per_pixel=raster_settings.points_per_pixel,
            bin_size=raster_settings.bin_size,
            max_points_per_bin=raster_settings.max_points_per_bin,
        )
        return PointFragments(idx=idx, zbuf=zbuf, dists=dists2)
