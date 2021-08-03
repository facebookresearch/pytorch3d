#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import NamedTuple, Optional, Tuple, Union

import torch
import torch.nn as nn

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
        image_size: Union[int, Tuple[int, int]] = 256,
        radius: Union[float, torch.Tensor] = 0.01,
        points_per_pixel: int = 8,
        bin_size: Optional[int] = None,
        max_points_per_bin: Optional[int] = None,
    ) -> None:
        self.image_size = image_size
        self.radius = radius
        self.points_per_pixel = points_per_pixel
        self.bin_size = bin_size
        self.max_points_per_bin = max_points_per_bin


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

    def transform(self, point_clouds, **kwargs) -> torch.Tensor:
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
        # view to NDC transform
        to_ndc_transform = cameras.get_ndc_camera_transform(**kwargs)
        projection_transform = cameras.get_projection_transform(**kwargs).compose(
            to_ndc_transform
        )
        pts_ndc = projection_transform.transform_points(pts_view, eps=eps)

        pts_ndc[..., 2] = pts_view[..., 2]
        point_clouds = point_clouds.update_padded(pts_ndc)
        return point_clouds

    def to(self, device):
        # Manually move to device cameras as it is not a subclass of nn.Module
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
