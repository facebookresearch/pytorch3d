#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import torch
import torch.nn as nn


# A renderer class should be initialized with a
# function for rasterization and a function for compositing.
# The rasterizer should:
#     - transform inputs from world -> screen space
#     - rasterize inputs
#     - return fragments
# The compositor can take fragments as input along with any other properties of
# the scene and generate images.

# E.g. rasterize inputs and then shade
#
# fragments = self.rasterize(point_clouds)
# images = self.compositor(fragments, point_clouds)
# return images


class PointsRenderer(nn.Module):
    """
    A class for rendering a batch of points. The class should
    be initialized with a rasterizer and compositor class which each have a forward
    function.
    """

    def __init__(self, rasterizer, compositor):
        super().__init__()
        self.rasterizer = rasterizer
        self.compositor = compositor

    def forward(self, point_clouds, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(point_clouds, **kwargs)

        # Construct weights based on the distance of a point to the true point.
        # However, this could be done differently: e.g. predicted as opposed
        # to a function of the weights.
        r = self.rasterizer.raster_settings.radius

        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs
        )

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)

        return images
