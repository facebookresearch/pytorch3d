# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from ..cameras import CamerasBase
from .utils import RayBundle


"""
This file defines three raysampling techniques:
    - GridRaysampler which can be used to sample rays from pixels of an image grid
    - NDCGridRaysampler which can be used to sample rays from pixels of an image grid,
        which follows the pytorch3d convention for image grid coordinates
    - MonteCarloRaysampler which randomly selects image pixels and emits rays from them
"""


class GridRaysampler(torch.nn.Module):
    """
    Samples a fixed number of points along rays which are regularly distributed
    in a batch of rectangular image grids. Points along each ray
    have uniformly-spaced z-coordinates between a predefined
    minimum and maximum depth.

    The raysampler first generates a 3D coordinate grid of the following form:
    ```
       / min_x, min_y, max_depth -------------- / max_x, min_y, max_depth
      /                                        /|
     /                                        / |     ^
    / min_depth                    min_depth /  |     |
    min_x ----------------------------- max_x   |     | image
    min_y                               min_y   |     | height
    |                                       |   |     |
    |                                       |   |     v
    |                                       |   |
    |                                       |   / max_x, max_y,     ^
    |                                       |  /  max_depth        /
    min_x                               max_y /                   / n_pts_per_ray
    max_y ----------------------------- max_x/ min_depth         v
              < --- image_width --- >
    ```

    In order to generate ray points, `GridRaysampler` takes each 3D point of
    the grid (with coordinates `[x, y, depth]`) and unprojects it
    with `cameras.unproject_points([x, y, depth])`, where `cameras` are an
    additional input to the `forward` function.

    Note that this is a generic implementation that can support any image grid
    coordinate convention. For a raysampler which follows the PyTorch3D
    coordinate conventions please refer to `NDCGridRaysampler`.
    As such, `NDCGridRaysampler` is a special case of `GridRaysampler`.
    """

    def __init__(
        self,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        image_width: int,
        image_height: int,
        n_pts_per_ray: int,
        min_depth: float,
        max_depth: float,
    ) -> None:
        """
        Args:
            min_x: The leftmost x-coordinate of each ray's source pixel's center.
            max_x: The rightmost x-coordinate of each ray's source pixel's center.
            min_y: The topmost y-coordinate of each ray's source pixel's center.
            max_y: The bottommost y-coordinate of each ray's source pixel's center.
            image_width: The horizontal size of the image grid.
            image_height: The vertical size of the image grid.
            n_pts_per_ray: The number of points sampled along each ray.
            min_depth: The minimum depth of a ray-point.
            max_depth: The maximum depth of a ray-point.
        """
        super().__init__()
        self._n_pts_per_ray = n_pts_per_ray
        self._min_depth = min_depth
        self._max_depth = max_depth

        # get the initial grid of image xy coords
        _xy_grid = torch.stack(
            tuple(
                reversed(
                    torch.meshgrid(
                        torch.linspace(min_y, max_y, image_height, dtype=torch.float32),
                        torch.linspace(min_x, max_x, image_width, dtype=torch.float32),
                    )
                )
            ),
            dim=-1,
        )
        self.register_buffer("_xy_grid", _xy_grid, persistent=False)

    def forward(self, cameras: CamerasBase, **kwargs) -> RayBundle:
        """
        Args:
            cameras: A batch of `batch_size` cameras from which the rays are emitted.
        Returns:
            A named tuple RayBundle with the following fields:
            origins: A tensor of shape
                `(batch_size, image_height, image_width, 3)`
                denoting the locations of ray origins in the world coordinates.
            directions: A tensor of shape
                `(batch_size, image_height, image_width, 3)`
                denoting the directions of each ray in the world coordinates.
            lengths: A tensor of shape
                `(batch_size, image_height, image_width, n_pts_per_ray)`
                containing the z-coordinate (=depth) of each ray in world units.
            xys: A tensor of shape
                `(batch_size, image_height, image_width, 2)`
                containing the 2D image coordinates of each ray.
        """

        batch_size = cameras.R.shape[0]

        device = cameras.device

        # expand the (H, W, 2) grid batch_size-times to (B, H, W, 2)
        xy_grid = self._xy_grid.to(device)[None].expand(
            batch_size, *self._xy_grid.shape
        )

        return _xy_to_ray_bundle(
            cameras, xy_grid, self._min_depth, self._max_depth, self._n_pts_per_ray
        )


class NDCGridRaysampler(GridRaysampler):
    """
    Samples a fixed number of points along rays which are regularly distributed
    in a batch of rectangular image grids. Points along each ray
    have uniformly-spaced z-coordinates between a predefined minimum and maximum depth.

    `NDCGridRaysampler` follows the screen conventions of the `Meshes` and `Pointclouds`
    renderers. I.e. the pixel coordinates are in [-1, 1]x[-u, u] or [-u, u]x[-1, 1]
    where u > 1 is the aspect ratio of the image.
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        n_pts_per_ray: int,
        min_depth: float,
        max_depth: float,
    ) -> None:
        """
        Args:
            image_width: The horizontal size of the image grid.
            image_height: The vertical size of the image grid.
            n_pts_per_ray: The number of points sampled along each ray.
            min_depth: The minimum depth of a ray-point.
            max_depth: The maximum depth of a ray-point.
        """
        if image_width >= image_height:
            range_x = image_width / image_height
            range_y = 1.0
        else:
            range_x = 1.0
            range_y = image_height / image_width

        half_pix_width = range_x / image_width
        half_pix_height = range_y / image_height
        super().__init__(
            min_x=range_x - half_pix_width,
            max_x=-range_x + half_pix_width,
            min_y=range_y - half_pix_height,
            max_y=-range_y + half_pix_height,
            image_width=image_width,
            image_height=image_height,
            n_pts_per_ray=n_pts_per_ray,
            min_depth=min_depth,
            max_depth=max_depth,
        )


class MonteCarloRaysampler(torch.nn.Module):
    """
    Samples a fixed number of pixels within denoted xy bounds uniformly at random.
    For each pixel, a fixed number of points is sampled along its ray at uniformly-spaced
    z-coordinates such that the z-coordinates range between a predefined minimum
    and maximum depth.
    """

    def __init__(
        self,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        n_rays_per_image: int,
        n_pts_per_ray: int,
        min_depth: float,
        max_depth: float,
    ) -> None:
        """
        Args:
            min_x: The smallest x-coordinate of each ray's source pixel.
            max_x: The largest x-coordinate of each ray's source pixel.
            min_y: The smallest y-coordinate of each ray's source pixel.
            max_y: The largest y-coordinate of each ray's source pixel.
            n_rays_per_image: The number of rays randomly sampled in each camera.
            n_pts_per_ray: The number of points sampled along each ray.
            min_depth: The minimum depth of each ray-point.
            max_depth: The maximum depth of each ray-point.
        """
        super().__init__()
        self._min_x = min_x
        self._max_x = max_x
        self._min_y = min_y
        self._max_y = max_y
        self._n_rays_per_image = n_rays_per_image
        self._n_pts_per_ray = n_pts_per_ray
        self._min_depth = min_depth
        self._max_depth = max_depth

    def forward(self, cameras: CamerasBase, **kwargs) -> RayBundle:
        """
        Args:
            cameras: A batch of `batch_size` cameras from which the rays are emitted.
        Returns:
            A named tuple RayBundle with the following fields:
            origins: A tensor of shape
                `(batch_size, n_rays_per_image, 3)`
                denoting the locations of ray origins in the world coordinates.
            directions: A tensor of shape
                `(batch_size, n_rays_per_image, 3)`
                denoting the directions of each ray in the world coordinates.
            lengths: A tensor of shape
                `(batch_size, n_rays_per_image, n_pts_per_ray)`
                containing the z-coordinate (=depth) of each ray in world units.
            xys: A tensor of shape
                `(batch_size, n_rays_per_image, 2)`
                containing the 2D image coordinates of each ray.
        """

        batch_size = cameras.R.shape[0]

        device = cameras.device

        # get the initial grid of image xy coords
        # of shape (batch_size, n_rays_per_image, 2)
        rays_xy = torch.cat(
            [
                torch.rand(
                    size=(batch_size, self._n_rays_per_image, 1),
                    dtype=torch.float32,
                    device=device,
                )
                * (high - low)
                + low
                for low, high in (
                    (self._min_x, self._max_x),
                    (self._min_y, self._max_y),
                )
            ],
            dim=2,
        )

        return _xy_to_ray_bundle(
            cameras, rays_xy, self._min_depth, self._max_depth, self._n_pts_per_ray
        )


def _xy_to_ray_bundle(
    cameras: CamerasBase,
    xy_grid: torch.Tensor,
    min_depth: float,
    max_depth: float,
    n_pts_per_ray: int,
) -> RayBundle:
    """
    Extends the `xy_grid` input of shape `(batch_size, ..., 2)` to rays.
    This adds to each xy location in the grid a vector of `n_pts_per_ray` depths
    uniformly spaced between `min_depth` and `max_depth`.

    The extended grid is then unprojected with `cameras` to yield
    ray origins, directions and depths.
    """
    batch_size = xy_grid.shape[0]
    spatial_size = xy_grid.shape[1:-1]
    n_rays_per_image = spatial_size.numel()  # pyre-ignore

    # ray z-coords
    depths = torch.linspace(
        min_depth, max_depth, n_pts_per_ray, dtype=xy_grid.dtype, device=xy_grid.device
    )
    rays_zs = depths[None, None].expand(batch_size, n_rays_per_image, n_pts_per_ray)

    # make two sets of points at a constant depth=1 and 2
    to_unproject = torch.cat(
        (
            xy_grid.view(batch_size, 1, n_rays_per_image, 2)
            .expand(batch_size, 2, n_rays_per_image, 2)
            .reshape(batch_size, n_rays_per_image * 2, 2),
            torch.cat(
                (
                    xy_grid.new_ones(batch_size, n_rays_per_image, 1),
                    2.0 * xy_grid.new_ones(batch_size, n_rays_per_image, 1),
                ),
                dim=1,
            ),
        ),
        dim=-1,
    )

    # unproject the points
    unprojected = cameras.unproject_points(to_unproject, from_ndc=True)  # pyre-ignore

    # split the two planes back
    rays_plane_1_world = unprojected[:, :n_rays_per_image]
    rays_plane_2_world = unprojected[:, n_rays_per_image:]

    # directions are the differences between the two planes of points
    rays_directions_world = rays_plane_2_world - rays_plane_1_world

    # origins are given by subtracting the ray directions from the first plane
    rays_origins_world = rays_plane_1_world - rays_directions_world

    return RayBundle(
        rays_origins_world.view(batch_size, *spatial_size, 3),
        rays_directions_world.view(batch_size, *spatial_size, 3),
        rays_zs.view(batch_size, *spatial_size, n_pts_per_ray),
        xy_grid,
    )
