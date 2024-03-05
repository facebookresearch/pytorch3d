# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import warnings
from typing import Optional, Tuple, Union

import torch
from pytorch3d.common.compat import meshgrid_ij
from pytorch3d.ops import padded_to_packed
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.renderer.implicit.utils import HeterogeneousRayBundle, RayBundle
from torch.nn import functional as F


"""
This file defines three raysampling techniques:
    - MultinomialRaysampler which can be used to sample rays from pixels of an image grid
    - NDCMultinomialRaysampler which can be used to sample rays from pixels of an image grid,
        which follows the pytorch3d convention for image grid coordinates
    - MonteCarloRaysampler which randomly selects real-valued locations in the image plane
        and emits rays from them
"""


class MultinomialRaysampler(torch.nn.Module):
    """
    Samples a fixed number of points along rays which are regularly distributed
    in a batch of rectangular image grids. Points along each ray
    have uniformly-spaced z-coordinates between a predefined
    minimum and maximum depth.

    The raysampler first generates a 3D coordinate grid of the following form::

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

    In order to generate ray points, `MultinomialRaysampler` takes each 3D point of
    the grid (with coordinates `[x, y, depth]`) and unprojects it
    with `cameras.unproject_points([x, y, depth])`, where `cameras` are an
    additional input to the `forward` function.

    Note that this is a generic implementation that can support any image grid
    coordinate convention. For a raysampler which follows the PyTorch3D
    coordinate conventions please refer to `NDCMultinomialRaysampler`.
    As such, `NDCMultinomialRaysampler` is a special case of `MultinomialRaysampler`.

    Attributes:
        min_x: The leftmost x-coordinate of each ray's source pixel's center.
        max_x: The rightmost x-coordinate of each ray's source pixel's center.
        min_y: The topmost y-coordinate of each ray's source pixel's center.
        max_y: The bottommost y-coordinate of each ray's source pixel's center.
    """

    def __init__(
        self,
        *,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        image_width: int,
        image_height: int,
        n_pts_per_ray: int,
        min_depth: float,
        max_depth: float,
        n_rays_per_image: Optional[int] = None,
        n_rays_total: Optional[int] = None,
        unit_directions: bool = False,
        stratified_sampling: bool = False,
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
            n_rays_per_image: If given, this amount of rays are sampled from the grid.
                `n_rays_per_image` and `n_rays_total` cannot both be defined.
            n_rays_total: How many rays in total to sample from the cameras provided. The result
                is as if `n_rays_total_training` cameras were sampled with replacement from the
                cameras provided and for every camera one ray was sampled. If set returns the
                HeterogeneousRayBundle with batch_size=n_rays_total.
                `n_rays_per_image` and `n_rays_total` cannot both be defined.
            unit_directions: whether to normalize direction vectors in ray bundle.
            stratified_sampling: if True, performs stratified random sampling
                along the ray; otherwise takes ray points at deterministic offsets.
        """
        super().__init__()
        self._n_pts_per_ray = n_pts_per_ray
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._n_rays_per_image = n_rays_per_image
        self._n_rays_total = n_rays_total
        self._unit_directions = unit_directions
        self._stratified_sampling = stratified_sampling
        self.min_x, self.max_x = min_x, max_x
        self.min_y, self.max_y = min_y, max_y
        # get the initial grid of image xy coords
        y, x = meshgrid_ij(
            torch.linspace(min_y, max_y, image_height, dtype=torch.float32),
            torch.linspace(min_x, max_x, image_width, dtype=torch.float32),
        )
        _xy_grid = torch.stack([x, y], dim=-1)

        self.register_buffer("_xy_grid", _xy_grid, persistent=False)

    def forward(
        self,
        cameras: CamerasBase,
        *,
        mask: Optional[torch.Tensor] = None,
        min_depth: Optional[float] = None,
        max_depth: Optional[float] = None,
        n_rays_per_image: Optional[int] = None,
        n_pts_per_ray: Optional[int] = None,
        stratified_sampling: Optional[bool] = None,
        n_rays_total: Optional[int] = None,
        **kwargs,
    ) -> Union[RayBundle, HeterogeneousRayBundle]:
        """
        Args:
            cameras: A batch of `batch_size` cameras from which the rays are emitted.
            mask: if given, the rays are sampled from the mask. Should be of size
                (batch_size, image_height, image_width).
            min_depth: The minimum depth of a ray-point.
            max_depth: The maximum depth of a ray-point.
            n_rays_per_image: If given, this amount of rays are sampled from the grid.
                `n_rays_per_image` and `n_rays_total` cannot both be defined.
            n_pts_per_ray: The number of points sampled along each ray.
            stratified_sampling: if set, overrides stratified_sampling provided
                in __init__.
            n_rays_total: How many rays in total to sample from the cameras provided. The result
                is as if `n_rays_total_training` cameras were sampled with replacement from the
                cameras provided and for every camera one ray was sampled. If set returns the
                HeterogeneousRayBundle with batch_size=n_rays_total.
                `n_rays_per_image` and `n_rays_total` cannot both be defined.
        Returns:
            A named tuple RayBundle or dataclass HeterogeneousRayBundle with the
            following fields:

            origins: A tensor of shape
                `(batch_size, s1, s2, 3)`
                denoting the locations of ray origins in the world coordinates.
            directions: A tensor of shape
                `(batch_size, s1, s2, 3)`
                denoting the directions of each ray in the world coordinates.
            lengths: A tensor of shape
                `(batch_size, s1, s2, n_pts_per_ray)`
                containing the z-coordinate (=depth) of each ray in world units.
            xys: A tensor of shape
                `(batch_size, s1, s2, 2)`
                containing the 2D image coordinates of each ray or,
                if mask is given, `(batch_size, n, 1, 2)`
            Here `s1, s2` refer to spatial dimensions.
            `(s1, s2)` refer to (highest priority first):
                - `(1, 1)` if `n_rays_total` is provided, (batch_size=n_rays_total)
                - `(n_rays_per_image, 1) if `n_rays_per_image` if provided,
                - `(n, 1)` where n is the minimum cardinality of the mask
                        in the batch if `mask` is provided
                - `(image_height, image_width)` if nothing from above is satisfied

            `HeterogeneousRayBundle` has additional members:
                - camera_ids: tensor of shape (M,), where `M` is the number of unique sampled
                    cameras. It represents unique ids of sampled cameras.
                - camera_counts: tensor of shape (M,), where `M` is the number of unique sampled
                    cameras. Represents how many times each camera from `camera_ids` was sampled

            `HeterogeneousRayBundle` is returned if `n_rays_total` is provided else `RayBundle`
            is returned.
        """
        n_rays_total = n_rays_total or self._n_rays_total
        n_rays_per_image = n_rays_per_image or self._n_rays_per_image
        if (n_rays_total is not None) and (n_rays_per_image is not None):
            raise ValueError(
                "`n_rays_total` and `n_rays_per_image` cannot both be defined."
            )
        if n_rays_total:
            (
                cameras,
                mask,
                camera_ids,  # unique ids of sampled cameras
                camera_counts,  # number of times unique camera id was sampled
                # `n_rays_per_image` is equal to the max number of times a simgle camera
                # was sampled. We sample all cameras at `camera_ids` `n_rays_per_image` times
                # and then discard the unneeded rays.
                # pyre-ignore[9]
                n_rays_per_image,
            ) = _sample_cameras_and_masks(n_rays_total, cameras, mask)
        else:
            # pyre-ignore[9]
            camera_ids: torch.LongTensor = torch.arange(len(cameras), dtype=torch.long)

        batch_size = cameras.R.shape[0]
        device = cameras.device

        # expand the (H, W, 2) grid batch_size-times to (B, H, W, 2)
        xy_grid = self._xy_grid.to(device).expand(batch_size, -1, -1, -1)

        if mask is not None and n_rays_per_image is None:
            # if num rays not given, sample according to the smallest mask
            n_rays_per_image = (
                n_rays_per_image or mask.sum(dim=(1, 2)).min().int().item()
            )

        if n_rays_per_image is not None:
            if mask is not None:
                assert mask.shape == xy_grid.shape[:3]
                weights = mask.reshape(batch_size, -1)
            else:
                # it is probably more efficient to use torch.randperm
                # for uniform weights but it is unlikely given that randperm
                # is not batched and does not support partial permutation
                _, width, height, _ = xy_grid.shape
                weights = xy_grid.new_ones(batch_size, width * height)
            # pyre-fixme[6]: For 2nd param expected `int` but got `Union[bool,
            #  float, int]`.
            rays_idx = _safe_multinomial(weights, n_rays_per_image)[..., None].expand(
                -1, -1, 2
            )

            xy_grid = torch.gather(xy_grid.reshape(batch_size, -1, 2), 1, rays_idx)[
                :, :, None
            ]

        min_depth = min_depth if min_depth is not None else self._min_depth
        max_depth = max_depth if max_depth is not None else self._max_depth
        n_pts_per_ray = (
            n_pts_per_ray if n_pts_per_ray is not None else self._n_pts_per_ray
        )
        stratified_sampling = (
            stratified_sampling
            if stratified_sampling is not None
            else self._stratified_sampling
        )

        ray_bundle = _xy_to_ray_bundle(
            cameras,
            xy_grid,
            min_depth,
            max_depth,
            n_pts_per_ray,
            self._unit_directions,
            stratified_sampling,
        )

        return (
            # pyre-ignore[61]
            _pack_ray_bundle(ray_bundle, camera_ids, camera_counts)
            if n_rays_total
            else ray_bundle
        )


class NDCMultinomialRaysampler(MultinomialRaysampler):
    """
    Samples a fixed number of points along rays which are regularly distributed
    in a batch of rectangular image grids. Points along each ray
    have uniformly-spaced z-coordinates between a predefined minimum and maximum depth.

    `NDCMultinomialRaysampler` follows the screen conventions of the `Meshes` and `Pointclouds`
    renderers. I.e. the pixel coordinates are in [-1, 1]x[-u, u] or [-u, u]x[-1, 1]
    where u > 1 is the aspect ratio of the image.

    For the description of arguments, see the documentation to MultinomialRaysampler.
    """

    def __init__(
        self,
        *,
        image_width: int,
        image_height: int,
        n_pts_per_ray: int,
        min_depth: float,
        max_depth: float,
        n_rays_per_image: Optional[int] = None,
        n_rays_total: Optional[int] = None,
        unit_directions: bool = False,
        stratified_sampling: bool = False,
    ) -> None:
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
            n_rays_per_image=n_rays_per_image,
            n_rays_total=n_rays_total,
            unit_directions=unit_directions,
            stratified_sampling=stratified_sampling,
        )


class MonteCarloRaysampler(torch.nn.Module):
    """
    Samples a fixed number of pixels within denoted xy bounds uniformly at random.
    For each pixel, a fixed number of points is sampled along its ray at uniformly-spaced
    z-coordinates such that the z-coordinates range between a predefined minimum
    and maximum depth.

    For practical purposes, this is similar to MultinomialRaysampler without a mask,
    however sampling at real-valued locations bypassing replacement checks may be faster.
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
        *,
        n_rays_total: Optional[int] = None,
        unit_directions: bool = False,
        stratified_sampling: bool = False,
    ) -> None:
        """
        Args:
            min_x: The smallest x-coordinate of each ray's source pixel.
            max_x: The largest x-coordinate of each ray's source pixel.
            min_y: The smallest y-coordinate of each ray's source pixel.
            max_y: The largest y-coordinate of each ray's source pixel.
            n_rays_per_image: The number of rays randomly sampled in each camera.
                `n_rays_per_image` and `n_rays_total` cannot both be defined.
            n_pts_per_ray: The number of points sampled along each ray.
            min_depth: The minimum depth of each ray-point.
            max_depth: The maximum depth of each ray-point.
            n_rays_total: How many rays in total to sample from the cameras provided. The result
                is as if `n_rays_total_training` cameras were sampled with replacement from the
                cameras provided and for every camera one ray was sampled. If set returns the
                HeterogeneousRayBundle with batch_size=n_rays_total.
                `n_rays_per_image` and `n_rays_total` cannot both be defined.
            unit_directions: whether to normalize direction vectors in ray bundle.
            stratified_sampling: if True, performs stratified sampling in n_pts_per_ray
                bins for each ray; otherwise takes n_pts_per_ray deterministic points
                on each ray with uniform offsets.
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
        self._n_rays_total = n_rays_total
        self._unit_directions = unit_directions
        self._stratified_sampling = stratified_sampling

    def forward(
        self,
        cameras: CamerasBase,
        *,
        stratified_sampling: Optional[bool] = None,
        **kwargs,
    ) -> Union[RayBundle, HeterogeneousRayBundle]:
        """
        Args:
            cameras: A batch of `batch_size` cameras from which the rays are emitted.
            stratified_sampling: if set, overrides stratified_sampling provided
                in __init__.
        Returns:
            A named tuple `RayBundle` or dataclass `HeterogeneousRayBundle` with the
            following fields:

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
            If `n_rays_total` is provided `batch_size=n_rays_total`and
            `n_rays_per_image=1` and `HeterogeneousRayBundle` is returned else `RayBundle`
            is returned.

            `HeterogeneousRayBundle` has additional members:
                - camera_ids: tensor of shape (M,), where `M` is the number of unique sampled
                    cameras. It represents unique ids of sampled cameras.
                - camera_counts: tensor of shape (M,), where `M` is the number of unique sampled
                    cameras. Represents how many times each camera from `camera_ids` was sampled
        """
        if (
            sum(x is not None for x in [self._n_rays_total, self._n_rays_per_image])
            != 1
        ):
            raise ValueError(
                "Exactly one of `self.n_rays_total` and `self.n_rays_per_image` "
                "must be given."
            )

        if self._n_rays_total:
            (
                cameras,
                _,
                camera_ids,
                camera_counts,
                n_rays_per_image,
            ) = _sample_cameras_and_masks(self._n_rays_total, cameras, None)
        else:
            # pyre-ignore[9]
            camera_ids: torch.LongTensor = torch.arange(len(cameras), dtype=torch.long)
            n_rays_per_image = self._n_rays_per_image

        batch_size = cameras.R.shape[0]

        device = cameras.device

        # get the initial grid of image xy coords
        # of shape (batch_size, n_rays_per_image, 2)
        rays_xy = torch.cat(
            [
                torch.rand(
                    size=(batch_size, n_rays_per_image, 1),
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

        stratified_sampling = (
            stratified_sampling
            if stratified_sampling is not None
            else self._stratified_sampling
        )

        ray_bundle = _xy_to_ray_bundle(
            cameras,
            rays_xy,
            self._min_depth,
            self._max_depth,
            self._n_pts_per_ray,
            self._unit_directions,
            stratified_sampling,
        )

        return (
            # pyre-ignore[61]
            _pack_ray_bundle(ray_bundle, camera_ids, camera_counts)
            if self._n_rays_total
            else ray_bundle
        )


# Settings for backwards compatibility
def GridRaysampler(
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    image_width: int,
    image_height: int,
    n_pts_per_ray: int,
    min_depth: float,
    max_depth: float,
) -> "MultinomialRaysampler":
    """
    GridRaysampler has been DEPRECATED. Use MultinomialRaysampler instead.
    Preserving GridRaysampler for backward compatibility.
    """

    warnings.warn(
        """GridRaysampler is deprecated,
        Use MultinomialRaysampler instead.
        GridRaysampler will be removed in future releases.""",
        PendingDeprecationWarning,
    )

    return MultinomialRaysampler(
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        image_width=image_width,
        image_height=image_height,
        n_pts_per_ray=n_pts_per_ray,
        min_depth=min_depth,
        max_depth=max_depth,
    )


# Settings for backwards compatibility
def NDCGridRaysampler(
    image_width: int,
    image_height: int,
    n_pts_per_ray: int,
    min_depth: float,
    max_depth: float,
) -> "NDCMultinomialRaysampler":
    """
    NDCGridRaysampler has been DEPRECATED. Use NDCMultinomialRaysampler instead.
    Preserving NDCGridRaysampler for backward compatibility.
    """

    warnings.warn(
        """NDCGridRaysampler is deprecated,
        Use NDCMultinomialRaysampler instead.
        NDCGridRaysampler will be removed in future releases.""",
        PendingDeprecationWarning,
    )

    return NDCMultinomialRaysampler(
        image_width=image_width,
        image_height=image_height,
        n_pts_per_ray=n_pts_per_ray,
        min_depth=min_depth,
        max_depth=max_depth,
    )


def _safe_multinomial(input: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Wrapper around torch.multinomial that attempts sampling without replacement
    when possible, otherwise resorts to sampling with replacement.

    Args:
        input: tensor of shape [B, n] containing non-negative values;
                rows are interpreted as unnormalized event probabilities
                in categorical distributions.
        num_samples: number of samples to take.

    Returns:
        LongTensor of shape [B, num_samples] containing
        values from {0, ..., n - 1} where the elements [i, :] of row i make
            (1) if there are num_samples or more non-zero values in input[i],
                a random subset of the indices of those values, with
                probabilities proportional to the values in input[i, :].

            (2) if not, a random sample with replacement of the indices of
                those values, with probabilities proportional to them.
                This sample might not contain all the indices of the
                non-zero values.
        Behavior undetermined if there are no non-zero values in a whole row
        or if there are negative values.
    """
    try:
        res = torch.multinomial(input, num_samples, replacement=False)
    except RuntimeError:
        # this is probably rare, so we don't mind sampling twice
        res = torch.multinomial(input, num_samples, replacement=True)
        no_repl = (input > 0.0).sum(dim=-1) >= num_samples
        res[no_repl] = torch.multinomial(input[no_repl], num_samples, replacement=False)
        return res

    # in some versions of Pytorch, zero probabilty samples can be drawn without an error
    # due to this bug: https://github.com/pytorch/pytorch/issues/50034. Handle this case:
    repl = (input > 0.0).sum(dim=-1) < num_samples
    if repl.any():
        res[repl] = torch.multinomial(input[repl], num_samples, replacement=True)

    return res


def _xy_to_ray_bundle(
    cameras: CamerasBase,
    xy_grid: torch.Tensor,
    min_depth: float,
    max_depth: float,
    n_pts_per_ray: int,
    unit_directions: bool,
    stratified_sampling: bool = False,
) -> RayBundle:
    """
    Extends the `xy_grid` input of shape `(batch_size, ..., 2)` to rays.
    This adds to each xy location in the grid a vector of `n_pts_per_ray` depths
    uniformly spaced between `min_depth` and `max_depth`.

    The extended grid is then unprojected with `cameras` to yield
    ray origins, directions and depths.

    Args:
        cameras: cameras object representing a batch of cameras.
        xy_grid: torch.tensor grid of image xy coords.
        min_depth: The minimum depth of each ray-point.
        max_depth: The maximum depth of each ray-point.
        n_pts_per_ray: The number of points sampled along each ray.
        unit_directions: whether to normalize direction vectors in ray bundle.
        stratified_sampling: if True, performs stratified sampling in n_pts_per_ray
            bins for each ray; otherwise takes n_pts_per_ray deterministic points
            on each ray with uniform offsets.
    """
    batch_size = xy_grid.shape[0]
    spatial_size = xy_grid.shape[1:-1]
    n_rays_per_image = spatial_size.numel()

    # ray z-coords
    rays_zs = xy_grid.new_empty((0,))
    if n_pts_per_ray > 0:
        depths = torch.linspace(
            min_depth,
            max_depth,
            n_pts_per_ray,
            dtype=xy_grid.dtype,
            device=xy_grid.device,
        )
        rays_zs = depths[None, None].expand(batch_size, n_rays_per_image, n_pts_per_ray)

        if stratified_sampling:
            rays_zs = _jiggle_within_stratas(rays_zs)

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
    unprojected = cameras.unproject_points(to_unproject, from_ndc=True)

    # split the two planes back
    rays_plane_1_world = unprojected[:, :n_rays_per_image]
    rays_plane_2_world = unprojected[:, n_rays_per_image:]

    # directions are the differences between the two planes of points
    rays_directions_world = rays_plane_2_world - rays_plane_1_world

    # origins are given by subtracting the ray directions from the first plane
    rays_origins_world = rays_plane_1_world - rays_directions_world

    if unit_directions:
        rays_directions_world = F.normalize(rays_directions_world, dim=-1)

    return RayBundle(
        rays_origins_world.view(batch_size, *spatial_size, 3),
        rays_directions_world.view(batch_size, *spatial_size, 3),
        rays_zs.view(batch_size, *spatial_size, n_pts_per_ray),
        xy_grid,
    )


def _jiggle_within_stratas(bin_centers: torch.Tensor) -> torch.Tensor:
    """
    Performs sampling of 1 point per bin given the bin centers.

    More specifically, it replaces each point's value `z`
    with a sample from a uniform random distribution on
    `[z - delta_-, z + delta_+]`, where `delta_-` is half of the difference
    between `z` and the previous point, and `delta_+` is half of the difference
    between the next point and `z`. For the first and last items, the
    corresponding boundary deltas are assumed zero.

    Args:
        `bin_centers`: The input points of size (..., N); the result is broadcast
            along all but the last dimension (the rows). Each row should be
            sorted in ascending order.

    Returns:
        a tensor of size (..., N) with the locations jiggled within stratas/bins.
    """
    # Get intervals between bin centers.
    mids = 0.5 * (bin_centers[..., 1:] + bin_centers[..., :-1])
    upper = torch.cat((mids, bin_centers[..., -1:]), dim=-1)
    lower = torch.cat((bin_centers[..., :1], mids), dim=-1)
    # Samples in those intervals.
    jiggled = lower + (upper - lower) * torch.rand_like(lower)
    return jiggled


def _sample_cameras_and_masks(
    n_samples: int, cameras: CamerasBase, mask: Optional[torch.Tensor] = None
) -> Tuple[
    CamerasBase,
    Optional[torch.Tensor],
    torch.LongTensor,
    torch.LongTensor,
    torch.LongTensor,
]:
    """
    Samples n_rays_total cameras and masks and returns them in a form
    (camera_idx, count), where count represents number of times the same camera
    has been sampled.

    Args:
        n_samples: how many camera and mask pairs to sample
        cameras: A batch of `batch_size` cameras from which the rays are emitted.
        mask: Optional. Should be of size (batch_size, image_height, image_width).
    Returns:
        tuple of a form (sampled_cameras, sampled_masks, unique_sampled_camera_ids,
            number_of_times_each_sampled_camera_has_been_sampled,
            max_number_of_times_camera_has_been_sampled,
            )
    """
    sampled_ids = torch.randint(
        0,
        len(cameras),
        size=(n_samples,),
        dtype=torch.long,
    )
    unique_ids, counts = torch.unique(sampled_ids, return_counts=True)
    # pyre-ignore[7]
    return (
        cameras[unique_ids],
        mask[unique_ids] if mask is not None else None,
        unique_ids,
        counts,
        torch.max(counts),
    )


# TODO: this function can be unified with ImplicitronRayBundle.get_padded_xys
def _pack_ray_bundle(
    ray_bundle: RayBundle, camera_ids: torch.LongTensor, camera_counts: torch.LongTensor
) -> HeterogeneousRayBundle:
    """
    Pack the raybundle from [n_cameras, max(rays_per_camera), ...] to
        [total_num_rays, 1, ...]

    Args:
        ray_bundle: A ray_bundle to pack
        camera_ids: Unique ids of cameras that were sampled
        camera_counts: how many of which camera to pack, each count coresponds to
            one 'row' of the ray_bundle and says how many rays wll be taken
            from it and packed.
    Returns:
        HeterogeneousRayBundle where batch_size=sum(camera_counts) and n_rays_per_image=1
    """
    # pyre-ignore[9]
    camera_counts = camera_counts.to(ray_bundle.origins.device)
    cumsum = torch.cumsum(camera_counts, dim=0, dtype=torch.long)
    # pyre-ignore[9]
    first_idxs: torch.LongTensor = torch.cat(
        (camera_counts.new_zeros((1,), dtype=torch.long), cumsum[:-1])
    )
    num_inputs = int(camera_counts.sum())

    return HeterogeneousRayBundle(
        origins=padded_to_packed(ray_bundle.origins, first_idxs, num_inputs)[:, None],
        directions=padded_to_packed(ray_bundle.directions, first_idxs, num_inputs)[
            :, None
        ],
        lengths=padded_to_packed(ray_bundle.lengths, first_idxs, num_inputs)[:, None],
        xys=padded_to_packed(ray_bundle.xys, first_idxs, num_inputs)[:, None],
        camera_ids=camera_ids,
        camera_counts=camera_counts,
    )
