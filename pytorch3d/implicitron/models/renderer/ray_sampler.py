# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
from pytorch3d.implicitron.tools import camera_utils
from pytorch3d.implicitron.tools.config import registry, ReplaceableBase
from pytorch3d.renderer import NDCMultinomialRaysampler
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.renderer.implicit.utils import HeterogeneousRayBundle

from .base import EvaluationMode, ImplicitronRayBundle, RenderSamplingMode


class RaySamplerBase(ReplaceableBase):
    """
    Base class for ray samplers.
    """

    def forward(
        self,
        cameras: CamerasBase,
        evaluation_mode: EvaluationMode,
        mask: Optional[torch.Tensor] = None,
    ) -> ImplicitronRayBundle:
        """
        Args:
            cameras: A batch of `batch_size` cameras from which the rays are emitted.
            evaluation_mode: one of `EvaluationMode.TRAINING` or
                `EvaluationMode.EVALUATION` which determines the sampling mode
                that is used.
            mask: Active for the `RenderSamplingMode.MASK_SAMPLE` sampling mode.
                Defines a non-negative mask of shape
                `(batch_size, image_height, image_width)` where each per-pixel
                value is proportional to the probability of sampling the
                corresponding pixel's ray.

        Returns:
            ray_bundle: A `ImplicitronRayBundle` object containing the parametrizations of the
                sampled rendering rays.
        """
        raise NotImplementedError()


class AbstractMaskRaySampler(RaySamplerBase, torch.nn.Module):
    """
    Samples a fixed number of points along rays which are in turn sampled for
    each camera in a batch.

    This class utilizes `NDCMultinomialRaysampler` which allows to either
    randomly sample rays from an input foreground saliency mask
    (`RenderSamplingMode.MASK_SAMPLE`), or on a rectangular image grid
    (`RenderSamplingMode.FULL_GRID`). The sampling mode can be set separately
    for training and evaluation by setting `self.sampling_mode_training`
    and `self.sampling_mode_training` accordingly.

    The class allows to adjust the sampling points along rays by overwriting the
    `AbstractMaskRaySampler._get_min_max_depth_bounds` function which returns
    the near/far planes (`min_depth`/`max_depth`) `NDCMultinomialRaysampler`.

    Settings:
        image_width: The horizontal size of the image grid.
        image_height: The vertical size of the image grid.
        sampling_mode_training: The ray sampling mode for training. This should be a str
            option from the RenderSamplingMode Enum
        sampling_mode_evaluation: Same as above but for evaluation.
        n_pts_per_ray_training: The number of points sampled along each ray during training.
        n_pts_per_ray_evaluation: The number of points sampled along each ray during evaluation.
        n_rays_per_image_sampled_from_mask: The amount of rays to be sampled from the image
            grid. Given a batch of image grids, this many is sampled from each.
            `n_rays_per_image_sampled_from_mask` and `n_rays_total_training` cannot both be
            defined.
        n_rays_total_training: (optional) How many rays in total to sample from the entire
            batch of provided image grid. The result is as if `n_rays_total_training`
            cameras/image grids were sampled with replacement from the cameras / image grids
            provided and for every camera one ray was sampled.
            `n_rays_per_image_sampled_from_mask` and `n_rays_total_training` cannot both be
            defined, to use you have to set `n_rays_per_image` to None.
            Used only for EvaluationMode.TRAINING.
        stratified_point_sampling_training: if set, performs stratified random sampling
            along the ray; otherwise takes ray points at deterministic offsets.
        stratified_point_sampling_evaluation: Same as above but for evaluation.
        cast_ray_bundle_as_cone: If True, the sampling will generate the bins and radii
            attribute of ImplicitronRayBundle. The `bins` contain the z-coordinate
            (=depth) of each ray in world units and are of shape
            `(batch_size, n_rays_per_image, n_pts_per_ray_training/evaluation + 1)`
            while `lengths` is equal to the midpoint of the bins:
            (0.5 * (bins[..., 1:] + bins[..., :-1]).
            If False, `bins` is None, `radii` is None and `lengths` contains
            the z-coordinate (=depth) of each ray in world units and are of shape
            `(batch_size, n_rays_per_image, n_pts_per_ray_training/evaluation)`

    Raises:
        TypeError: if cast_ray_bundle_as_cone is set to True and n_rays_total_training
            is not None will result in an error. HeterogeneousRayBundle is
            not supported for conical frustum computation yet.
    """

    image_width: int = 400
    image_height: int = 400
    sampling_mode_training: str = "mask_sample"
    sampling_mode_evaluation: str = "full_grid"
    n_pts_per_ray_training: int = 64
    n_pts_per_ray_evaluation: int = 64
    n_rays_per_image_sampled_from_mask: Optional[int] = 1024
    n_rays_total_training: Optional[int] = None
    # stratified sampling vs taking points at deterministic offsets
    stratified_point_sampling_training: bool = True
    stratified_point_sampling_evaluation: bool = False
    cast_ray_bundle_as_cone: bool = False

    def __post_init__(self):
        if (self.n_rays_per_image_sampled_from_mask is not None) and (
            self.n_rays_total_training is not None
        ):
            raise ValueError(
                "Cannot both define n_rays_total_training and "
                "n_rays_per_image_sampled_from_mask."
            )

        self._sampling_mode = {
            EvaluationMode.TRAINING: RenderSamplingMode(self.sampling_mode_training),
            EvaluationMode.EVALUATION: RenderSamplingMode(
                self.sampling_mode_evaluation
            ),
        }

        n_pts_per_ray_training = (
            self.n_pts_per_ray_training + 1
            if self.cast_ray_bundle_as_cone
            else self.n_pts_per_ray_training
        )
        n_pts_per_ray_evaluation = (
            self.n_pts_per_ray_evaluation + 1
            if self.cast_ray_bundle_as_cone
            else self.n_pts_per_ray_evaluation
        )
        self._training_raysampler = NDCMultinomialRaysampler(
            image_width=self.image_width,
            image_height=self.image_height,
            n_pts_per_ray=n_pts_per_ray_training,
            min_depth=0.0,
            max_depth=0.0,
            n_rays_per_image=self.n_rays_per_image_sampled_from_mask
            if self._sampling_mode[EvaluationMode.TRAINING]
            == RenderSamplingMode.MASK_SAMPLE
            else None,
            n_rays_total=self.n_rays_total_training,
            unit_directions=True,
            stratified_sampling=self.stratified_point_sampling_training,
        )

        self._evaluation_raysampler = NDCMultinomialRaysampler(
            image_width=self.image_width,
            image_height=self.image_height,
            n_pts_per_ray=n_pts_per_ray_evaluation,
            min_depth=0.0,
            max_depth=0.0,
            n_rays_per_image=self.n_rays_per_image_sampled_from_mask
            if self._sampling_mode[EvaluationMode.EVALUATION]
            == RenderSamplingMode.MASK_SAMPLE
            else None,
            unit_directions=True,
            stratified_sampling=self.stratified_point_sampling_evaluation,
        )

        max_y, min_y = self._training_raysampler.max_y, self._training_raysampler.min_y
        max_x, min_x = self._training_raysampler.max_x, self._training_raysampler.min_x
        self.pixel_height: float = (max_y - min_y) / (self.image_height - 1)
        self.pixel_width: float = (max_x - min_x) / (self.image_width - 1)

    def _get_min_max_depth_bounds(self, cameras: CamerasBase) -> Tuple[float, float]:
        raise NotImplementedError()

    def forward(
        self,
        cameras: CamerasBase,
        evaluation_mode: EvaluationMode,
        mask: Optional[torch.Tensor] = None,
    ) -> ImplicitronRayBundle:
        """

        Args:
            cameras: A batch of `batch_size` cameras from which the rays are emitted.
            evaluation_mode: one of `EvaluationMode.TRAINING` or
                `EvaluationMode.EVALUATION` which determines the sampling mode
                that is used.
            mask: Active for the `RenderSamplingMode.MASK_SAMPLE` sampling mode.
                Defines a non-negative mask of shape
                `(batch_size, image_height, image_width)` where each per-pixel
                value is proportional to the probability of sampling the
                corresponding pixel's ray.

        Returns:
            ray_bundle: A `ImplicitronRayBundle` object containing the parametrizations of the
                sampled rendering rays.
        """
        sample_mask = None
        if (
            self._sampling_mode[evaluation_mode] == RenderSamplingMode.MASK_SAMPLE
            and mask is not None
        ):
            sample_mask = torch.nn.functional.interpolate(
                mask,
                size=[self.image_height, self.image_width],
                mode="nearest",
            )[:, 0]

        min_depth, max_depth = self._get_min_max_depth_bounds(cameras)

        raysampler = {
            EvaluationMode.TRAINING: self._training_raysampler,
            EvaluationMode.EVALUATION: self._evaluation_raysampler,
        }[evaluation_mode]

        ray_bundle = raysampler(
            cameras=cameras,
            mask=sample_mask,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        if self.cast_ray_bundle_as_cone and isinstance(
            ray_bundle, HeterogeneousRayBundle
        ):
            # If this error rises it means that raysampler has among
            # its arguments `n_ray_totals`. If it is the case
            # then you should update the radii computation and lengths
            # computation to handle padding and unpadding.
            raise TypeError(
                "Heterogeneous ray bundle is not supported for conical frustum computation yet"
            )
        elif self.cast_ray_bundle_as_cone:
            pixel_hw: Tuple[float, float] = (self.pixel_height, self.pixel_width)
            pixel_radii_2d = compute_radii(cameras, ray_bundle.xys[..., :2], pixel_hw)
            return ImplicitronRayBundle(
                directions=ray_bundle.directions,
                origins=ray_bundle.origins,
                lengths=None,
                xys=ray_bundle.xys,
                bins=ray_bundle.lengths,
                pixel_radii_2d=pixel_radii_2d,
            )

        return ImplicitronRayBundle(
            directions=ray_bundle.directions,
            origins=ray_bundle.origins,
            lengths=ray_bundle.lengths,
            xys=ray_bundle.xys,
            camera_counts=getattr(ray_bundle, "camera_counts", None),
            camera_ids=getattr(ray_bundle, "camera_ids", None),
        )


@registry.register
class AdaptiveRaySampler(AbstractMaskRaySampler):
    """
    Adaptively samples points on each ray between near and far planes whose
    depths are determined based on the distance from the camera center
    to a predefined scene center.

    More specifically,
    `min_depth = max(
        (self.scene_center-camera_center).norm() - self.scene_extent, eps
    )` and
    `max_depth = (self.scene_center-camera_center).norm() + self.scene_extent`.

    This sampling is ideal for object-centric scenes whose contents are
    centered around a known `self.scene_center` and fit into a bounding sphere
    with a radius of `self.scene_extent`.

    Args:
        scene_center: The xyz coordinates of the center of the scene used
            along with `scene_extent` to compute the min and max depth planes
            for sampling ray-points.
        scene_extent: The radius of the scene bounding box centered at `scene_center`.
    """

    scene_extent: float = 8.0
    scene_center: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        super().__post_init__()
        if self.scene_extent <= 0.0:
            raise ValueError("Adaptive raysampler requires self.scene_extent > 0.")
        self._scene_center = torch.FloatTensor(self.scene_center)

    def _get_min_max_depth_bounds(self, cameras: CamerasBase) -> Tuple[float, float]:
        """
        Returns the adaptively calculated near/far planes.
        """
        min_depth, max_depth = camera_utils.get_min_max_depth_bounds(
            cameras, self._scene_center, self.scene_extent
        )
        return float(min_depth[0]), float(max_depth[0])


@registry.register
class NearFarRaySampler(AbstractMaskRaySampler):
    """
    Samples a fixed number of points between fixed near and far z-planes.
    Specifically, samples points along each ray with approximately uniform spacing
    of z-coordinates between the minimum depth `self.min_depth` and the maximum depth
    `self.max_depth`. This sampling is useful for rendering scenes where the camera is
    in a constant distance from the focal point of the scene.

    Args:
        min_depth: The minimum depth of a ray-point.
        max_depth: The maximum depth of a ray-point.
    """

    min_depth: float = 0.1
    max_depth: float = 8.0

    def _get_min_max_depth_bounds(self, cameras: CamerasBase) -> Tuple[float, float]:
        """
        Returns the stored near/far planes.
        """
        return self.min_depth, self.max_depth


def compute_radii(
    cameras: CamerasBase,
    xy_grid: torch.Tensor,
    pixel_hw_ndc: Tuple[float, float],
) -> torch.Tensor:
    """
    Compute radii of conical frustums in world coordinates.

    Args:
        cameras: cameras object representing a batch of cameras.
        xy_grid: torch.tensor grid of image xy coords.
        pixel_hw_ndc: pixel height and width in NDC

    Returns:
        radii: A tensor of shape `(..., 1)` radii of a cone.
    """
    batch_size = xy_grid.shape[0]
    spatial_size = xy_grid.shape[1:-1]
    n_rays_per_image = spatial_size.numel()

    xy = xy_grid.view(batch_size, n_rays_per_image, 2)

    # [batch_size, 3 * n_rays_per_image, 2]
    xy = torch.cat(
        [
            xy,
            # Will allow to find the norm on the x axis
            xy + torch.tensor([pixel_hw_ndc[1], 0], device=xy.device),
            # Will allow to find the norm on the y axis
            xy + torch.tensor([0, pixel_hw_ndc[0]], device=xy.device),
        ],
        dim=1,
    )
    # [batch_size, 3 * n_rays_per_image, 3]
    xyz = torch.cat(
        (
            xy,
            xy.new_ones(batch_size, 3 * n_rays_per_image, 1),
        ),
        dim=-1,
    )

    # unproject the points
    unprojected_xyz = cameras.unproject_points(xyz, from_ndc=True)

    plane_world, plane_world_dx, plane_world_dy = torch.split(
        unprojected_xyz, n_rays_per_image, dim=1
    )

    # Distance from each unit-norm direction vector to its neighbors.
    dx_norm = torch.linalg.norm(plane_world_dx - plane_world, dim=-1, keepdims=True)
    dy_norm = torch.linalg.norm(plane_world_dy - plane_world, dim=-1, keepdims=True)
    # Cut the distance in half to obtain the base radius: (dx_norm + dy_norm) * 0.5
    # Scale it by 2/12**0.5 to match the variance of the pixel’s footprint
    radii = (dx_norm + dy_norm) / 12**0.5

    return radii.view(batch_size, *spatial_size, 1)
