# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import field
from typing import Optional, Tuple

import torch
from pytorch3d.implicitron.tools import camera_utils
from pytorch3d.implicitron.tools.config import Configurable
from pytorch3d.renderer import NDCMultinomialRaysampler, RayBundle
from pytorch3d.renderer.cameras import CamerasBase

from .base import EvaluationMode, RenderSamplingMode


class RaySampler(Configurable, torch.nn.Module):
    """

    Samples a fixed number of points along rays which are in turn sampled for
    each camera in a batch.

    This class utilizes `NDCMultinomialRaysampler` which allows to either
    randomly sample rays from an input foreground saliency mask
    (`RenderSamplingMode.MASK_SAMPLE`), or on a rectangular image grid
    (`RenderSamplingMode.FULL_GRID`). The sampling mode can be set separately
    for training and evaluation by setting `self.sampling_mode_training`
    and `self.sampling_mode_training` accordingly.

    The class allows two modes of sampling points along the rays:
        1) Sampling between fixed near and far z-planes:
            Active when `self.scene_extent <= 0`, samples points along each ray
            with approximately uniform spacing of z-coordinates between
            the minimum depth `self.min_depth` and the maximum depth `self.max_depth`.
            This sampling is useful for rendering scenes where the camera is
            in a constant distance from the focal point of the scene.
        2) Adaptive near/far plane estimation around the world scene center:
            Active when `self.scene_extent > 0`. Samples points on each
            ray between near and far planes whose depths are determined based on
            the distance from the camera center to a predefined scene center.
            More specifically,
            `min_depth = max(
                (self.scene_center-camera_center).norm() - self.scene_extent, eps
            )` and
            `max_depth = (self.scene_center-camera_center).norm() + self.scene_extent`.
            This sampling is ideal for object-centric scenes whose contents are
            centered around a known `self.scene_center` and fit into a bounding sphere
            with a radius of `self.scene_extent`.

        Similar to the sampling mode, the sampling parameters can be set separately
        for training and evaluation.

    Settings:
        image_width: The horizontal size of the image grid.
        image_height: The vertical size of the image grid.
        scene_center: The xyz coordinates of the center of the scene used
            along with `scene_extent` to compute the min and max depth planes
            for sampling ray-points.
        scene_extent: The radius of the scene bounding sphere centered at `scene_center`.
            If `scene_extent <= 0`, the raysampler samples points between
            `self.min_depth` and `self.max_depth` depths instead.
        sampling_mode_training: The ray sampling mode for training. This should be a str
            option from the RenderSamplingMode Enum
        sampling_mode_evaluation: Same as above but for evaluation.
        n_pts_per_ray_training: The number of points sampled along each ray during training.
        n_pts_per_ray_evaluation: The number of points sampled along each ray during evaluation.
        n_rays_per_image_sampled_from_mask: The amount of rays to be sampled from the image grid
        min_depth: The minimum depth of a ray-point. Active when `self.scene_extent > 0`.
        max_depth: The maximum depth of a ray-point. Active when `self.scene_extent > 0`.
        stratified_point_sampling_training: if set, performs stratified random sampling
            along the ray; otherwise takes ray points at deterministic offsets.
        stratified_point_sampling_evaluation: Same as above but for evaluation.

    """

    image_width: int = 400
    image_height: int = 400
    scene_center: Tuple[float, float, float] = field(
        default_factory=lambda: (0.0, 0.0, 0.0)
    )
    scene_extent: float = 0.0
    sampling_mode_training: str = "mask_sample"
    sampling_mode_evaluation: str = "full_grid"
    n_pts_per_ray_training: int = 64
    n_pts_per_ray_evaluation: int = 64
    n_rays_per_image_sampled_from_mask: int = 1024
    min_depth: float = 0.1
    max_depth: float = 8.0
    # stratified sampling vs taking points at deterministic offsets
    stratified_point_sampling_training: bool = True
    stratified_point_sampling_evaluation: bool = False

    def __post_init__(self):
        super().__init__()
        self.scene_center = torch.FloatTensor(self.scene_center)

        self._sampling_mode = {
            EvaluationMode.TRAINING: RenderSamplingMode(self.sampling_mode_training),
            EvaluationMode.EVALUATION: RenderSamplingMode(
                self.sampling_mode_evaluation
            ),
        }

        self._raysamplers = {
            EvaluationMode.TRAINING: NDCMultinomialRaysampler(
                image_width=self.image_width,
                image_height=self.image_height,
                n_pts_per_ray=self.n_pts_per_ray_training,
                min_depth=self.min_depth,
                max_depth=self.max_depth,
                n_rays_per_image=self.n_rays_per_image_sampled_from_mask
                if self._sampling_mode[EvaluationMode.TRAINING]
                == RenderSamplingMode.MASK_SAMPLE
                else None,
                unit_directions=True,
                stratified_sampling=self.stratified_point_sampling_training,
            ),
            EvaluationMode.EVALUATION: NDCMultinomialRaysampler(
                image_width=self.image_width,
                image_height=self.image_height,
                n_pts_per_ray=self.n_pts_per_ray_evaluation,
                min_depth=self.min_depth,
                max_depth=self.max_depth,
                n_rays_per_image=self.n_rays_per_image_sampled_from_mask
                if self._sampling_mode[EvaluationMode.EVALUATION]
                == RenderSamplingMode.MASK_SAMPLE
                else None,
                unit_directions=True,
                stratified_sampling=self.stratified_point_sampling_evaluation,
            ),
        }

    def forward(
        self,
        cameras: CamerasBase,
        evaluation_mode: EvaluationMode,
        mask: Optional[torch.Tensor] = None,
    ) -> RayBundle:
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
            ray_bundle: A `RayBundle` object containing the parametrizations of the
                sampled rendering rays.
        """
        sample_mask = None
        if (
            # pyre-fixme[29]
            self._sampling_mode[evaluation_mode] == RenderSamplingMode.MASK_SAMPLE
            and mask is not None
        ):
            sample_mask = torch.nn.functional.interpolate(
                mask,
                # pyre-fixme[6]: Expected `Optional[int]` for 2nd param but got
                #  `List[int]`.
                size=[self.image_height, self.image_width],
                mode="nearest",
            )[:, 0]

        if self.scene_extent > 0.0:
            # Override the min/max depth set in initialization based on the
            # input cameras.
            min_depth, max_depth = camera_utils.get_min_max_depth_bounds(
                cameras, self.scene_center, self.scene_extent
            )

        # pyre-fixme[29]:
        #  `Union[BoundMethod[typing.Callable(torch.Tensor.__getitem__)[[Named(self,
        #  torch.Tensor), Named(item, typing.Any)], typing.Any], torch.Tensor],
        #  torch.Tensor, torch.nn.Module]` is not a function.
        ray_bundle = self._raysamplers[evaluation_mode](
            cameras=cameras,
            mask=sample_mask,
            min_depth=float(min_depth[0]) if self.scene_extent > 0.0 else None,
            max_depth=float(max_depth[0]) if self.scene_extent > 0.0 else None,
        )

        return ray_bundle
