# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from pytorch3d.implicitron.tools.config import registry

from .base import BaseRenderer, EvaluationMode, RendererOutput
from .ray_point_refiner import RayPointRefiner
from .raymarcher import GenericRaymarcher


@registry.register
class MultiPassEmissionAbsorptionRenderer(BaseRenderer, torch.nn.Module):
    """
    Implements the multi-pass rendering function, in particular,
    with emission-absorption ray marching used in NeRF [1]. First, it evaluates
    opacity-based ray-point weights and then optionally (in case more implicit
    functions are given) resamples points using importance sampling and evaluates
    new weights.

    During each ray marching pass, features, depth map, and masks
    are integrated: Let o_i be the opacity estimated by the implicit function,
    and d_i be the offset between points `i` and `i+1` along the respective ray.
    Ray marching is performed using the following equations:
    ```
    ray_opacity_n = cap_fn(sum_i=1^n cap_fn(d_i * o_i)),
    weight_n = weight_fn(cap_fn(d_i * o_i), 1 - ray_opacity_{n-1}),
    ```
    and the final rendered quantities are computed by a dot-product of ray values
    with the weights, e.g. `features = sum_n(weight_n * ray_features_n)`.
    See below for possible values of `cap_fn` and `weight_fn`.

    Settings:
        n_pts_per_ray_fine_training: The number of points sampled per ray for the
            fine rendering pass during training.
        n_pts_per_ray_fine_evaluation: The number of points sampled per ray for the
            fine rendering pass during evaluation.
        stratified_sampling_coarse_training: Enable/disable stratified sampling during
            training.
        stratified_sampling_coarse_evaluation: Enable/disable stratified sampling during
            evaluation.
        append_coarse_samples_to_fine: Add the fine ray points to the coarse points
            after sampling.
        bg_color: The background color. A tuple of either 1 element or of D elements,
            where D matches the feature dimensionality; it is broadcasted when necessary.
        density_noise_std_train: Standard deviation of the noise added to the
            opacity field.
        capping_function: The capping function of the raymarcher.
            Options:
                - "exponential" (`cap_fn(x) = 1 - exp(-x)`)
                - "cap1" (`cap_fn(x) = min(x, 1)`)
            Set to "exponential" for the standard Emission Absorption raymarching.
        weight_function: The weighting function of the raymarcher.
            Options:
                - "product" (`weight_fn(w, x) = w * x`)
                - "minimum" (`weight_fn(w, x) = min(w, x)`)
            Set to "product" for the standard Emission Absorption raymarching.
        background_opacity: The raw opacity value (i.e. before exponentiation)
            of the background.
        blend_output: If `True`, alpha-blends the output renders with the
            background color using the rendered opacity mask.

    References:
        [1] Mildenhall, Ben, et al. "Nerf: Representing scenes as neural radiance
            fields for view synthesis." ECCV 2020.

    """

    n_pts_per_ray_fine_training: int = 64
    n_pts_per_ray_fine_evaluation: int = 64
    stratified_sampling_coarse_training: bool = True
    stratified_sampling_coarse_evaluation: bool = False
    append_coarse_samples_to_fine: bool = True
    bg_color: Tuple[float, ...] = (0.0,)
    density_noise_std_train: float = 0.0
    capping_function: str = "exponential"  # exponential | cap1
    weight_function: str = "product"  # product | minimum
    background_opacity: float = 1e10
    blend_output: bool = False

    def __post_init__(self):
        super().__init__()
        self._refiners = {
            EvaluationMode.TRAINING: RayPointRefiner(
                n_pts_per_ray=self.n_pts_per_ray_fine_training,
                random_sampling=self.stratified_sampling_coarse_training,
                add_input_samples=self.append_coarse_samples_to_fine,
            ),
            EvaluationMode.EVALUATION: RayPointRefiner(
                n_pts_per_ray=self.n_pts_per_ray_fine_evaluation,
                random_sampling=self.stratified_sampling_coarse_evaluation,
                add_input_samples=self.append_coarse_samples_to_fine,
            ),
        }

        self._raymarcher = GenericRaymarcher(
            1,
            self.bg_color,
            capping_function=self.capping_function,
            weight_function=self.weight_function,
            background_opacity=self.background_opacity,
            blend_output=self.blend_output,
        )

    def forward(
        self,
        ray_bundle,
        implicit_functions=[],
        evaluation_mode: EvaluationMode = EvaluationMode.EVALUATION,
        **kwargs
    ) -> RendererOutput:
        """
        Args:
            ray_bundle: A `RayBundle` object containing the parametrizations of the
                sampled rendering rays.
            implicit_functions: List of ImplicitFunctionWrappers which
                define the implicit functions to be used sequentially in
                the raymarching step. The output of raymarching with
                implicit_functions[n-1] is refined, and then used as
                input for raymarching with implicit_functions[n].
            evaluation_mode: one of EvaluationMode.TRAINING or
                EvaluationMode.EVALUATION which determines the settings used for
                rendering

        Returns:
            instance of RendererOutput
        """
        if not implicit_functions:
            raise ValueError("EA renderer expects implicit functions")

        return self._run_raymarcher(
            ray_bundle,
            implicit_functions,
            None,
            evaluation_mode,
        )

    def _run_raymarcher(
        self, ray_bundle, implicit_functions, prev_stage, evaluation_mode
    ):
        density_noise_std = (
            self.density_noise_std_train
            if evaluation_mode == EvaluationMode.TRAINING
            else 0.0
        )

        features, depth, mask, weights, aux = self._raymarcher(
            *implicit_functions[0](ray_bundle),
            ray_lengths=ray_bundle.lengths,
            density_noise_std=density_noise_std,
        )
        output = RendererOutput(
            features=features, depths=depth, masks=mask, aux=aux, prev_stage=prev_stage
        )

        # we may need to make a recursive call
        if len(implicit_functions) > 1:
            fine_ray_bundle = self._refiners[evaluation_mode](ray_bundle, weights)
            output = self._run_raymarcher(
                fine_ray_bundle,
                implicit_functions[1:],
                output,
                evaluation_mode,
            )

        return output
