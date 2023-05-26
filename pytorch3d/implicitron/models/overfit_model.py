# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# Note: The #noqa comments below are for unused imports of pluggable implementations
# which are part of implicitron. They ensure that the registry is prepopulated.

import functools
import logging
from dataclasses import field
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import torch
from omegaconf import DictConfig

from pytorch3d.implicitron.models.base_model import (
    ImplicitronModelBase,
    ImplicitronRender,
)
from pytorch3d.implicitron.models.global_encoder.global_encoder import GlobalEncoderBase
from pytorch3d.implicitron.models.implicit_function.base import ImplicitFunctionBase
from pytorch3d.implicitron.models.metrics import (
    RegularizationMetricsBase,
    ViewMetricsBase,
)

from pytorch3d.implicitron.models.renderer.base import (
    BaseRenderer,
    EvaluationMode,
    ImplicitronRayBundle,
    RendererOutput,
    RenderSamplingMode,
)
from pytorch3d.implicitron.models.renderer.ray_sampler import RaySamplerBase
from pytorch3d.implicitron.models.utils import (
    apply_chunked,
    chunk_generator,
    log_loss_weights,
    preprocess_input,
    weighted_sum_losses,
)
from pytorch3d.implicitron.tools import vis_utils
from pytorch3d.implicitron.tools.config import (
    expand_args_fields,
    registry,
    run_auto_creation,
)

from pytorch3d.implicitron.tools.rasterize_mc import rasterize_sparse_ray_bundle
from pytorch3d.renderer import utils as rend_utils
from pytorch3d.renderer.cameras import CamerasBase


if TYPE_CHECKING:
    from visdom import Visdom
logger = logging.getLogger(__name__)

IMPLICIT_FUNCTION_ARGS_TO_REMOVE: List[str] = [
    "feature_vector_size",
    "encoding_dim",
    "latent_dim",
    "color_dim",
]


@registry.register
class OverfitModel(ImplicitronModelBase):  # pyre-ignore: 13
    """
    OverfitModel is a wrapper for the neural implicit
    rendering and reconstruction pipeline which consists
    of the following sequence of 4 steps:


        (1) Ray Sampling
        ------------------
        Rays are sampled from an image grid based on the target view(s).
                │
                ▼
        (2) Implicit Function Evaluation
        ------------------
        Evaluate the implicit function(s) at the sampled ray points
        (also optionally pass in a global encoding from global_encoder).
                │
                ▼
        (3) Rendering
        ------------------
        Render the image into the target cameras by raymarching along
        the sampled rays and aggregating the colors and densities
        output by the implicit function in (2).
                │
                ▼
        (4) Loss Computation
        ------------------
        Compute losses based on the predicted target image(s).


    The `forward` function of OverfitModel executes
    this sequence of steps. Currently, steps 1, 2, 3
    can be customized by intializing a subclass of the appropriate
    base class and adding the newly created module to the registry.
    Please see https://github.com/facebookresearch/pytorch3d/blob/main/projects/implicitron_trainer/README.md#custom-plugins
    for more details on how to create and register a custom component.

    In the config .yaml files for experiments, the parameters below are
    contained in the
    `model_factory_ImplicitronModelFactory_args.model_OverfitModel_args`
    node. As OverfitModel derives from ReplaceableBase, the input arguments are
    parsed by the run_auto_creation function to initialize the
    necessary member modules. Please see implicitron_trainer/README.md
    for more details on this process.

    Args:
        mask_images: Whether or not to mask the RGB image background given the
            foreground mask (the `fg_probability` argument of `GenericModel.forward`)
        mask_depths: Whether or not to mask the depth image background given the
            foreground mask (the `fg_probability` argument of `GenericModel.forward`)
        render_image_width: Width of the output image to render
        render_image_height: Height of the output image to render
        mask_threshold: If greater than 0.0, the foreground mask is
            thresholded by this value before being applied to the RGB/Depth images
        output_rasterized_mc: If True, visualize the Monte-Carlo pixel renders by
            splatting onto an image grid. Default: False.
        bg_color: RGB values for setting the background color of input image
            if mask_images=True. Defaults to (0.0, 0.0, 0.0). Each renderer has its own
            way to determine the background color of its output, unrelated to this.
        chunk_size_grid: The total number of points which can be rendered
            per chunk. This is used to compute the number of rays used
            per chunk when the chunked version of the renderer is used (in order
            to fit rendering on all rays in memory)
        render_features_dimensions: The number of output features to render.
            Defaults to 3, corresponding to RGB images.
        sampling_mode_training: The sampling method to use during training. Must be
            a value from the RenderSamplingMode Enum.
        sampling_mode_evaluation: Same as above but for evaluation.
        global_encoder_class_type: The name of the class to use for global_encoder,
            which must be available in the registry. Or `None` to disable global encoder.
        global_encoder: An instance of `GlobalEncoder`. This is used to generate an encoding
            of the image (referred to as the global_code) that can be used to model aspects of
            the scene such as multiple objects or morphing objects. It is up to the implicit
            function definition how to use it, but the most typical way is to broadcast and
            concatenate to the other inputs for the implicit function.
        raysampler_class_type: The name of the raysampler class which is available
            in the global registry.
        raysampler: An instance of RaySampler which is used to emit
            rays from the target view(s).
        renderer_class_type: The name of the renderer class which is available in the global
            registry.
        renderer: A renderer class which inherits from BaseRenderer. This is used to
            generate the images from the target view(s).
        share_implicit_function_across_passes: If set to True
            coarse_implicit_function is automatically set as implicit_function
            (coarse_implicit_function=implicit_funciton). The
            implicit_functions are then run sequentially during the rendering.
        implicit_function_class_type: The type of implicit function to use which
            is available in the global registry.
        implicit_function: An instance of ImplicitFunctionBase.
        coarse_implicit_function_class_type: The type of implicit function to use which
            is available in the global registry.
        coarse_implicit_function: An instance of ImplicitFunctionBase.
            If set and `share_implicit_function_across_passes` is set to False,
            coarse_implicit_function is instantiated on itself. It
            is then used as the second pass during the rendering.
            If set to None, we only do a single pass with implicit_function.
        view_metrics: An instance of ViewMetricsBase used to compute loss terms which
            are independent of the model's parameters.
        view_metrics_class_type: The type of view metrics to use, must be available in
            the global registry.
        regularization_metrics: An instance of RegularizationMetricsBase used to compute
            regularization terms which can depend on the model's parameters.
        regularization_metrics_class_type: The type of regularization metrics to use,
            must be available in the global registry.
        loss_weights: A dictionary with a {loss_name: weight} mapping; see documentation
            for `ViewMetrics` class for available loss functions.
        log_vars: A list of variable names which should be logged.
            The names should correspond to a subset of the keys of the
            dict `preds` output by the `forward` function.
    """  # noqa: B950

    mask_images: bool = True
    mask_depths: bool = True
    render_image_width: int = 400
    render_image_height: int = 400
    mask_threshold: float = 0.5
    output_rasterized_mc: bool = False
    bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    chunk_size_grid: int = 4096
    render_features_dimensions: int = 3
    tqdm_trigger_threshold: int = 16

    n_train_target_views: int = 1
    sampling_mode_training: str = "mask_sample"
    sampling_mode_evaluation: str = "full_grid"

    # ---- global encoder settings
    global_encoder_class_type: Optional[str] = None
    global_encoder: Optional[GlobalEncoderBase]

    # ---- raysampler
    raysampler_class_type: str = "AdaptiveRaySampler"
    raysampler: RaySamplerBase

    # ---- renderer configs
    renderer_class_type: str = "MultiPassEmissionAbsorptionRenderer"
    renderer: BaseRenderer

    # ---- implicit function settings
    share_implicit_function_across_passes: bool = False
    implicit_function_class_type: str = "NeuralRadianceFieldImplicitFunction"
    implicit_function: ImplicitFunctionBase
    coarse_implicit_function_class_type: Optional[str] = None
    coarse_implicit_function: Optional[ImplicitFunctionBase]

    # ----- metrics
    view_metrics: ViewMetricsBase
    view_metrics_class_type: str = "ViewMetrics"

    regularization_metrics: RegularizationMetricsBase
    regularization_metrics_class_type: str = "RegularizationMetrics"

    # ---- loss weights
    loss_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "loss_rgb_mse": 1.0,
            "loss_prev_stage_rgb_mse": 1.0,
            "loss_mask_bce": 0.0,
            "loss_prev_stage_mask_bce": 0.0,
        }
    )

    # ---- variables to be logged (logger automatically ignores if not computed)
    log_vars: List[str] = field(
        default_factory=lambda: [
            "loss_rgb_psnr_fg",
            "loss_rgb_psnr",
            "loss_rgb_mse",
            "loss_rgb_huber",
            "loss_depth_abs",
            "loss_depth_abs_fg",
            "loss_mask_neg_iou",
            "loss_mask_bce",
            "loss_mask_beta_prior",
            "loss_eikonal",
            "loss_density_tv",
            "loss_depth_neg_penalty",
            "loss_autodecoder_norm",
            # metrics that are only logged in 2+stage renderes
            "loss_prev_stage_rgb_mse",
            "loss_prev_stage_rgb_psnr_fg",
            "loss_prev_stage_rgb_psnr",
            "loss_prev_stage_mask_bce",
            # basic metrics
            "objective",
            "epoch",
            "sec/it",
        ]
    )

    @classmethod
    def pre_expand(cls) -> None:
        # use try/finally to bypass cinder's lazy imports
        try:
            from pytorch3d.implicitron.models.implicit_function.idr_feature_field import (  # noqa: F401, B950
                IdrFeatureField,
            )
            from pytorch3d.implicitron.models.implicit_function.neural_radiance_field import (  # noqa: F401, B950
                NeuralRadianceFieldImplicitFunction,
            )
            from pytorch3d.implicitron.models.implicit_function.scene_representation_networks import (  # noqa: F401, B950
                SRNImplicitFunction,
            )
            from pytorch3d.implicitron.models.renderer.lstm_renderer import (  # noqa: F401
                LSTMRenderer,
            )
            from pytorch3d.implicitron.models.renderer.multipass_ea import (  # noqa: F401
                MultiPassEmissionAbsorptionRenderer,
            )
            from pytorch3d.implicitron.models.renderer.sdf_renderer import (  # noqa: F401
                SignedDistanceFunctionRenderer,
            )
        finally:
            pass

    def __post_init__(self):
        # The attribute will be filled by run_auto_creation
        run_auto_creation(self)
        log_loss_weights(self.loss_weights, logger)
        # We need to set it here since run_auto_creation
        # will create coarse_implicit_function before implicit_function
        if self.share_implicit_function_across_passes:
            self.coarse_implicit_function = self.implicit_function

    def forward(
        self,
        *,  # force keyword-only arguments
        image_rgb: Optional[torch.Tensor],
        camera: CamerasBase,
        fg_probability: Optional[torch.Tensor] = None,
        mask_crop: Optional[torch.Tensor] = None,
        depth_map: Optional[torch.Tensor] = None,
        sequence_name: Optional[List[str]] = None,
        frame_timestamp: Optional[torch.Tensor] = None,
        evaluation_mode: EvaluationMode = EvaluationMode.EVALUATION,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Args:
            image_rgb: A tensor of shape `(B, 3, H, W)` containing a batch of rgb images;
                the first `min(B, n_train_target_views)` images are considered targets and
                are used to supervise the renders; the rest corresponding to the source
                viewpoints from which features will be extracted.
            camera: An instance of CamerasBase containing a batch of `B` cameras corresponding
                to the viewpoints of target images, from which the rays will be sampled,
                and source images, which will be used for intersecting with target rays.
            fg_probability: A tensor of shape `(B, 1, H, W)` containing a batch of
                foreground masks.
            mask_crop: A binary tensor of shape `(B, 1, H, W)` deonting valid
                regions in the input images (i.e. regions that do not correspond
                to, e.g., zero-padding). When the `RaySampler`'s sampling mode is set to
                "mask_sample", rays  will be sampled in the non zero regions.
            depth_map: A tensor of shape `(B, 1, H, W)` containing a batch of depth maps.
            sequence_name: A list of `B` strings corresponding to the sequence names
                from which images `image_rgb` were extracted. They are used to match
                target frames with relevant source frames.
            frame_timestamp: Optionally a tensor of shape `(B,)` containing a batch
                of frame timestamps.
            evaluation_mode: one of EvaluationMode.TRAINING or
                EvaluationMode.EVALUATION which determines the settings used for
                rendering.

        Returns:
            preds: A dictionary containing all outputs of the forward pass including the
                rendered images, depths, masks, losses and other metrics.
        """
        image_rgb, fg_probability, depth_map = preprocess_input(
            image_rgb,
            fg_probability,
            depth_map,
            self.mask_images,
            self.mask_depths,
            self.mask_threshold,
            self.bg_color,
        )

        # Determine the used ray sampling mode.
        sampling_mode = RenderSamplingMode(
            self.sampling_mode_training
            if evaluation_mode == EvaluationMode.TRAINING
            else self.sampling_mode_evaluation
        )

        # (1) Sample rendering rays with the ray sampler.
        # pyre-ignore[29]
        ray_bundle: ImplicitronRayBundle = self.raysampler(
            camera,
            evaluation_mode,
            mask=mask_crop
            if mask_crop is not None and sampling_mode == RenderSamplingMode.MASK_SAMPLE
            else None,
        )

        inputs_to_be_chunked = {}
        if fg_probability is not None and self.renderer.requires_object_mask():
            sampled_fb_prob = rend_utils.ndc_grid_sample(
                fg_probability, ray_bundle.xys, mode="nearest"
            )
            inputs_to_be_chunked["object_mask"] = sampled_fb_prob > 0.5

        # (2)-(3) Implicit function evaluation and Rendering
        implicit_functions: List[Union[Callable, ImplicitFunctionBase]] = [
            self.implicit_function
        ]
        if self.coarse_implicit_function is not None:
            implicit_functions = [self.coarse_implicit_function, self.implicit_function]

        if self.global_encoder is not None:
            global_code = self.global_encoder(  # pyre-fixme[29]
                sequence_name=sequence_name,
                frame_timestamp=frame_timestamp,
            )
            implicit_functions = [
                functools.partial(implicit_function, global_code=global_code)
                if isinstance(implicit_function, Callable)
                else functools.partial(
                    implicit_function.forward, global_code=global_code
                )
                for implicit_function in implicit_functions
            ]
        rendered = self._render(
            ray_bundle=ray_bundle,
            sampling_mode=sampling_mode,
            evaluation_mode=evaluation_mode,
            implicit_functions=implicit_functions,
            inputs_to_be_chunked=inputs_to_be_chunked,
        )

        # A dict to store losses as well as rendering results.
        preds: Dict[str, Any] = self.view_metrics(
            results={},
            raymarched=rendered,
            ray_bundle=ray_bundle,
            image_rgb=image_rgb,
            depth_map=depth_map,
            fg_probability=fg_probability,
            mask_crop=mask_crop,
        )

        preds.update(
            self.regularization_metrics(
                results=preds,
                model=self,
            )
        )

        if sampling_mode == RenderSamplingMode.MASK_SAMPLE:
            if self.output_rasterized_mc:
                # Visualize the monte-carlo pixel renders by splatting onto
                # an image grid.
                (
                    preds["images_render"],
                    preds["depths_render"],
                    preds["masks_render"],
                ) = rasterize_sparse_ray_bundle(
                    ray_bundle,
                    rendered.features,
                    (self.render_image_height, self.render_image_width),
                    rendered.depths,
                    masks=rendered.masks,
                )
        elif sampling_mode == RenderSamplingMode.FULL_GRID:
            preds["images_render"] = rendered.features.permute(0, 3, 1, 2)
            preds["depths_render"] = rendered.depths.permute(0, 3, 1, 2)
            preds["masks_render"] = rendered.masks.permute(0, 3, 1, 2)

            preds["implicitron_render"] = ImplicitronRender(
                image_render=preds["images_render"],
                depth_render=preds["depths_render"],
                mask_render=preds["masks_render"],
            )
        else:
            raise AssertionError("Unreachable state")

        # (4) Compute losses
        # finally get the optimization objective using self.loss_weights
        objective = self._get_objective(preds)
        if objective is not None:
            preds["objective"] = objective

        return preds

    def _get_objective(self, preds: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """
        A helper function to compute the overall loss as the dot product
        of individual loss functions with the corresponding weights.
        """
        return weighted_sum_losses(preds, self.loss_weights)

    def visualize(
        self,
        viz: Optional["Visdom"],
        visdom_env_imgs: str,
        preds: Dict[str, Any],
        prefix: str,
    ) -> None:
        """
        Helper function to visualize the predictions generated
        in the forward pass.

        Args:
            viz: Visdom connection object
            visdom_env_imgs: name of visdom environment for the images.
            preds: predictions dict like returned by forward()
            prefix: prepended to the names of images
        """
        if viz is None or not viz.check_connection():
            logger.info("no visdom server! -> skipping batch vis")
            return

        idx_image = 0
        title = f"{prefix}_im{idx_image}"

        vis_utils.visualize_basics(viz, preds, visdom_env_imgs, title=title)

    def _render(
        self,
        *,
        ray_bundle: ImplicitronRayBundle,
        inputs_to_be_chunked: Dict[str, torch.Tensor],
        sampling_mode: RenderSamplingMode,
        **kwargs,
    ) -> RendererOutput:
        """
        Args:
            ray_bundle: A `ImplicitronRayBundle` object containing the parametrizations of the
                sampled rendering rays.
            inputs_to_be_chunked: A collection of tensor of shape `(B, _, H, W)`. E.g.
                SignedDistanceFunctionRenderer requires "object_mask", shape
                (B, 1, H, W), the silhouette of the object in the image. When
                chunking, they are passed to the renderer as shape
                `(B, _, chunksize)`.
            sampling_mode: The sampling method to use. Must be a value from the
                RenderSamplingMode Enum.

        Returns:
            An instance of RendererOutput
        """
        if sampling_mode == RenderSamplingMode.FULL_GRID and self.chunk_size_grid > 0:
            return apply_chunked(
                self.renderer,
                chunk_generator(
                    self.chunk_size_grid,
                    ray_bundle,
                    inputs_to_be_chunked,
                    self.tqdm_trigger_threshold,
                    **kwargs,
                ),
                lambda batch: torch.cat(batch, dim=1).reshape(
                    *ray_bundle.lengths.shape[:-1], -1
                ),
            )
        else:
            # pyre-fixme[29]: `BaseRenderer` is not a function.
            return self.renderer(
                ray_bundle=ray_bundle,
                **inputs_to_be_chunked,
                **kwargs,
            )

    @classmethod
    def raysampler_tweak_args(cls, type, args: DictConfig) -> None:
        """
        We don't expose certain fields of the raysampler because we want to set
        them from our own members.
        """
        del args["sampling_mode_training"]
        del args["sampling_mode_evaluation"]
        del args["image_width"]
        del args["image_height"]

    def create_raysampler(self):
        extra_args = {
            "sampling_mode_training": self.sampling_mode_training,
            "sampling_mode_evaluation": self.sampling_mode_evaluation,
            "image_width": self.render_image_width,
            "image_height": self.render_image_height,
        }
        raysampler_args = getattr(
            self, "raysampler_" + self.raysampler_class_type + "_args"
        )
        self.raysampler = registry.get(RaySamplerBase, self.raysampler_class_type)(
            **raysampler_args, **extra_args
        )

    @classmethod
    def renderer_tweak_args(cls, type, args: DictConfig) -> None:
        """
        We don't expose certain fields of the renderer because we want to set
        them based on other inputs.
        """
        args.pop("render_features_dimensions", None)
        args.pop("object_bounding_sphere", None)

    def create_renderer(self):
        extra_args = {}

        if self.renderer_class_type == "SignedDistanceFunctionRenderer":
            extra_args["render_features_dimensions"] = self.render_features_dimensions
            if not hasattr(self.raysampler, "scene_extent"):
                raise ValueError(
                    "SignedDistanceFunctionRenderer requires"
                    + " a raysampler that defines the 'scene_extent' field"
                    + " (this field is supported by, e.g., the adaptive raysampler - "
                    + " self.raysampler_class_type='AdaptiveRaySampler')."
                )
            extra_args["object_bounding_sphere"] = self.raysampler.scene_extent

        renderer_args = getattr(self, "renderer_" + self.renderer_class_type + "_args")
        self.renderer = registry.get(BaseRenderer, self.renderer_class_type)(
            **renderer_args, **extra_args
        )

    @classmethod
    def implicit_function_tweak_args(cls, type, args: DictConfig) -> None:
        """
        We don't expose certain implicit_function fields because we want to set
        them based on other inputs.
        """
        for arg in IMPLICIT_FUNCTION_ARGS_TO_REMOVE:
            args.pop(arg, None)

    @classmethod
    def coarse_implicit_function_tweak_args(cls, type, args: DictConfig) -> None:
        """
        We don't expose certain implicit_function fields because we want to set
        them based on other inputs.
        """
        for arg in IMPLICIT_FUNCTION_ARGS_TO_REMOVE:
            args.pop(arg, None)

    def _create_extra_args_for_implicit_function(self) -> Dict[str, Any]:
        extra_args = {}
        global_encoder_dim = (
            0 if self.global_encoder is None else self.global_encoder.get_encoding_dim()
        )
        if self.implicit_function_class_type in (
            "NeuralRadianceFieldImplicitFunction",
            "NeRFormerImplicitFunction",
        ):
            extra_args["latent_dim"] = global_encoder_dim
            extra_args["color_dim"] = self.render_features_dimensions

        if self.implicit_function_class_type == "IdrFeatureField":
            extra_args["feature_work_size"] = global_encoder_dim
            extra_args["feature_vector_size"] = self.render_features_dimensions

        if self.implicit_function_class_type == "SRNImplicitFunction":
            extra_args["latent_dim"] = global_encoder_dim
        return extra_args

    def create_implicit_function(self) -> None:
        implicit_function_type = registry.get(
            ImplicitFunctionBase, self.implicit_function_class_type
        )
        expand_args_fields(implicit_function_type)

        config_name = f"implicit_function_{self.implicit_function_class_type}_args"
        config = getattr(self, config_name, None)
        if config is None:
            raise ValueError(f"{config_name} not present")

        extra_args = self._create_extra_args_for_implicit_function()
        self.implicit_function = implicit_function_type(**config, **extra_args)

    def create_coarse_implicit_function(self) -> None:
        # If coarse_implicit_function_class_type has been defined
        # then we init a module based on its arguments
        if (
            self.coarse_implicit_function_class_type is not None
            and not self.share_implicit_function_across_passes
        ):
            config_name = "coarse_implicit_function_{0}_args".format(
                self.coarse_implicit_function_class_type
            )
            config = getattr(self, config_name, {})

            implicit_function_type = registry.get(
                ImplicitFunctionBase,
                # pyre-ignore: config is None allow to check if this is None.
                self.coarse_implicit_function_class_type,
            )
            expand_args_fields(implicit_function_type)

            extra_args = self._create_extra_args_for_implicit_function()
            self.coarse_implicit_function = implicit_function_type(
                **config, **extra_args
            )
        elif self.share_implicit_function_across_passes:
            # Since coarse_implicit_function is initialised before
            # implicit_function we handle this case in the post_init.
            pass
        else:
            self.coarse_implicit_function = None
