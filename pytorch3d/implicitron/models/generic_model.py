# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# Note: The #noqa comments below are for unused imports of pluggable implementations
# which are part of implicitron. They ensure that the registry is prepopulated.

import logging
import math
import warnings
from dataclasses import field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import tqdm
from pytorch3d.implicitron.models.metrics import (  # noqa
    RegularizationMetrics,
    RegularizationMetricsBase,
    ViewMetrics,
    ViewMetricsBase,
)
from pytorch3d.implicitron.tools import image_utils, vis_utils
from pytorch3d.implicitron.tools.config import (
    expand_args_fields,
    registry,
    run_auto_creation,
)
from pytorch3d.implicitron.tools.rasterize_mc import rasterize_mc_samples
from pytorch3d.implicitron.tools.utils import cat_dataclass, setattr_if_hasattr
from pytorch3d.renderer import RayBundle, utils as rend_utils
from pytorch3d.renderer.cameras import CamerasBase
from visdom import Visdom

from .base_model import ImplicitronModelBase, ImplicitronRender
from .feature_extractor import FeatureExtractorBase
from .feature_extractor.resnet_feature_extractor import ResNetFeatureExtractor  # noqa
from .global_encoder.global_encoder import GlobalEncoderBase
from .implicit_function.base import ImplicitFunctionBase
from .implicit_function.idr_feature_field import IdrFeatureField  # noqa
from .implicit_function.neural_radiance_field import (  # noqa
    NeRFormerImplicitFunction,
    NeuralRadianceFieldImplicitFunction,
)
from .implicit_function.scene_representation_networks import (  # noqa
    SRNHyperNetImplicitFunction,
    SRNImplicitFunction,
)

from .renderer.base import (
    BaseRenderer,
    EvaluationMode,
    ImplicitFunctionWrapper,
    RendererOutput,
    RenderSamplingMode,
)
from .renderer.lstm_renderer import LSTMRenderer  # noqa
from .renderer.multipass_ea import MultiPassEmissionAbsorptionRenderer  # noqa
from .renderer.ray_sampler import RaySamplerBase
from .renderer.sdf_renderer import SignedDistanceFunctionRenderer  # noqa
from .view_pooler.view_pooler import ViewPooler


logger = logging.getLogger(__name__)


@registry.register
class GenericModel(ImplicitronModelBase, torch.nn.Module):  # pyre-ignore: 13
    """
    GenericModel is a wrapper for the neural implicit
    rendering and reconstruction pipeline which consists
    of the following sequence of 7 steps (steps 2–4 are normally
    skipped in overfitting scenario, since conditioning on source views
    does not add much information; otherwise they should be present altogether):


        (1) Ray Sampling
        ------------------
        Rays are sampled from an image grid based on the target view(s).
                │_____________
                │             │
                │             ▼
                │    (2) Feature Extraction (optional)
                │    -----------------------
                │    A feature extractor (e.g. a convolutional
                │    neural net) is used to extract image features
                │    from the source view(s).
                │            │
                │            ▼
                │    (3) View Sampling  (optional)
                │    ------------------
                │    Image features are sampled at the 2D projections
                │    of a set of 3D points along each of the sampled
                │    target rays from (1).
                │            │
                │            ▼
                │    (4) Feature Aggregation  (optional)
                │    ------------------
                │    Aggregate features and masks sampled from
                │    image view(s) in (3).
                │            │
                │____________▼
                │
                ▼
        (5) Implicit Function Evaluation
        ------------------
        Evaluate the implicit function(s) at the sampled ray points
        (optionally pass in the aggregated image features from (4)).
        (also optionally pass in a global encoding from global_encoder).
                │
                ▼
        (6) Rendering
        ------------------
        Render the image into the target cameras by raymarching along
        the sampled rays and aggregating the colors and densities
        output by the implicit function in (5).
                │
                ▼
        (7) Loss Computation
        ------------------
        Compute losses based on the predicted target image(s).


    The `forward` function of GenericModel executes
    this sequence of steps. Currently, steps 1, 3, 4, 5, 6
    can be customized by intializing a subclass of the appropriate
    baseclass and adding the newly created module to the registry.
    Please see https://github.com/facebookresearch/pytorch3d/blob/main/projects/implicitron_trainer/README.md#custom-plugins
    for more details on how to create and register a custom component.

    In the config .yaml files for experiments, the parameters below are
    contained in the `generic_model_args` node. As GenericModel
    derives from Configurable, the input arguments are
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
        bg_color: RGB values for the background color. Default (0.0, 0.0, 0.0)
        num_passes: The specified implicit_function is initialized num_passes
            times and run sequentially.
        chunk_size_grid: The total number of points which can be rendered
            per chunk. This is used to compute the number of rays used
            per chunk when the chunked version of the renderer is used (in order
            to fit rendering on all rays in memory)
        render_features_dimensions: The number of output features to render.
            Defaults to 3, corresponding to RGB images.
        n_train_target_views: The number of cameras to render into at training
            time; first `n_train_target_views` in the batch are considered targets,
            the rest are sources.
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
        image_feature_extractor_class_type: If a str, constructs and enables
            the `image_feature_extractor` object of this type. Or None if not needed.
        image_feature_extractor: A module for extrating features from an input image.
        view_pooler_enabled: If `True`, constructs and enables the `view_pooler` object.
            This means features are sampled from the source image(s)
            at the projected 2d locations of the sampled 3d ray points from the target
            view(s), i.e. this activates step (3) above.
        view_pooler: An instance of ViewPooler which is used for sampling of
            image-based features at the 2D projections of a set
            of 3D points and aggregating the sampled features.
        implicit_function_class_type: The type of implicit function to use which
            is available in the global registry.
        implicit_function: An instance of ImplicitFunctionBase. The actual implicit functions
            are initialised to be in self._implicit_functions.
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
    num_passes: int = 1
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

    # ---- image feature extractor settings
    # (This is only created if view_pooler is enabled)
    image_feature_extractor: Optional[FeatureExtractorBase]
    image_feature_extractor_class_type: Optional[str] = None
    # ---- view pooler settings
    view_pooler_enabled: bool = False
    view_pooler: Optional[ViewPooler]

    # ---- implicit function settings
    implicit_function_class_type: str = "NeuralRadianceFieldImplicitFunction"
    # This is just a model, never constructed.
    # The actual implicit functions live in self._implicit_functions
    implicit_function: ImplicitFunctionBase

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

    def __post_init__(self):
        super().__init__()

        if self.view_pooler_enabled:
            if self.image_feature_extractor_class_type is None:
                raise ValueError(
                    "image_feature_extractor must be present for view pooling."
                )
        run_auto_creation(self)

        self._implicit_functions = self._construct_implicit_functions()

        self.log_loss_weights()

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
        image_rgb, fg_probability, depth_map = self._preprocess_input(
            image_rgb, fg_probability, depth_map
        )

        # Obtain the batch size from the camera as this is the only required input.
        batch_size = camera.R.shape[0]

        # Determine the number of target views, i.e. cameras we render into.
        n_targets = (
            1
            if evaluation_mode == EvaluationMode.EVALUATION
            else batch_size
            if self.n_train_target_views <= 0
            else min(self.n_train_target_views, batch_size)
        )

        # A helper function for selecting n_target first elements from the input
        # where the latter can be None.
        def safe_slice_targets(
            tensor: Optional[Union[torch.Tensor, List[str]]],
        ) -> Optional[Union[torch.Tensor, List[str]]]:
            return None if tensor is None else tensor[:n_targets]

        # Select the target cameras.
        target_cameras = camera[list(range(n_targets))]

        # Determine the used ray sampling mode.
        sampling_mode = RenderSamplingMode(
            self.sampling_mode_training
            if evaluation_mode == EvaluationMode.TRAINING
            else self.sampling_mode_evaluation
        )

        # (1) Sample rendering rays with the ray sampler.
        ray_bundle: RayBundle = self.raysampler(  # pyre-fixme[29]
            target_cameras,
            evaluation_mode,
            mask=mask_crop[:n_targets]
            if mask_crop is not None and sampling_mode == RenderSamplingMode.MASK_SAMPLE
            else None,
        )

        # custom_args hold additional arguments to the implicit function.
        custom_args = {}

        if self.image_feature_extractor is not None:
            # (2) Extract features for the image
            img_feats = self.image_feature_extractor(image_rgb, fg_probability)
        else:
            img_feats = None

        if self.view_pooler_enabled:
            if sequence_name is None:
                raise ValueError("sequence_name must be provided for view pooling")
            assert img_feats is not None

            # (3-4) Sample features and masks at the ray points.
            #       Aggregate features from multiple views.
            def curried_viewpooler(pts):
                return self.view_pooler(
                    pts=pts,
                    seq_id_pts=sequence_name[:n_targets],
                    camera=camera,
                    seq_id_camera=sequence_name,
                    feats=img_feats,
                    masks=mask_crop,
                )

            custom_args["fun_viewpool"] = curried_viewpooler

        global_code = None
        if self.global_encoder is not None:
            global_code = self.global_encoder(  # pyre-fixme[29]
                sequence_name=safe_slice_targets(sequence_name),
                frame_timestamp=safe_slice_targets(frame_timestamp),
            )
        custom_args["global_code"] = global_code

        # pyre-fixme[29]:
        #  `Union[BoundMethod[typing.Callable(torch.Tensor.__iter__)[[Named(self,
        #  torch.Tensor)], typing.Iterator[typing.Any]], torch.Tensor], torch.Tensor,
        #  torch.nn.Module]` is not a function.
        for func in self._implicit_functions:
            func.bind_args(**custom_args)

        chunked_renderer_inputs = {}
        if fg_probability is not None and self.renderer.requires_object_mask():
            sampled_fb_prob = rend_utils.ndc_grid_sample(
                fg_probability[:n_targets], ray_bundle.xys, mode="nearest"
            )
            chunked_renderer_inputs["object_mask"] = sampled_fb_prob > 0.5

        # (5)-(6) Implicit function evaluation and Rendering
        rendered = self._render(
            ray_bundle=ray_bundle,
            sampling_mode=sampling_mode,
            evaluation_mode=evaluation_mode,
            implicit_functions=self._implicit_functions,
            chunked_inputs=chunked_renderer_inputs,
        )

        # Unbind the custom arguments to prevent pytorch from storing
        # large buffers of intermediate results due to points in the
        # bound arguments.
        # pyre-fixme[29]:
        #  `Union[BoundMethod[typing.Callable(torch.Tensor.__iter__)[[Named(self,
        #  torch.Tensor)], typing.Iterator[typing.Any]], torch.Tensor], torch.Tensor,
        #  torch.nn.Module]` is not a function.
        for func in self._implicit_functions:
            func.unbind_args()

        # A dict to store losses as well as rendering results.
        preds: Dict[str, Any] = {}

        preds.update(
            self.view_metrics(
                results=preds,
                raymarched=rendered,
                xys=ray_bundle.xys,
                image_rgb=safe_slice_targets(image_rgb),
                depth_map=safe_slice_targets(depth_map),
                fg_probability=safe_slice_targets(fg_probability),
                mask_crop=safe_slice_targets(mask_crop),
            )
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
                ) = self._rasterize_mc_samples(
                    ray_bundle.xys,
                    rendered.features,
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

        # (7) Compute losses
        # finally get the optimization objective using self.loss_weights
        objective = self._get_objective(preds)
        if objective is not None:
            preds["objective"] = objective

        return preds

    def _get_objective(self, preds) -> Optional[torch.Tensor]:
        """
        A helper function to compute the overall loss as the dot product
        of individual loss functions with the corresponding weights.
        """
        losses_weighted = [
            preds[k] * float(w)
            for k, w in self.loss_weights.items()
            if (k in preds and w != 0.0)
        ]
        if len(losses_weighted) == 0:
            warnings.warn("No main objective found.")
            return None
        loss = sum(losses_weighted)
        assert torch.is_tensor(loss)
        return loss

    def visualize(
        self,
        viz: Visdom,
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
        if not viz.check_connection():
            logger.info("no visdom server! -> skipping batch vis")
            return

        idx_image = 0
        title = f"{prefix}_im{idx_image}"

        vis_utils.visualize_basics(viz, preds, visdom_env_imgs, title=title)

    def _render(
        self,
        *,
        ray_bundle: RayBundle,
        chunked_inputs: Dict[str, torch.Tensor],
        sampling_mode: RenderSamplingMode,
        **kwargs,
    ) -> RendererOutput:
        """
        Args:
            ray_bundle: A `RayBundle` object containing the parametrizations of the
                sampled rendering rays.
            chunked_inputs: A collection of tensor of shape `(B, _, H, W)`. E.g.
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
            return _apply_chunked(
                self.renderer,
                _chunk_generator(
                    self.chunk_size_grid,
                    ray_bundle,
                    chunked_inputs,
                    self.tqdm_trigger_threshold,
                    **kwargs,
                ),
                lambda batch: _tensor_collator(batch, ray_bundle.lengths.shape[:-1]),
            )
        else:
            # pyre-fixme[29]: `BaseRenderer` is not a function.
            return self.renderer(
                ray_bundle=ray_bundle,
                **chunked_inputs,
                **kwargs,
            )

    def _get_global_encoder_encoding_dim(self) -> int:
        if self.global_encoder is None:
            return 0
        return self.global_encoder.get_encoding_dim()

    def _get_viewpooled_feature_dim(self) -> int:
        if self.view_pooler is None:
            return 0
        assert self.image_feature_extractor is not None
        return self.view_pooler.get_aggregated_feature_dim(
            self.image_feature_extractor.get_feat_dims()
        )

    def create_raysampler(self):
        raysampler_args = getattr(
            self, "raysampler_" + self.raysampler_class_type + "_args"
        )
        setattr_if_hasattr(
            raysampler_args, "sampling_mode_training", self.sampling_mode_training
        )
        setattr_if_hasattr(
            raysampler_args, "sampling_mode_evaluation", self.sampling_mode_evaluation
        )
        setattr_if_hasattr(raysampler_args, "image_width", self.render_image_width)
        setattr_if_hasattr(raysampler_args, "image_height", self.render_image_height)
        self.raysampler = registry.get(RaySamplerBase, self.raysampler_class_type)(
            **raysampler_args
        )

    def create_renderer(self):
        raysampler_args = getattr(
            self, "raysampler_" + self.raysampler_class_type + "_args"
        )
        self.renderer_MultiPassEmissionAbsorptionRenderer_args[
            "stratified_sampling_coarse_training"
        ] = raysampler_args["stratified_point_sampling_training"]
        self.renderer_MultiPassEmissionAbsorptionRenderer_args[
            "stratified_sampling_coarse_evaluation"
        ] = raysampler_args["stratified_point_sampling_evaluation"]
        self.renderer_SignedDistanceFunctionRenderer_args[
            "render_features_dimensions"
        ] = self.render_features_dimensions

        if self.renderer_class_type == "SignedDistanceFunctionRenderer":
            if "scene_extent" not in raysampler_args:
                raise ValueError(
                    "SignedDistanceFunctionRenderer requires"
                    + " a raysampler that defines the 'scene_extent' field"
                    + " (this field is supported by, e.g., the adaptive raysampler - "
                    + " self.raysampler_class_type='AdaptiveRaySampler')."
                )
            self.renderer_SignedDistanceFunctionRenderer_args.ray_tracer_args[
                "object_bounding_sphere"
            ] = self.raysampler_AdaptiveRaySampler_args["scene_extent"]

        renderer_args = getattr(self, "renderer_" + self.renderer_class_type + "_args")
        self.renderer = registry.get(BaseRenderer, self.renderer_class_type)(
            **renderer_args
        )

    def create_implicit_function(self) -> None:
        """
        No-op called by run_auto_creation so that self.implicit_function
        does not get created. __post_init__ creates the implicit function(s)
        in wrappers explicitly in self._implicit_functions.
        """
        pass

    def _construct_implicit_functions(self):
        """
        After run_auto_creation has been called, the arguments
        for each of the possible implicit function methods are
        available. `GenericModel` arguments are first validated
        based on the custom requirements for each specific
        implicit function method. Then the required implicit
        function(s) are initialized.
        """
        # nerf preprocessing
        nerf_args = self.implicit_function_NeuralRadianceFieldImplicitFunction_args
        nerformer_args = self.implicit_function_NeRFormerImplicitFunction_args
        nerf_args["latent_dim"] = nerformer_args["latent_dim"] = (
            self._get_viewpooled_feature_dim() + self._get_global_encoder_encoding_dim()
        )
        nerf_args["color_dim"] = nerformer_args[
            "color_dim"
        ] = self.render_features_dimensions

        # idr preprocessing
        idr = self.implicit_function_IdrFeatureField_args
        idr["feature_vector_size"] = self.render_features_dimensions
        idr["encoding_dim"] = self._get_global_encoder_encoding_dim()

        # srn preprocessing
        srn = self.implicit_function_SRNImplicitFunction_args
        srn.raymarch_function_args.latent_dim = (
            self._get_viewpooled_feature_dim() + self._get_global_encoder_encoding_dim()
        )

        # srn_hypernet preprocessing
        srn_hypernet = self.implicit_function_SRNHyperNetImplicitFunction_args
        srn_hypernet_args = srn_hypernet.hypernet_args
        srn_hypernet_args.latent_dim_hypernet = self._get_global_encoder_encoding_dim()
        srn_hypernet_args.latent_dim = self._get_viewpooled_feature_dim()

        # check that for srn, srn_hypernet, idr we have self.num_passes=1
        implicit_function_type = registry.get(
            ImplicitFunctionBase, self.implicit_function_class_type
        )
        expand_args_fields(implicit_function_type)
        if self.num_passes != 1 and not implicit_function_type.allows_multiple_passes():
            raise ValueError(
                self.implicit_function_class_type
                + f"requires num_passes=1 not {self.num_passes}"
            )

        if implicit_function_type.requires_pooling_without_aggregation():
            if self.view_pooler_enabled and self.view_pooler.has_aggregation():
                raise ValueError(
                    "The chosen implicit function requires view pooling without aggregation."
                )
        config_name = f"implicit_function_{self.implicit_function_class_type}_args"
        config = getattr(self, config_name, None)
        if config is None:
            raise ValueError(f"{config_name} not present")
        implicit_functions_list = [
            ImplicitFunctionWrapper(implicit_function_type(**config))
            for _ in range(self.num_passes)
        ]
        return torch.nn.ModuleList(implicit_functions_list)

    def log_loss_weights(self) -> None:
        """
        Print a table of the loss weights.
        """
        loss_weights_message = (
            "-------\nloss_weights:\n"
            + "\n".join(f"{k:40s}: {w:1.2e}" for k, w in self.loss_weights.items())
            + "-------"
        )
        logger.info(loss_weights_message)

    def _preprocess_input(
        self,
        image_rgb: Optional[torch.Tensor],
        fg_probability: Optional[torch.Tensor],
        depth_map: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Helper function to preprocess the input images and optional depth maps
        to apply masking if required.

        Args:
            image_rgb: A tensor of shape `(B, 3, H, W)` containing a batch of rgb images
                corresponding to the source viewpoints from which features will be extracted
            fg_probability: A tensor of shape `(B, 1, H, W)` containing a batch
                of foreground masks with values in [0, 1].
            depth_map: A tensor of shape `(B, 1, H, W)` containing a batch of depth maps.

        Returns:
            Modified image_rgb, fg_mask, depth_map
        """
        fg_mask = fg_probability
        if fg_mask is not None and self.mask_threshold > 0.0:
            # threshold masks
            warnings.warn("Thresholding masks!")
            fg_mask = (fg_mask >= self.mask_threshold).type_as(fg_mask)

        if self.mask_images and fg_mask is not None and image_rgb is not None:
            # mask the image
            warnings.warn("Masking images!")
            image_rgb = image_utils.mask_background(
                image_rgb, fg_mask, dim_color=1, bg_color=torch.tensor(self.bg_color)
            )

        if self.mask_depths and fg_mask is not None and depth_map is not None:
            # mask the depths
            assert (
                self.mask_threshold > 0.0
            ), "Depths should be masked only with thresholded masks"
            warnings.warn("Masking depths!")
            depth_map = depth_map * fg_mask

        return image_rgb, fg_mask, depth_map

    @torch.no_grad()
    def _rasterize_mc_samples(
        self,
        xys: torch.Tensor,
        features: torch.Tensor,
        depth: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Rasterizes Monte-Carlo features back onto the image.

        Args:
            xys: B x ... x 2 2D point locations in PyTorch3D NDC convention
            features: B x ... x C tensor containing per-point rendered features.
            depth: B x ... x 1 tensor containing per-point rendered depth.
        """
        ba = xys.shape[0]

        # Flatten the features and xy locations.
        features_depth_ras = torch.cat(
            (
                features.reshape(ba, -1, features.shape[-1]),
                depth.reshape(ba, -1, 1),
            ),
            dim=-1,
        )
        xys_ras = xys.reshape(ba, -1, 2)
        if masks is not None:
            masks_ras = masks.reshape(ba, -1, 1)
        else:
            masks_ras = None

        if min(self.render_image_height, self.render_image_width) <= 0:
            raise ValueError(
                "Need to specify a positive"
                " self.render_image_height and self.render_image_width"
                " for MC rasterisation."
            )

        # Estimate the rasterization point radius so that we approximately fill
        # the whole image given the number of rasterized points.
        pt_radius = 2.0 * math.sqrt(xys.shape[1])

        # Rasterize the samples.
        features_depth_render, masks_render = rasterize_mc_samples(
            xys_ras,
            features_depth_ras,
            (self.render_image_height, self.render_image_width),
            radius=pt_radius,
            masks=masks_ras,
        )
        images_render = features_depth_render[:, :-1]
        depths_render = features_depth_render[:, -1:]
        return images_render, depths_render, masks_render


def _apply_chunked(func, chunk_generator, tensor_collator):
    """
    Helper function to apply a function on a sequence of
    chunked inputs yielded by a generator and collate
    the result.
    """
    processed_chunks = [
        func(*chunk_args, **chunk_kwargs)
        for chunk_args, chunk_kwargs in chunk_generator
    ]

    return cat_dataclass(processed_chunks, tensor_collator)


def _tensor_collator(batch, new_dims) -> torch.Tensor:
    """
    Helper function to reshape the batch to the desired shape
    """
    return torch.cat(batch, dim=1).reshape(*new_dims, -1)


def _chunk_generator(
    chunk_size: int,
    ray_bundle: RayBundle,
    chunked_inputs: Dict[str, torch.Tensor],
    tqdm_trigger_threshold: int,
    *args,
    **kwargs,
):
    """
    Helper function which yields chunks of rays from the
    input ray_bundle, to be used when the number of rays is
    large and will not fit in memory for rendering.
    """
    (
        batch_size,
        *spatial_dim,
        n_pts_per_ray,
    ) = ray_bundle.lengths.shape  # B x ... x n_pts_per_ray
    if n_pts_per_ray > 0 and chunk_size % n_pts_per_ray != 0:
        raise ValueError(
            f"chunk_size_grid ({chunk_size}) should be divisible "
            f"by n_pts_per_ray ({n_pts_per_ray})"
        )

    n_rays = math.prod(spatial_dim)
    # special handling for raytracing-based methods
    n_chunks = -(-n_rays * max(n_pts_per_ray, 1) // chunk_size)
    chunk_size_in_rays = -(-n_rays // n_chunks)

    iter = range(0, n_rays, chunk_size_in_rays)
    if len(iter) >= tqdm_trigger_threshold:
        iter = tqdm.tqdm(iter)

    for start_idx in iter:
        end_idx = min(start_idx + chunk_size_in_rays, n_rays)
        ray_bundle_chunk = RayBundle(
            origins=ray_bundle.origins.reshape(batch_size, -1, 3)[:, start_idx:end_idx],
            directions=ray_bundle.directions.reshape(batch_size, -1, 3)[
                :, start_idx:end_idx
            ],
            lengths=ray_bundle.lengths.reshape(
                batch_size, math.prod(spatial_dim), n_pts_per_ray
            )[:, start_idx:end_idx],
            xys=ray_bundle.xys.reshape(batch_size, -1, 2)[:, start_idx:end_idx],
        )
        extra_args = kwargs.copy()
        for k, v in chunked_inputs.items():
            extra_args[k] = v.flatten(2)[:, :, start_idx:end_idx]
        yield [ray_bundle_chunk, *args], extra_args
