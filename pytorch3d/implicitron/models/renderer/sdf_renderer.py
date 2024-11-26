# @lint-ignore-every LICENSELINT
# Adapted from https://github.com/lioryariv/idr/blob/main/code/model/
#              implicit_differentiable_renderer.py
# Copyright (c) 2020 Lior Yariv

# pyre-unsafe
import functools
from typing import List, Optional, Tuple

import torch
from omegaconf import DictConfig
from pytorch3d.common.compat import prod
from pytorch3d.implicitron.models.renderer.base import ImplicitronRayBundle
from pytorch3d.implicitron.tools.config import (
    get_default_args_field,
    registry,
    run_auto_creation,
)
from pytorch3d.implicitron.tools.utils import evaluating

from .base import BaseRenderer, EvaluationMode, ImplicitFunctionWrapper, RendererOutput
from .ray_tracing import RayTracing
from .rgb_net import RayNormalColoringNetwork


@registry.register
class SignedDistanceFunctionRenderer(BaseRenderer, torch.nn.Module):
    render_features_dimensions: int = 3
    object_bounding_sphere: float = 1.0
    # pyre-fixme[13]: Attribute `ray_tracer` is never initialized.
    ray_tracer: RayTracing
    ray_normal_coloring_network_args: DictConfig = get_default_args_field(
        RayNormalColoringNetwork
    )
    bg_color: Tuple[float, ...] = (0.0,)
    soft_mask_alpha: float = 50.0

    def __post_init__(
        self,
    ):
        render_features_dimensions = self.render_features_dimensions
        if len(self.bg_color) not in [1, render_features_dimensions]:
            raise ValueError(
                f"Background color should have {render_features_dimensions} entries."
            )

        run_auto_creation(self)

        self.ray_normal_coloring_network_args["feature_vector_size"] = (
            render_features_dimensions
        )
        self._rgb_network = RayNormalColoringNetwork(
            **self.ray_normal_coloring_network_args
        )

        self.register_buffer("_bg_color", torch.tensor(self.bg_color), persistent=False)

    @classmethod
    def ray_tracer_tweak_args(cls, type, args: DictConfig) -> None:
        del args["object_bounding_sphere"]

    def create_ray_tracer(self) -> None:
        self.ray_tracer = RayTracing(
            # pyre-fixme[32]: Keyword argument must be a mapping with string keys.
            **self.ray_tracer_args,
            object_bounding_sphere=self.object_bounding_sphere,
        )

    def requires_object_mask(self) -> bool:
        return True

    def forward(
        self,
        ray_bundle: ImplicitronRayBundle,
        implicit_functions: List[ImplicitFunctionWrapper],
        evaluation_mode: EvaluationMode = EvaluationMode.EVALUATION,
        object_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> RendererOutput:
        """
        Args:
            ray_bundle: A `ImplicitronRayBundle` object containing the parametrizations of the
                sampled rendering rays.
            implicit_functions: single element list of ImplicitFunctionWrappers which
                defines the implicit function to be used.
            evaluation_mode: one of EvaluationMode.TRAINING or
                EvaluationMode.EVALUATION which determines the settings used for
                rendering.
            kwargs:
                object_mask: BoolTensor, denoting the silhouette of the object.
                    This is a required keyword argument for SignedDistanceFunctionRenderer

        Returns:
            instance of RendererOutput
        """
        if len(implicit_functions) != 1:
            raise ValueError(
                "SignedDistanceFunctionRenderer supports only single pass."
            )

        if object_mask is None:
            raise ValueError("Expected object_mask to be provided in the kwargs")
        object_mask = object_mask.bool()

        implicit_function = implicit_functions[0]
        implicit_function_gradient = functools.partial(_gradient, implicit_function)

        # object_mask: silhouette of the object
        batch_size, *spatial_size, _ = ray_bundle.lengths.shape
        num_pixels = prod(spatial_size)

        cam_loc = ray_bundle.origins.reshape(batch_size, -1, 3)
        ray_dirs = ray_bundle.directions.reshape(batch_size, -1, 3)
        object_mask = object_mask.reshape(batch_size, -1)

        with torch.no_grad(), evaluating(implicit_function):
            points, network_object_mask, dists = self.ray_tracer(
                sdf=lambda x: implicit_function(rays_points_world=x)[
                    :, 0
                ],  # TODO: get rid of this wrapper
                cam_loc=cam_loc,
                object_mask=object_mask,
                ray_directions=ray_dirs,
            )

        # TODO: below, cam_loc might as well be different
        depth = dists.reshape(batch_size, num_pixels, 1)
        points = (cam_loc + depth * ray_dirs).reshape(-1, 3)

        sdf_output = implicit_function(rays_points_world=points)[:, 0:1]
        # NOTE most of the intermediate variables are flattened for
        # no apparent reason (here and in the ray tracer)
        ray_dirs = ray_dirs.reshape(-1, 3)
        object_mask = object_mask.reshape(-1)

        # TODO: move it to loss computation
        if evaluation_mode == EvaluationMode.TRAINING:
            surface_mask = network_object_mask & object_mask
            surface_points = points[surface_mask]
            surface_dists = dists[surface_mask].unsqueeze(-1)
            surface_ray_dirs = ray_dirs[surface_mask]
            surface_cam_loc = cam_loc.reshape(-1, 3)[surface_mask]
            surface_output = sdf_output[surface_mask]
            N = surface_points.shape[0]

            # Sample points for the eikonal loss
            eik_bounding_box: float = self.object_bounding_sphere
            n_eik_points = batch_size * num_pixels // 2
            eikonal_points = torch.empty(
                n_eik_points,
                3,
                #  but got `Union[device, Tensor, Module]`.
                # pyre-fixme[6]: For 3rd argument expected `Union[None, int, str,
                #  device]` but got `Union[device, Tensor, Module]`.
                device=self._bg_color.device,
            ).uniform_(-eik_bounding_box, eik_bounding_box)
            eikonal_pixel_points = points.clone()
            eikonal_pixel_points = eikonal_pixel_points.detach()
            eikonal_points = torch.cat([eikonal_points, eikonal_pixel_points], 0)

            points_all = torch.cat([surface_points, eikonal_points], dim=0)

            output = implicit_function(rays_points_world=surface_points)
            surface_sdf_values = output[
                :N, 0:1
            ].detach()  # how is it different from sdf_output?

            g = implicit_function_gradient(points_all)
            surface_points_grad = g[:N, 0, :].clone().detach()
            grad_theta = g[N:, 0, :]

            differentiable_surface_points = _sample_network(
                surface_output,
                surface_sdf_values,
                surface_points_grad,
                surface_dists,
                surface_cam_loc,
                surface_ray_dirs,
            )

        else:
            surface_mask = network_object_mask
            differentiable_surface_points = points[surface_mask]
            grad_theta = None

        empty_render = differentiable_surface_points.shape[0] == 0
        features = implicit_function(rays_points_world=differentiable_surface_points)[
            None, :, 1:
        ]
        normals_full = features.new_zeros(
            batch_size, *spatial_size, 3, requires_grad=empty_render
        )
        render_full = (
            features.new_ones(
                batch_size,
                *spatial_size,
                self.render_features_dimensions,
                requires_grad=empty_render,
            )
            * self._bg_color
        )
        mask_full = features.new_ones(
            batch_size, *spatial_size, 1, requires_grad=empty_render
        )
        if not empty_render:
            normals = implicit_function_gradient(differentiable_surface_points)[
                None, :, 0, :
            ]
            normals_full.view(-1, 3)[surface_mask] = normals
            render_full.view(-1, self.render_features_dimensions)[surface_mask] = (
                # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
                self._rgb_network(
                    features,
                    differentiable_surface_points[None],
                    normals,
                    ray_bundle,
                    surface_mask[None, :, None],
                    pooling_fn=None,  # TODO
                )
            )
            mask_full.view(-1, 1)[~surface_mask] = torch.sigmoid(
                # pyre-fixme[6]: For 1st param expected `Tensor` but got `float`.
                -self.soft_mask_alpha * sdf_output[~surface_mask]
            )

        # scatter points with surface_mask
        points_full = ray_bundle.origins.detach().clone()
        points_full.view(-1, 3)[surface_mask] = differentiable_surface_points

        # TODO: it is sparse here but otherwise dense
        return RendererOutput(
            features=render_full,
            normals=normals_full,
            depths=depth.reshape(batch_size, *spatial_size, 1),
            masks=mask_full,  # this is a differentiable approximation, see (7) in the paper
            points=points_full,
            aux={"grad_theta": grad_theta},  # TODO: will be moved to eikonal loss
            # TODO: do we need sdf_output, grad_theta? Only for loss probably
        )


def _sample_network(
    surface_output,
    surface_sdf_values,
    surface_points_grad,
    surface_dists,
    surface_cam_loc,
    surface_ray_dirs,
    eps: float = 1e-4,
):
    # t -> t(theta)
    surface_ray_dirs_0 = surface_ray_dirs.detach()
    surface_points_dot = torch.bmm(
        surface_points_grad.view(-1, 1, 3), surface_ray_dirs_0.view(-1, 3, 1)
    ).squeeze(-1)
    dot_sign = (surface_points_dot >= 0).to(surface_points_dot) * 2 - 1
    surface_dists_theta = surface_dists - (surface_output - surface_sdf_values) / (
        surface_points_dot.abs().clip(eps) * dot_sign
    )

    # t(theta) -> x(theta,c,v)
    surface_points_theta_c_v = surface_cam_loc + surface_dists_theta * surface_ray_dirs

    return surface_points_theta_c_v


@torch.enable_grad()
def _gradient(module, rays_points_world):
    rays_points_world.requires_grad_(True)
    y = module.forward(rays_points_world=rays_points_world)[:, :1]
    d_output = torch.ones_like(y, requires_grad=False, device=y.device)
    gradients = torch.autograd.grad(
        outputs=y,
        inputs=rays_points_world,
        grad_outputs=d_output,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return gradients.unsqueeze(1)
