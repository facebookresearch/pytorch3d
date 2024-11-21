# @lint-ignore-every LICENSELINT
# Adapted from https://github.com/lioryariv/idr/blob/main/code/model/
#              implicit_differentiable_renderer.py
# Copyright (c) 2020 Lior Yariv

# pyre-unsafe
import math
from typing import Optional, Tuple

import torch
from pytorch3d.implicitron.models.renderer.base import ImplicitronRayBundle
from pytorch3d.implicitron.tools.config import registry
from pytorch3d.renderer.implicit import HarmonicEmbedding

from torch import nn

from .base import ImplicitFunctionBase
from .utils import get_rays_points_world


@registry.register
class IdrFeatureField(ImplicitFunctionBase, torch.nn.Module):
    """
    Implicit function as used in http://github.com/lioryariv/idr.

    Members:
        d_in: dimension of the input point.
        n_harmonic_functions_xyz: If -1, do not embed the point.
            If >=0, use a harmonic embedding with this number of
            harmonic functions. (The harmonic embedding includes the input
            itself, so a value of 0 means the point is used but without
            any harmonic functions.)
        d_out and feature_vector_size: Sum of these is the output
            dimension. This implicit function thus returns a concatenation
            of `d_out` signed distance function values and `feature_vector_size`
            features (such as colors). When used in `GenericModel`,
            `feature_vector_size` corresponds is automatically set to
            `render_features_dimensions`.
        dims: list of hidden layer sizes.
        geometric_init: whether to use custom weight initialization
            in linear layers. If False, pytorch default (uniform sampling)
            is used.
        bias: if geometric_init=True, initial value for bias subtracted
            in the last layer.
        skip_in: List of indices of layers that receive as input the initial
            value concatenated with the output of the previous layers.
        weight_norm: whether to apply weight normalization to each layer.
        pooled_feature_dim: If view pooling is in use (provided as
            fun_viewpool to forward()) this must be its number of features.
            Otherwise this must be set to 0. (If used from GenericModel,
            this config value will be overridden automatically.)
        encoding_dim: If global coding is in use (provided as global_code
            to forward()) this must be its number of featuress.
            Otherwise this must be set to 0. (If used from GenericModel,
            this config value will be overridden automatically.)
    """

    feature_vector_size: int = 3
    d_in: int = 3
    d_out: int = 1
    dims: Tuple[int, ...] = (512, 512, 512, 512, 512, 512, 512, 512)
    geometric_init: bool = True
    bias: float = 1.0
    skip_in: Tuple[int, ...] = ()
    weight_norm: bool = True
    n_harmonic_functions_xyz: int = 0
    pooled_feature_dim: int = 0
    encoding_dim: int = 0

    def __post_init__(self):
        dims = [self.d_in] + list(self.dims) + [self.d_out + self.feature_vector_size]

        self.embed_fn = None
        if self.n_harmonic_functions_xyz >= 0:
            self.embed_fn = HarmonicEmbedding(
                self.n_harmonic_functions_xyz, append_input=True
            )
            dims[0] = self.embed_fn.get_output_dim()
        if self.pooled_feature_dim > 0:
            dims[0] += self.pooled_feature_dim
        if self.encoding_dim > 0:
            dims[0] += self.encoding_dim

        self.num_layers = len(dims)

        out_dim = 0
        layers = []
        for layer_idx in range(self.num_layers - 1):
            if layer_idx + 1 in self.skip_in:
                out_dim = dims[layer_idx + 1] - dims[0]
            else:
                out_dim = dims[layer_idx + 1]

            lin = nn.Linear(dims[layer_idx], out_dim)

            if self.geometric_init:
                if layer_idx == self.num_layers - 2:
                    torch.nn.init.normal_(
                        lin.weight,
                        mean=math.pi**0.5 / dims[layer_idx] ** 0.5,
                        std=0.0001,
                    )
                    torch.nn.init.constant_(lin.bias, -self.bias)
                elif self.n_harmonic_functions_xyz >= 0 and layer_idx == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, 2**0.5 / out_dim**0.5)
                elif self.n_harmonic_functions_xyz >= 0 and layer_idx in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, 2**0.5 / out_dim**0.5)
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3) :], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, 2**0.5 / out_dim**0.5)

            if self.weight_norm:
                lin = nn.utils.weight_norm(lin)

            layers.append(lin)

        self.linear_layers = torch.nn.ModuleList(layers)
        self.out_dim = out_dim
        self.softplus = nn.Softplus(beta=100)

    # pyre-fixme[14]: `forward` overrides method defined in `ImplicitFunctionBase`
    #  inconsistently.
    def forward(
        self,
        *,
        ray_bundle: Optional[ImplicitronRayBundle] = None,
        rays_points_world: Optional[torch.Tensor] = None,
        fun_viewpool=None,
        global_code=None,
        **kwargs,
    ):
        # this field only uses point locations
        # rays_points_world.shape = [minibatch x ... x pts_per_ray x 3]
        rays_points_world = get_rays_points_world(ray_bundle, rays_points_world)

        if rays_points_world.numel() == 0 or (
            self.embed_fn is None and fun_viewpool is None and global_code is None
        ):
            return torch.tensor(
                [],
                device=rays_points_world.device,
                dtype=rays_points_world.dtype,
                # pyre-fixme[6]: For 2nd argument expected `Union[int, SymInt]` but got
                #  `Union[Module, Tensor]`.
            ).view(0, self.out_dim)

        embeddings = []
        if self.embed_fn is not None:
            # pyre-fixme[29]: `Union[Module, Tensor]` is not a function.
            embeddings.append(self.embed_fn(rays_points_world))

        if fun_viewpool is not None:
            assert rays_points_world.ndim == 2
            pooled_feature = fun_viewpool(rays_points_world[None])
            # TODO: pooled features are 4D!
            embeddings.append(pooled_feature)

        if global_code is not None:
            assert global_code.shape[0] == 1  # TODO: generalize to batches!
            # This will require changing raytracer code
            # embedding = embedding[None].expand(global_code.shape[0], *embedding.shape)
            embeddings.append(
                global_code[0, None, :].expand(rays_points_world.shape[0], -1)
            )

        embedding = torch.cat(embeddings, dim=-1)
        x = embedding
        # pyre-fixme[29]: `Union[(self: TensorBase, other: Union[bool, complex,
        #  float, int, Tensor]) -> Tensor, Module, Tensor]` is not a function.
        for layer_idx in range(self.num_layers - 1):
            if layer_idx in self.skip_in:
                x = torch.cat([x, embedding], dim=-1) / 2**0.5

            # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, slice[An...
            x = self.linear_layers[layer_idx](x)

            # pyre-fixme[29]: `Union[(self: TensorBase, other: Union[bool, complex,
            #  float, int, Tensor]) -> Tensor, Module, Tensor]` is not a function.
            if layer_idx < self.num_layers - 2:
                # pyre-fixme[29]: `Union[Module, Tensor]` is not a function.
                x = self.softplus(x)

        return x
