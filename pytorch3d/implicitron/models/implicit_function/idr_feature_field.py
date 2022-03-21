# @lint-ignore-every LICENSELINT
# Adapted from https://github.com/lioryariv/idr/blob/main/code/model/
#              implicit_differentiable_renderer.py
# Copyright (c) 2020 Lior Yariv
import math
from typing import Sequence

import torch
from pytorch3d.implicitron.tools.config import registry
from pytorch3d.renderer.implicit import HarmonicEmbedding
from torch import nn

from .base import ImplicitFunctionBase


@registry.register
class IdrFeatureField(ImplicitFunctionBase, torch.nn.Module):
    feature_vector_size: int = 3
    d_in: int = 3
    d_out: int = 1
    dims: Sequence[int] = (512, 512, 512, 512, 512, 512, 512, 512)
    geometric_init: bool = True
    bias: float = 1.0
    skip_in: Sequence[int] = ()
    weight_norm: bool = True
    n_harmonic_functions_xyz: int = 0
    pooled_feature_dim: int = 0
    encoding_dim: int = 0

    def __post_init__(self):
        super().__init__()

        dims = [self.d_in] + list(self.dims) + [self.d_out + self.feature_vector_size]

        self.embed_fn = None
        if self.n_harmonic_functions_xyz > 0:
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
                        mean=math.pi ** 0.5 / dims[layer_idx] ** 0.5,
                        std=0.0001,
                    )
                    torch.nn.init.constant_(lin.bias, -self.bias)
                elif self.n_harmonic_functions_xyz > 0 and layer_idx == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(
                        lin.weight[:, :3], 0.0, 2 ** 0.5 / out_dim ** 0.5
                    )
                elif self.n_harmonic_functions_xyz > 0 and layer_idx in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, 2 ** 0.5 / out_dim ** 0.5)
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3) :], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, 2 ** 0.5 / out_dim ** 0.5)

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
        # ray_bundle: RayBundle,
        rays_points_world: torch.Tensor,  # TODO: unify the APIs
        fun_viewpool=None,
        global_code=None,
    ):
        # this field only uses point locations
        # rays_points_world = ray_bundle_to_ray_points(ray_bundle)
        # rays_points_world.shape = [minibatch x ... x pts_per_ray x 3]

        if rays_points_world.numel() == 0 or (
            self.embed_fn is None and fun_viewpool is None and global_code is None
        ):
            return torch.tensor(
                [], device=rays_points_world.device, dtype=rays_points_world.dtype
            ).view(0, self.out_dim)

        embedding = None
        if self.embed_fn is not None:
            # pyre-fixme[29]: `Union[torch.Tensor, torch.nn.Module]` is not a function.
            embedding = self.embed_fn(rays_points_world)

        if fun_viewpool is not None:
            assert rays_points_world.ndim == 2
            pooled_feature = fun_viewpool(rays_points_world[None])
            # TODO: pooled features are 4D!
            embedding = torch.cat((embedding, pooled_feature), dim=-1)

        if global_code is not None:
            assert embedding.ndim == 2
            assert global_code.shape[0] == 1  # TODO: generalize to batches!
            # This will require changing raytracer code
            # embedding = embedding[None].expand(global_code.shape[0], *embedding.shape)
            embedding = torch.cat(
                (embedding, global_code[0, None, :].expand(*embedding.shape[:-1], -1)),
                dim=-1,
            )

        x = embedding
        for layer_idx in range(self.num_layers - 1):
            if layer_idx in self.skip_in:
                x = torch.cat([x, embedding], dim=-1) / 2 ** 0.5

            # pyre-fixme[29]: `Union[torch.Tensor, torch.nn.Module]` is not a function.
            x = self.linear_layers[layer_idx](x)

            if layer_idx < self.num_layers - 2:
                # pyre-fixme[29]: `Union[torch.Tensor, torch.nn.Module]` is not a function.
                x = self.softplus(x)

        return x  # TODO: unify the APIs
