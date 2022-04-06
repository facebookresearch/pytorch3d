# @lint-ignore-every LICENSELINT
# Adapted from RenderingNetwork from IDR
# https://github.com/lioryariv/idr/
# Copyright (c) 2020 Lior Yariv

import logging
from typing import List, Tuple

import torch
from pytorch3d.implicitron.tools.config import enable_get_default_args
from pytorch3d.renderer.implicit import HarmonicEmbedding, RayBundle
from torch import nn


logger = logging.getLogger(__name__)


class RayNormalColoringNetwork(torch.nn.Module):
    def __init__(
        self,
        feature_vector_size: int = 3,
        mode: str = "idr",
        d_in: int = 9,
        d_out: int = 3,
        dims: Tuple[int, ...] = (512, 512, 512, 512),
        weight_norm: bool = True,
        n_harmonic_functions_dir: int = 0,
        pooled_feature_dim: int = 0,
    ) -> None:
        super().__init__()

        self.mode = mode
        self.output_dimensions = d_out
        dims_full: List[int] = [d_in + feature_vector_size] + list(dims) + [d_out]

        self.embedview_fn = None
        if n_harmonic_functions_dir > 0:
            self.embedview_fn = HarmonicEmbedding(
                n_harmonic_functions_dir, append_input=True
            )
            dims_full[0] += self.embedview_fn.get_output_dim() - 3

        if pooled_feature_dim > 0:
            logger.info("Pooled features in rendering network.")
            dims_full[0] += pooled_feature_dim

        self.num_layers = len(dims_full)

        layers = []
        for layer_idx in range(self.num_layers - 1):
            out_dim = dims_full[layer_idx + 1]
            lin = nn.Linear(dims_full[layer_idx], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            layers.append(lin)
        self.linear_layers = torch.nn.ModuleList(layers)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(
        self,
        feature_vectors: torch.Tensor,
        points,
        normals,
        ray_bundle: RayBundle,
        masks=None,
        pooling_fn=None,
    ):
        if masks is not None and not masks.any():
            return torch.zeros_like(normals)

        view_dirs = ray_bundle.directions
        if masks is not None:
            # in case of IDR, other outputs are passed here after applying the mask
            view_dirs = view_dirs.reshape(view_dirs.shape[0], -1, 3)[
                :, masks.reshape(-1)
            ]

        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.mode == "idr":
            rendering_input = torch.cat(
                [points, view_dirs, normals, feature_vectors], dim=-1
            )
        elif self.mode == "no_view_dir":
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == "no_normal":
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)
        else:
            raise ValueError(f"Unsupported rendering mode: {self.mode}")

        if pooling_fn is not None:
            featspool = pooling_fn(points[None])[0]
            rendering_input = torch.cat((rendering_input, featspool), dim=-1)

        x = rendering_input

        for layer_idx in range(self.num_layers - 1):
            x = self.linear_layers[layer_idx](x)

            if layer_idx < self.num_layers - 2:
                x = self.relu(x)

        x = self.tanh(x)
        return x


enable_get_default_args(RayNormalColoringNetwork)
