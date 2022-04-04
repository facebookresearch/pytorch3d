# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List, Optional, Tuple

import torch
from pytorch3d.implicitron.tools.config import registry
from pytorch3d.renderer import RayBundle

from .base import BaseRenderer, EvaluationMode, ImplicitFunctionWrapper, RendererOutput


logger = logging.getLogger(__name__)


@registry.register
class LSTMRenderer(BaseRenderer, torch.nn.Module):
    """
    Implements the learnable LSTM raymarching function from SRN [1].

    Settings:
        num_raymarch_steps: The number of LSTM raymarching steps.
        init_depth: Initializes the bias of the last raymarching LSTM layer so that
            the farthest point from the camera reaches a far z-plane that
            lies `init_depth` units from the camera plane.
        init_depth_noise_std: The standard deviation of the random normal noise
            added to the initial depth of each marched ray.
        hidden_size: The dimensionality of the LSTM's hidden state.
        n_feature_channels: The number of feature channels returned by the
            implicit_function evaluated at each raymarching step.
        verbose: If `True`, logs raymarching debug info.

    References:
        [1] Sitzmann, V. and Zollhöfer, M. and Wetzstein, G..
            "Scene representation networks: Continuous 3d-structure-aware
            neural scene representations." NeurIPS 2019.
    """

    num_raymarch_steps: int = 10
    init_depth: float = 17.0
    init_depth_noise_std: float = 5e-4
    hidden_size: int = 16
    n_feature_channels: int = 256
    verbose: bool = False

    def __post_init__(self):
        super().__init__()
        self._lstm = torch.nn.LSTMCell(
            input_size=self.n_feature_channels,
            hidden_size=self.hidden_size,
        )
        self._lstm.apply(_init_recurrent_weights)
        _lstm_forget_gate_init(self._lstm)
        self._out_layer = torch.nn.Linear(self.hidden_size, 1)

        one_step = self.init_depth / self.num_raymarch_steps
        self._out_layer.bias.data.fill_(one_step)
        self._out_layer.weight.data.normal_(mean=0.0, std=1e-3)

    def forward(
        self,
        ray_bundle: RayBundle,
        implicit_functions: List[ImplicitFunctionWrapper],
        evaluation_mode: EvaluationMode = EvaluationMode.EVALUATION,
        **kwargs,
    ) -> RendererOutput:
        """

        Args:
            ray_bundle: A `RayBundle` object containing the parametrizations of the
                sampled rendering rays.
            implicit_functions: A single-element list of ImplicitFunctionWrappers which
                defines the implicit function to be used.
            evaluation_mode: one of EvaluationMode.TRAINING or
                EvaluationMode.EVALUATION which determines the settings used for
                rendering, specifically the RayPointRefiner and the density_noise_std.

        Returns:
            instance of RendererOutput
        """
        if len(implicit_functions) != 1:
            raise ValueError("LSTM renderer expects a single implicit function.")

        implicit_function = implicit_functions[0]

        if ray_bundle.lengths.shape[-1] != 1:
            raise ValueError(
                "LSTM renderer requires a ray-bundle with a single point per ray"
                + " which is the initial raymarching point."
            )

        # jitter the initial depths
        ray_bundle_t = ray_bundle._replace(
            lengths=ray_bundle.lengths
            + torch.randn_like(ray_bundle.lengths) * self.init_depth_noise_std
        )

        states: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None]
        signed_distance = torch.zeros_like(ray_bundle_t.lengths)
        raymarch_features = None
        for t in range(self.num_raymarch_steps + 1):
            # move signed_distance along each ray
            ray_bundle_t = ray_bundle_t._replace(
                lengths=ray_bundle_t.lengths + signed_distance
            )

            # eval the raymarching function
            raymarch_features, _ = implicit_function(
                ray_bundle_t,
                raymarch_features=None,
            )
            if self.verbose:
                msg = (
                    f"{t}: mu={float(signed_distance.mean()):1.2e};"
                    + f" std={float(signed_distance.std()):1.2e};"
                    # pyre-fixme[6]: Expected `Union[bytearray, bytes, str,
                    #  typing.SupportsFloat, typing_extensions.SupportsIndex]` for 1st
                    #  param but got `Tensor`.
                    + f" mu_d={float(ray_bundle_t.lengths.mean()):1.2e};"
                    # pyre-fixme[6]: Expected `Union[bytearray, bytes, str,
                    #  typing.SupportsFloat, typing_extensions.SupportsIndex]` for 1st
                    #  param but got `Tensor`.
                    + f" std_d={float(ray_bundle_t.lengths.std()):1.2e};"
                )
                logger.info(msg)
            if t == self.num_raymarch_steps:
                break

            # run the lstm marcher
            # pyre-fixme[29]: `Union[torch.Tensor, torch.nn.Module]` is not a function.
            state_h, state_c = self._lstm(
                raymarch_features.view(-1, raymarch_features.shape[-1]),
                states[-1],
            )
            if state_h.requires_grad:
                state_h.register_hook(lambda x: x.clamp(min=-10, max=10))
            # predict the next step size
            # pyre-fixme[29]: `Union[torch.Tensor, torch.nn.Module]` is not a function.
            signed_distance = self._out_layer(state_h).view(ray_bundle_t.lengths.shape)
            # log the lstm states
            states.append((state_h, state_c))

        opacity_logits, features = implicit_function(
            raymarch_features=raymarch_features,
            ray_bundle=ray_bundle_t,
        )
        mask = torch.sigmoid(opacity_logits)
        depth = ray_bundle_t.lengths * ray_bundle_t.directions.norm(
            dim=-1, keepdim=True
        )

        return RendererOutput(
            features=features[..., 0, :],
            depths=depth,
            masks=mask[..., 0, :],
        )


def _init_recurrent_weights(self) -> None:
    # copied from SRN codebase
    for m in self.modules():
        if type(m) in [torch.nn.GRU, torch.nn.LSTM, torch.nn.RNN]:
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    torch.nn.init.kaiming_normal_(param.data)
                elif "weight_hh" in name:
                    torch.nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    param.data.fill_(0)


def _lstm_forget_gate_init(lstm_layer) -> None:
    # copied from SRN codebase
    for name, parameter in lstm_layer.named_parameters():
        if "bias" not in name:
            continue
        n = parameter.size(0)
        start, end = n // 4, n // 2
        parameter.data[start:end].fill_(1.0)
