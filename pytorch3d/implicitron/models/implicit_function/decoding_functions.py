# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
This file contains
    - modules which get used by ImplicitFunction objects for decoding an embedding defined in
        space, e.g. to color or opacity.
    - DecoderFunctionBase and its subclasses, which wrap some of those modules, providing
        some such modules as an extension point which an ImplicitFunction object could use.
"""

import logging
from dataclasses import field

from enum import Enum
from typing import Dict, Optional, Tuple

import torch

from omegaconf import DictConfig

from pytorch3d.implicitron.tools.config import (
    Configurable,
    registry,
    ReplaceableBase,
    run_auto_creation,
)

logger = logging.getLogger(__name__)


class DecoderActivation(Enum):
    RELU = "relu"
    SOFTPLUS = "softplus"
    SIGMOID = "sigmoid"
    IDENTITY = "identity"


class DecoderFunctionBase(ReplaceableBase, torch.nn.Module):
    """
    Decoding function is a torch.nn.Module which takes the embedding of a location in
    space and transforms it into the required quantity (for example density and color).
    """

    def forward(
        self, features: torch.Tensor, z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): tensor of shape (batch, ..., num_in_features)
            z: optional tensor to append to parts of the decoding function
        Returns:
            decoded_features (torch.Tensor) : tensor of
                shape (batch, ..., num_out_features)
        """
        raise NotImplementedError()


@registry.register
class ElementwiseDecoder(DecoderFunctionBase):
    """
    Decoding function which scales the input, adds shift and then applies
    `relu`, `softplus`, `sigmoid` or nothing on its input:
    `result = operation(input * scale + shift)`

    Members:
        scale: a scalar with which input is multiplied before being shifted.
            Defaults to 1.
        shift: a scalar which is added to the scaled input before performing
            the operation. Defaults to 0.
        operation: which operation to perform on the transformed input. Options are:
            `RELU`, `SOFTPLUS`, `SIGMOID` or `IDENTITY`. Defaults to `IDENTITY`.
    """

    scale: float = 1
    shift: float = 0
    operation: DecoderActivation = DecoderActivation.IDENTITY

    def __post_init__(self):
        if self.operation not in [
            DecoderActivation.RELU,
            DecoderActivation.SOFTPLUS,
            DecoderActivation.SIGMOID,
            DecoderActivation.IDENTITY,
        ]:
            raise ValueError(
                "`operation` can only be `RELU`, `SOFTPLUS`, `SIGMOID` or `IDENTITY`."
            )

    def forward(
        self, features: torch.Tensor, z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        transfomed_input = features * self.scale + self.shift
        if self.operation == DecoderActivation.SOFTPLUS:
            return torch.nn.functional.softplus(transfomed_input)
        if self.operation == DecoderActivation.RELU:
            return torch.nn.functional.relu(transfomed_input)
        if self.operation == DecoderActivation.SIGMOID:
            return torch.nn.functional.sigmoid(transfomed_input)
        return transfomed_input


class MLPWithInputSkips(Configurable, torch.nn.Module):
    """
    Implements the multi-layer perceptron architecture of the Neural Radiance Field.

    As such, `MLPWithInputSkips` is a multi layer perceptron consisting
    of a sequence of linear layers with ReLU activations.

    Additionally, for a set of predefined layers `input_skips`, the forward pass
    appends a skip tensor `z` to the output of the preceding layer.

    Note that this follows the architecture described in the Supplementary
    Material (Fig. 7) of [1], for which keep the defaults for:
        - `last_layer_bias_init` to None
        - `last_activation` to "relu"
        - `use_xavier_init` to `true`

    If you want to use this as a part of the color prediction in TensoRF model set:
        - `last_layer_bias_init` to 0
        - `last_activation` to "sigmoid"
        - `use_xavier_init` to `False`

    References:
        [1] Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik
            and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng:
            NeRF: Representing Scenes as Neural Radiance Fields for View
            Synthesis, ECCV2020

    Members:
        n_layers: The number of linear layers of the MLP.
        input_dim: The number of channels of the input tensor.
        output_dim: The number of channels of the output.
        skip_dim: The number of channels of the tensor `z` appended when
            evaluating the skip layers.
        hidden_dim: The number of hidden units of the MLP.
        input_skips: The list of layer indices at which we append the skip
            tensor `z`.
        last_layer_bias_init: If set then all the biases in the last layer
            are initialized to that value.
        last_activation: Which activation to use in the last layer. Options are:
            "relu", "softplus", "sigmoid" and "identity". Default is "relu".
        use_xavier_init: If True uses xavier init for all linear layer weights.
            Otherwise the default PyTorch initialization is used. Default True.
    """

    n_layers: int = 8
    input_dim: int = 39
    output_dim: int = 256
    skip_dim: int = 39
    hidden_dim: int = 256
    input_skips: Tuple[int, ...] = (5,)
    skip_affine_trans: bool = False
    last_layer_bias_init: Optional[float] = None
    last_activation: DecoderActivation = DecoderActivation.RELU
    use_xavier_init: bool = True

    def __post_init__(self):
        try:
            last_activation = {
                DecoderActivation.RELU: torch.nn.ReLU(True),
                DecoderActivation.SOFTPLUS: torch.nn.Softplus(),
                DecoderActivation.SIGMOID: torch.nn.Sigmoid(),
                DecoderActivation.IDENTITY: torch.nn.Identity(),
            }[self.last_activation]
        except KeyError as e:
            raise ValueError(
                "`last_activation` can only be `RELU`,"
                " `SOFTPLUS`, `SIGMOID` or `IDENTITY`."
            ) from e

        layers = []
        skip_affine_layers = []
        for layeri in range(self.n_layers):
            dimin = self.hidden_dim if layeri > 0 else self.input_dim
            dimout = self.hidden_dim if layeri + 1 < self.n_layers else self.output_dim

            if layeri > 0 and layeri in self.input_skips:
                if self.skip_affine_trans:
                    skip_affine_layers.append(
                        self._make_affine_layer(self.skip_dim, self.hidden_dim)
                    )
                else:
                    dimin = self.hidden_dim + self.skip_dim

            linear = torch.nn.Linear(dimin, dimout)
            if self.use_xavier_init:
                _xavier_init(linear)
            if layeri == self.n_layers - 1 and self.last_layer_bias_init is not None:
                torch.nn.init.constant_(linear.bias, self.last_layer_bias_init)
            layers.append(
                torch.nn.Sequential(linear, torch.nn.ReLU(True))
                if not layeri + 1 < self.n_layers
                else torch.nn.Sequential(linear, last_activation)
            )
        self.mlp = torch.nn.ModuleList(layers)
        if self.skip_affine_trans:
            self.skip_affines = torch.nn.ModuleList(skip_affine_layers)
        self._input_skips = set(self.input_skips)
        self._skip_affine_trans = self.skip_affine_trans

    def _make_affine_layer(self, input_dim, hidden_dim):
        l1 = torch.nn.Linear(input_dim, hidden_dim * 2)
        l2 = torch.nn.Linear(hidden_dim * 2, hidden_dim * 2)
        if self.use_xavier_init:
            _xavier_init(l1)
            _xavier_init(l2)
        return torch.nn.Sequential(l1, torch.nn.ReLU(True), l2)

    def _apply_affine_layer(self, layer, x, z):
        mu_log_std = layer(z)
        mu, log_std = mu_log_std.split(mu_log_std.shape[-1] // 2, dim=-1)
        std = torch.nn.functional.softplus(log_std)
        return (x - mu) * std

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None):
        """
        Args:
            x: The input tensor of shape `(..., input_dim)`.
            z: The input skip tensor of shape `(..., skip_dim)` which is appended
                to layers whose indices are specified by `input_skips`.
        Returns:
            y: The output tensor of shape `(..., output_dim)`.
        """
        y = x
        if z is None:
            # if the skip tensor is None, we use `x` instead.
            z = x
        skipi = 0
        # pyre-fixme[6]: For 1st argument expected `Iterable[_T]` but got
        #  `Union[Tensor, Module]`.
        for li, layer in enumerate(self.mlp):
            # pyre-fixme[58]: `in` is not supported for right operand type
            #  `Union[Tensor, Module]`.
            if li in self._input_skips:
                if self._skip_affine_trans:
                    # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, ...
                    y = self._apply_affine_layer(self.skip_affines[skipi], y, z)
                else:
                    y = torch.cat((y, z), dim=-1)
                skipi += 1
            y = layer(y)
        return y


@registry.register
class MLPDecoder(DecoderFunctionBase):
    """
    Decoding function which uses `MLPWithIputSkips` to convert the embedding to output.
    The `input_dim` of the `network` is set from the value of `input_dim` member.

    Members:
        input_dim: dimension of input.
        param_groups: dictionary where keys are names of individual parameters
            or module members and values are the parameter group where the
            parameter/member will be sorted to. "self" key is used to denote the
            parameter group at the module level. Possible keys, including the "self" key
            do not have to be defined. By default all parameters are put into "default"
            parameter group and have the learning rate defined in the optimizer,
            it can be overridden at the:
                - module level with “self” key, all the parameters and child
                    module's parameters will be put to that parameter group
                - member level, which is the same as if the `param_groups` in that
                    member has key=“self” and value equal to that parameter group.
                    This is useful if members do not have `param_groups`, for
                    example torch.nn.Linear.
                - parameter level, parameter with the same name as the key
                    will be put to that parameter group.
        network_args: configuration for MLPWithInputSkips
    """

    input_dim: int = 3
    param_groups: Dict[str, str] = field(default_factory=lambda: {})
    # pyre-fixme[13]: Attribute `network` is never initialized.
    network: MLPWithInputSkips

    def __post_init__(self):
        run_auto_creation(self)

    def forward(
        self, features: torch.Tensor, z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.network(features, z)

    @classmethod
    def network_tweak_args(cls, type, args: DictConfig) -> None:
        """
        Special method to stop get_default_args exposing member's `input_dim`.
        """
        args.pop("input_dim", None)

    def create_network_impl(self, type, args: DictConfig) -> None:
        """
        Set the input dimension of the `network` to the input dimension of the
        decoding function.
        """
        self.network = MLPWithInputSkips(input_dim=self.input_dim, **args)


class TransformerWithInputSkips(torch.nn.Module):
    def __init__(
        self,
        n_layers: int = 8,
        input_dim: int = 39,
        output_dim: int = 256,
        skip_dim: int = 39,
        hidden_dim: int = 64,
        input_skips: Tuple[int, ...] = (5,),
        dim_down_factor: float = 1,
    ):
        """
        Args:
            n_layers: The number of linear layers of the MLP.
            input_dim: The number of channels of the input tensor.
            output_dim: The number of channels of the output.
            skip_dim: The number of channels of the tensor `z` appended when
                evaluating the skip layers.
            hidden_dim: The number of hidden units of the MLP.
            input_skips: The list of layer indices at which we append the skip
                tensor `z`.
        """
        super().__init__()

        self.first = torch.nn.Linear(input_dim, hidden_dim)
        _xavier_init(self.first)

        self.skip_linear = torch.nn.ModuleList()

        layers_pool, layers_ray = [], []
        dimout = 0
        for layeri in range(n_layers):
            dimin = int(round(hidden_dim / (dim_down_factor**layeri)))
            dimout = int(round(hidden_dim / (dim_down_factor ** (layeri + 1))))
            logger.info(f"Tr: {dimin} -> {dimout}")
            for _i, l in enumerate((layers_pool, layers_ray)):
                l.append(
                    TransformerEncoderLayer(
                        d_model=[dimin, dimout][_i],
                        nhead=4,
                        dim_feedforward=hidden_dim,
                        dropout=0.0,
                        d_model_out=dimout,
                    )
                )

            if layeri in input_skips:
                self.skip_linear.append(torch.nn.Linear(input_dim, dimin))

        self.last = torch.nn.Linear(dimout, output_dim)
        _xavier_init(self.last)

        # pyre-fixme[8]: Attribute has type `Tuple[ModuleList, ModuleList]`; used as
        #  `ModuleList`.
        self.layers_pool, self.layers_ray = (
            torch.nn.ModuleList(layers_pool),
            torch.nn.ModuleList(layers_ray),
        )
        self._input_skips = set(input_skips)

    def forward(
        self,
        x: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x: The input tensor of shape
                `(minibatch, n_pooled_feats, ..., n_ray_pts, input_dim)`.
            z: The input skip tensor of shape
                `(minibatch, n_pooled_feats, ..., n_ray_pts, skip_dim)`
                which is appended to layers whose indices are specified by `input_skips`.
        Returns:
            y: The output tensor of shape
                `(minibatch, 1, ..., n_ray_pts, input_dim)`.
        """

        if z is None:
            # if the skip tensor is None, we use `x` instead.
            z = x

        y = self.first(x)

        B, n_pool, n_rays, n_pts, dim = y.shape

        # y_p in n_pool, n_pts, B x n_rays x dim
        y_p = y.permute(1, 3, 0, 2, 4)

        skipi = 0
        dimh = dim
        for li, (layer_pool, layer_ray) in enumerate(
            zip(self.layers_pool, self.layers_ray)
        ):
            y_pool_attn = y_p.reshape(n_pool, n_pts * B * n_rays, dimh)
            if li in self._input_skips:
                z_skip = self.skip_linear[skipi](z)
                y_pool_attn = y_pool_attn + z_skip.permute(1, 3, 0, 2, 4).reshape(
                    n_pool, n_pts * B * n_rays, dimh
                )
                skipi += 1
            # n_pool x B*n_rays*n_pts x dim
            y_pool_attn, pool_attn = layer_pool(y_pool_attn, src_key_padding_mask=None)
            dimh = y_pool_attn.shape[-1]

            y_ray_attn = (
                y_pool_attn.view(n_pool, n_pts, B * n_rays, dimh)
                .permute(1, 0, 2, 3)
                .reshape(n_pts, n_pool * B * n_rays, dimh)
            )
            # n_pts x n_pool*B*n_rays x dim
            y_ray_attn, ray_attn = layer_ray(
                y_ray_attn,
                src_key_padding_mask=None,
            )

            y_p = y_ray_attn.view(n_pts, n_pool, B * n_rays, dimh).permute(1, 0, 2, 3)

        y = y_p.view(n_pool, n_pts, B, n_rays, dimh).permute(2, 0, 3, 1, 4)

        W = torch.softmax(y[..., :1], dim=1)
        y = (y * W).sum(dim=1)
        y = self.last(y)

        return y


class TransformerEncoderLayer(torch.nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, d_model_out=-1
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        d_model_out = d_model if d_model_out <= 0 else d_model_out
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model_out)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model_out)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.activation = torch.nn.functional.relu

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2, attn = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        d_out = src2.shape[-1]
        src = src[..., :d_out] + self.dropout2(src2)[..., :d_out]
        src = self.norm2(src)
        return src, attn


def _xavier_init(linear) -> None:
    """
    Performs the Xavier weight initialization of the linear layer `linear`.
    """
    torch.nn.init.xavier_uniform_(linear.weight.data)
