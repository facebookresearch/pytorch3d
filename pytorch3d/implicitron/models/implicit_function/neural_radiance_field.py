# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import field
from typing import List, Optional

import torch
from pytorch3d.common.linear_with_repeat import LinearWithRepeat
from pytorch3d.implicitron.tools.config import registry
from pytorch3d.renderer import ray_bundle_to_ray_points, RayBundle
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.renderer.implicit import HarmonicEmbedding

from .base import ImplicitFunctionBase
from .utils import create_embeddings_for_implicit_function


logger = logging.getLogger(__name__)


class NeuralRadianceFieldBase(ImplicitFunctionBase, torch.nn.Module):
    n_harmonic_functions_xyz: int = 10
    n_harmonic_functions_dir: int = 4
    n_hidden_neurons_dir: int = 128
    latent_dim: int = 0
    input_xyz: bool = True
    xyz_ray_dir_in_camera_coords: bool = False
    color_dim: int = 3
    """
    Args:
        n_harmonic_functions_xyz: The number of harmonic functions
            used to form the harmonic embedding of 3D point locations.
        n_harmonic_functions_dir: The number of harmonic functions
            used to form the harmonic embedding of the ray directions.
        n_hidden_neurons_xyz: The number of hidden units in the
            fully connected layers of the MLP that accepts the 3D point
            locations and outputs the occupancy field with the intermediate
            features.
        n_hidden_neurons_dir: The number of hidden units in the
            fully connected layers of the MLP that accepts the intermediate
            features and ray directions and outputs the radiance field
            (per-point colors).
        n_layers_xyz: The number of layers of the MLP that outputs the
            occupancy field.
        append_xyz: The list of indices of the skip layers of the occupancy MLP.
    """

    def __post_init__(self):
        super().__init__()
        # The harmonic embedding layer converts input 3D coordinates
        # to a representation that is more suitable for
        # processing with a deep neural network.
        self.harmonic_embedding_xyz = HarmonicEmbedding(
            self.n_harmonic_functions_xyz, append_input=True
        )
        self.harmonic_embedding_dir = HarmonicEmbedding(
            self.n_harmonic_functions_dir, append_input=True
        )
        if not self.input_xyz and self.latent_dim <= 0:
            raise ValueError("The latent dimension has to be > 0 if xyz is not input!")

        embedding_dim_dir = self.harmonic_embedding_dir.get_output_dim()

        self.xyz_encoder = self._construct_xyz_encoder(
            input_dim=self.get_xyz_embedding_dim()
        )

        self.intermediate_linear = torch.nn.Linear(
            self.n_hidden_neurons_xyz, self.n_hidden_neurons_xyz
        )
        _xavier_init(self.intermediate_linear)

        self.density_layer = torch.nn.Linear(self.n_hidden_neurons_xyz, 1)
        _xavier_init(self.density_layer)

        # Zero the bias of the density layer to avoid
        # a completely transparent initialization.
        self.density_layer.bias.data[:] = 0.0  # fixme: Sometimes this is not enough

        self.color_layer = torch.nn.Sequential(
            LinearWithRepeat(
                self.n_hidden_neurons_xyz + embedding_dim_dir, self.n_hidden_neurons_dir
            ),
            torch.nn.ReLU(True),
            torch.nn.Linear(self.n_hidden_neurons_dir, self.color_dim),
            torch.nn.Sigmoid(),
        )

    def get_xyz_embedding_dim(self):
        return (
            self.harmonic_embedding_xyz.get_output_dim() * int(self.input_xyz)
            + self.latent_dim
        )

    def _construct_xyz_encoder(self, input_dim: int):
        raise NotImplementedError()

    def _get_colors(self, features: torch.Tensor, rays_directions: torch.Tensor):
        """
        This function takes per-point `features` predicted by `self.xyz_encoder`
        and evaluates the color model in order to attach to each
        point a 3D vector of its RGB color.
        """
        # Normalize the ray_directions to unit l2 norm.
        rays_directions_normed = torch.nn.functional.normalize(rays_directions, dim=-1)
        # Obtain the harmonic embedding of the normalized ray directions.
        # pyre-fixme[29]: `Union[torch.Tensor, torch.nn.Module]` is not a function.
        rays_embedding = self.harmonic_embedding_dir(rays_directions_normed)

        # pyre-fixme[29]: `Union[torch.Tensor, torch.nn.Module]` is not a function.
        return self.color_layer((self.intermediate_linear(features), rays_embedding))

    @staticmethod
    def allows_multiple_passes() -> bool:
        """
        Returns True as this implicit function allows
        multiple passes. Overridden from ImplicitFunctionBase.
        """
        return True

    def forward(
        self,
        ray_bundle: RayBundle,
        fun_viewpool=None,
        camera: Optional[CamerasBase] = None,
        global_code=None,
        **kwargs,
    ):
        """
        The forward function accepts the parametrizations of
        3D points sampled along projection rays. The forward
        pass is responsible for attaching a 3D vector
        and a 1D scalar representing the point's
        RGB color and opacity respectively.

        Args:
            ray_bundle: A RayBundle object containing the following variables:
                origins: A tensor of shape `(minibatch, ..., 3)` denoting the
                    origins of the sampling rays in world coords.
                directions: A tensor of shape `(minibatch, ..., 3)`
                    containing the direction vectors of sampling rays in world coords.
                lengths: A tensor of shape `(minibatch, ..., num_points_per_ray)`
                    containing the lengths at which the rays are sampled.
            fun_viewpool: an optional callback with the signature
                    fun_fiewpool(points) -> pooled_features
                where points is a [N_TGT x N x 3] tensor of world coords,
                and pooled_features is a [N_TGT x ... x N_SRC x latent_dim] tensor
                of the features pooled from the context images.

        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacitiy of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.
        """
        # We first convert the ray parametrizations to world
        # coordinates with `ray_bundle_to_ray_points`.
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)
        # rays_points_world.shape = [minibatch x ... x pts_per_ray x 3]

        embeds = create_embeddings_for_implicit_function(
            xyz_world=ray_bundle_to_ray_points(ray_bundle),
            # pyre-fixme[6]: Expected `Optional[typing.Callable[..., typing.Any]]`
            #  for 2nd param but got `Union[None, torch.Tensor, torch.nn.Module]`.
            xyz_embedding_function=self.harmonic_embedding_xyz
            if self.input_xyz
            else None,
            global_code=global_code,
            fun_viewpool=fun_viewpool,
            xyz_in_camera_coords=self.xyz_ray_dir_in_camera_coords,
            camera=camera,
        )

        # embeds.shape = [minibatch x n_src x n_rays x n_pts x self.n_harmonic_functions*6+3]
        # pyre-fixme[29]: `Union[torch.Tensor, torch.nn.Module]` is not a function.
        features = self.xyz_encoder(embeds)
        # features.shape = [minibatch x ... x self.n_hidden_neurons_xyz]
        # NNs operate on the flattenned rays; reshaping to the correct spatial size
        # TODO: maybe make the transformer work on non-flattened tensors to avoid this reshape
        features = features.reshape(*rays_points_world.shape[:-1], -1)

        # pyre-fixme[29]: `Union[torch.Tensor, torch.nn.Module]` is not a function.
        raw_densities = self.density_layer(features)
        # raw_densities.shape = [minibatch x ... x 1] in [0-1]

        if self.xyz_ray_dir_in_camera_coords:
            if camera is None:
                raise ValueError("Camera must be given if xyz_ray_dir_in_camera_coords")

            directions = ray_bundle.directions @ camera.R
        else:
            directions = ray_bundle.directions

        rays_colors = self._get_colors(features, directions)
        # rays_colors.shape = [minibatch x ... x 3] in [0-1]

        return raw_densities, rays_colors, {}


@registry.register
class NeuralRadianceFieldImplicitFunction(NeuralRadianceFieldBase):
    transformer_dim_down_factor: float = 1.0
    n_hidden_neurons_xyz: int = 256
    n_layers_xyz: int = 8
    append_xyz: List[int] = field(default_factory=lambda: [5])

    def _construct_xyz_encoder(self, input_dim: int):
        return MLPWithInputSkips(
            self.n_layers_xyz,
            input_dim,
            self.n_hidden_neurons_xyz,
            input_dim,
            self.n_hidden_neurons_xyz,
            input_skips=self.append_xyz,
        )


@registry.register
class NeRFormerImplicitFunction(NeuralRadianceFieldBase):
    transformer_dim_down_factor: float = 2.0
    n_hidden_neurons_xyz: int = 80
    n_layers_xyz: int = 2
    append_xyz: List[int] = field(default_factory=lambda: [1])

    def _construct_xyz_encoder(self, input_dim: int):
        return TransformerWithInputSkips(
            self.n_layers_xyz,
            input_dim,
            self.n_hidden_neurons_xyz,
            input_dim,
            self.n_hidden_neurons_xyz,
            input_skips=self.append_xyz,
            dim_down_factor=self.transformer_dim_down_factor,
        )

    @staticmethod
    def requires_pooling_without_aggregation() -> bool:
        """
        Returns True as this implicit function needs
        pooling without aggregation. Overridden from ImplicitFunctionBase.
        """
        return True


class MLPWithInputSkips(torch.nn.Module):
    """
    Implements the multi-layer perceptron architecture of the Neural Radiance Field.

    As such, `MLPWithInputSkips` is a multi layer perceptron consisting
    of a sequence of linear layers with ReLU activations.

    Additionally, for a set of predefined layers `input_skips`, the forward pass
    appends a skip tensor `z` to the output of the preceding layer.

    Note that this follows the architecture described in the Supplementary
    Material (Fig. 7) of [1].

    References:
        [1] Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik
            and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng:
            NeRF: Representing Scenes as Neural Radiance Fields for View
            Synthesis, ECCV2020
    """

    def _make_affine_layer(self, input_dim, hidden_dim):
        l1 = torch.nn.Linear(input_dim, hidden_dim * 2)
        l2 = torch.nn.Linear(hidden_dim * 2, hidden_dim * 2)
        _xavier_init(l1)
        _xavier_init(l2)
        return torch.nn.Sequential(l1, torch.nn.ReLU(True), l2)

    def _apply_affine_layer(self, layer, x, z):
        mu_log_std = layer(z)
        mu, log_std = mu_log_std.split(mu_log_std.shape[-1] // 2, dim=-1)
        std = torch.nn.functional.softplus(log_std)
        return (x - mu) * std

    def __init__(
        self,
        n_layers: int = 8,
        input_dim: int = 39,
        output_dim: int = 256,
        skip_dim: int = 39,
        hidden_dim: int = 256,
        input_skips: List[int] = [5],
        skip_affine_trans: bool = False,
        no_last_relu=False,
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
        layers = []
        skip_affine_layers = []
        for layeri in range(n_layers):
            dimin = hidden_dim if layeri > 0 else input_dim
            dimout = hidden_dim if layeri + 1 < n_layers else output_dim

            if layeri > 0 and layeri in input_skips:
                if skip_affine_trans:
                    skip_affine_layers.append(
                        self._make_affine_layer(skip_dim, hidden_dim)
                    )
                else:
                    dimin = hidden_dim + skip_dim

            linear = torch.nn.Linear(dimin, dimout)
            _xavier_init(linear)
            layers.append(
                torch.nn.Sequential(linear, torch.nn.ReLU(True))
                if not no_last_relu or layeri + 1 < n_layers
                else linear
            )
        self.mlp = torch.nn.ModuleList(layers)
        if skip_affine_trans:
            self.skip_affines = torch.nn.ModuleList(skip_affine_layers)
        self._input_skips = set(input_skips)
        self._skip_affine_trans = skip_affine_trans

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
        for li, layer in enumerate(self.mlp):
            if li in self._input_skips:
                if self._skip_affine_trans:
                    y = self._apply_affine_layer(self.skip_affines[skipi], y, z)
                else:
                    y = torch.cat((y, z), dim=-1)
                skipi += 1
            y = layer(y)
        return y


class TransformerWithInputSkips(torch.nn.Module):
    def __init__(
        self,
        n_layers: int = 8,
        input_dim: int = 39,
        output_dim: int = 256,
        skip_dim: int = 39,
        hidden_dim: int = 64,
        input_skips: List[int] = [5],
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
            dimin = int(round(hidden_dim / (dim_down_factor ** layeri)))
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
