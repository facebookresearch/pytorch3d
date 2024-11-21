# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import logging
from typing import Optional, Tuple

import torch
from pytorch3d.common.linear_with_repeat import LinearWithRepeat
from pytorch3d.implicitron.models.renderer.base import (
    conical_frustum_to_gaussian,
    ImplicitronRayBundle,
)
from pytorch3d.implicitron.tools.config import expand_args_fields, registry
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.renderer.implicit import HarmonicEmbedding
from pytorch3d.renderer.implicit.utils import ray_bundle_to_ray_points

from .base import ImplicitFunctionBase

from .decoding_functions import (  # noqa
    _xavier_init,
    MLPWithInputSkips,
    TransformerWithInputSkips,
)
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
    use_integrated_positional_encoding: bool = False
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
        use_integrated_positional_encoding: If True, use integrated positional enoding
            as defined in `MIP-NeRF <https://arxiv.org/abs/2103.13415>`_.
            If False, use the classical harmonic embedding
            defined in `NeRF <https://arxiv.org/abs/2003.08934>`_.
    """

    def __post_init__(self):
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
        # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
        rays_embedding = self.harmonic_embedding_dir(rays_directions_normed)

        # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
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
        *,
        ray_bundle: ImplicitronRayBundle,
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
            ray_bundle: An ImplicitronRayBundle object containing the following variables:
                origins: A tensor of shape `(minibatch, ..., 3)` denoting the
                    origins of the sampling rays in world coords.
                directions: A tensor of shape `(minibatch, ..., 3)`
                    containing the direction vectors of sampling rays in world coords.
                lengths: A tensor of shape `(minibatch, ..., num_points_per_ray)`
                    containing the lengths at which the rays are sampled.
                bins: An optional tensor of shape `(minibatch,..., num_points_per_ray + 1)`
                    containing the bins at which the rays are sampled. In this case
                    lengths is equal to the midpoints of bins.

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

        Raises:
            ValueError: If `use_integrated_positional_encoding` is True and
                `ray_bundle.bins` is None.
        """
        if self.use_integrated_positional_encoding and ray_bundle.bins is None:
            raise ValueError(
                "When use_integrated_positional_encoding is True, ray_bundle.bins must be set."
                "Have you set to True `AbstractMaskRaySampler.use_bins_for_ray_sampling`?"
            )

        rays_points_world, diag_cov = (
            conical_frustum_to_gaussian(ray_bundle)
            if self.use_integrated_positional_encoding
            else (ray_bundle_to_ray_points(ray_bundle), None)  # pyre-ignore
        )
        # rays_points_world.shape = [minibatch x ... x pts_per_ray x 3]

        embeds = create_embeddings_for_implicit_function(
            xyz_world=rays_points_world,
            #  for 2nd param but got `Union[None, torch.Tensor, torch.nn.Module]`.
            # pyre-fixme[6]: For 2nd argument expected `Optional[(...) -> Any]` but
            #  got `Union[None, Tensor, Module]`.
            xyz_embedding_function=(
                self.harmonic_embedding_xyz if self.input_xyz else None
            ),
            global_code=global_code,
            fun_viewpool=fun_viewpool,
            xyz_in_camera_coords=self.xyz_ray_dir_in_camera_coords,
            camera=camera,
            diag_cov=diag_cov,
        )

        # embeds.shape = [minibatch x n_src x n_rays x n_pts x self.n_harmonic_functions*6+3]
        # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
        features = self.xyz_encoder(embeds)
        # features.shape = [minibatch x ... x self.n_hidden_neurons_xyz]
        # NNs operate on the flattenned rays; reshaping to the correct spatial size
        # TODO: maybe make the transformer work on non-flattened tensors to avoid this reshape
        features = features.reshape(*rays_points_world.shape[:-1], -1)

        # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
        raw_densities = self.density_layer(features)
        # raw_densities.shape = [minibatch x ... x 1] in [0-1]

        if self.xyz_ray_dir_in_camera_coords:
            if camera is None:
                raise ValueError("Camera must be given if xyz_ray_dir_in_camera_coords")

            # pyre-fixme[58]: `@` is not supported for operand types `Tensor` and
            #  `Union[Tensor, Module]`.
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
    append_xyz: Tuple[int, ...] = (5,)

    def _construct_xyz_encoder(self, input_dim: int):
        expand_args_fields(MLPWithInputSkips)
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
    append_xyz: Tuple[int, ...] = (1,)

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
