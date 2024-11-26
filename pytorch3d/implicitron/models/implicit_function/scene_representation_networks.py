# @lint-ignore-every LICENSELINT
# Adapted from https://github.com/vsitzmann/scene-representation-networks
# Copyright (c) 2019 Vincent Sitzmann

# pyre-unsafe
from typing import Any, cast, Optional, Tuple

import torch
from omegaconf import DictConfig
from pytorch3d.common.linear_with_repeat import LinearWithRepeat
from pytorch3d.implicitron.models.renderer.base import ImplicitronRayBundle
from pytorch3d.implicitron.third_party import hyperlayers, pytorch_prototyping
from pytorch3d.implicitron.tools.config import Configurable, registry, run_auto_creation
from pytorch3d.renderer import ray_bundle_to_ray_points
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.renderer.implicit import HarmonicEmbedding

from .base import ImplicitFunctionBase
from .utils import create_embeddings_for_implicit_function


def _kaiming_normal_init(module: torch.nn.Module) -> None:
    if isinstance(module, (torch.nn.Linear, LinearWithRepeat)):
        torch.nn.init.kaiming_normal_(
            module.weight, a=0.0, nonlinearity="relu", mode="fan_in"
        )


class SRNRaymarchFunction(Configurable, torch.nn.Module):
    n_harmonic_functions: int = 3  # 0 means raw 3D coord inputs
    n_hidden_units: int = 256
    n_layers: int = 2
    in_features: int = 3
    out_features: int = 256
    latent_dim: int = 0
    xyz_in_camera_coords: bool = False

    # The internal network can be set as an output of an SRNHyperNet.
    # Note that, in order to avoid Pytorch's automatic registering of the
    # raymarch_function module on construction, we input the network wrapped
    # as a 1-tuple.

    # raymarch_function should ideally be typed as Optional[Tuple[Callable]]
    # but Omegaconf.structured doesn't like that. TODO: revisit after new
    # release of omegaconf including https://github.com/omry/omegaconf/pull/749 .
    raymarch_function: Any = None

    def __post_init__(self):
        self._harmonic_embedding = HarmonicEmbedding(
            self.n_harmonic_functions, append_input=True
        )
        input_embedding_dim = (
            HarmonicEmbedding.get_output_dim_static(
                self.in_features,
                self.n_harmonic_functions,
                True,
            )
            + self.latent_dim
        )

        if self.raymarch_function is not None:
            self._net = self.raymarch_function[0]
        else:
            self._net = pytorch_prototyping.FCBlock(
                hidden_ch=self.n_hidden_units,
                num_hidden_layers=self.n_layers,
                in_features=input_embedding_dim,
                out_features=self.out_features,
            )

    def forward(
        self,
        ray_bundle: ImplicitronRayBundle,
        fun_viewpool=None,
        camera: Optional[CamerasBase] = None,
        global_code=None,
        **kwargs,
    ):
        """
        Args:
            ray_bundle: An ImplicitronRayBundle object containing the following variables:
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
            rays_colors: Set to None.
        """
        # We first convert the ray parametrizations to world
        # coordinates with `ray_bundle_to_ray_points`.
        # pyre-ignore[6]
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)

        embeds = create_embeddings_for_implicit_function(
            xyz_world=rays_points_world,
            # pyre-fixme[6]: For 2nd argument expected `Optional[(...) -> Any]` but
            #  got `Union[Tensor, Module]`.
            xyz_embedding_function=self._harmonic_embedding,
            global_code=global_code,
            fun_viewpool=fun_viewpool,
            xyz_in_camera_coords=self.xyz_in_camera_coords,
            camera=camera,
        )

        # Before running the network, we have to resize embeds to ndims=3,
        # otherwise the SRN layers consume huge amounts of memory.
        # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
        raymarch_features = self._net(
            embeds.view(embeds.shape[0], -1, embeds.shape[-1])
        )
        # raymarch_features.shape = [minibatch x ... x self.n_hidden_neurons_xyz]

        # NNs operate on the flattenned rays; reshaping to the correct spatial size
        raymarch_features = raymarch_features.reshape(*rays_points_world.shape[:-1], -1)

        return raymarch_features, None


class SRNPixelGenerator(Configurable, torch.nn.Module):
    n_harmonic_functions: int = 4
    n_hidden_units: int = 256
    n_hidden_units_color: int = 128
    n_layers: int = 2
    in_features: int = 256
    out_features: int = 3
    ray_dir_in_camera_coords: bool = False

    def __post_init__(self):
        self._harmonic_embedding = HarmonicEmbedding(
            self.n_harmonic_functions, append_input=True
        )
        self._net = pytorch_prototyping.FCBlock(
            hidden_ch=self.n_hidden_units,
            num_hidden_layers=self.n_layers,
            in_features=self.in_features,
            out_features=self.n_hidden_units,
        )
        self._density_layer = torch.nn.Linear(self.n_hidden_units, 1)
        self._density_layer.apply(_kaiming_normal_init)
        embedding_dim_dir = self._harmonic_embedding.get_output_dim(input_dims=3)
        self._color_layer = torch.nn.Sequential(
            LinearWithRepeat(
                self.n_hidden_units + embedding_dim_dir,
                self.n_hidden_units_color,
            ),
            torch.nn.LayerNorm([self.n_hidden_units_color]),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.n_hidden_units_color, self.out_features),
        )
        self._color_layer.apply(_kaiming_normal_init)

    # TODO: merge with NeuralRadianceFieldBase's _get_colors
    def _get_colors(self, features: torch.Tensor, rays_directions: torch.Tensor):
        """
        This function takes per-point `features` predicted by `self.net`
        and evaluates the color model in order to attach to each
        point a 3D vector of its RGB color.
        """
        # Normalize the ray_directions to unit l2 norm.
        rays_directions_normed = torch.nn.functional.normalize(rays_directions, dim=-1)
        # Obtain the harmonic embedding of the normalized ray directions.
        # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
        rays_embedding = self._harmonic_embedding(rays_directions_normed)
        # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
        return self._color_layer((features, rays_embedding))

    def forward(
        self,
        raymarch_features: torch.Tensor,
        ray_bundle: ImplicitronRayBundle,
        camera: Optional[CamerasBase] = None,
        **kwargs,
    ):
        """
        Args:
            raymarch_features: Features from the raymarching network of shape
                `(minibatch, ..., self.in_features)`
            ray_bundle: An ImplicitronRayBundle object containing the following variables:
                origins: A tensor of shape `(minibatch, ..., 3)` denoting the
                    origins of the sampling rays in world coords.
                directions: A tensor of shape `(minibatch, ..., 3)`
                    containing the direction vectors of sampling rays in world coords.
                lengths: A tensor of shape `(minibatch, ..., num_points_per_ray)`
                    containing the lengths at which the rays are sampled.

        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacitiy of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.
        """
        # raymarch_features.shape = [minibatch x ... x pts_per_ray x 3]
        # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
        features = self._net(raymarch_features)
        # features.shape = [minibatch x ... x self.n_hidden_units]

        if self.ray_dir_in_camera_coords:
            if camera is None:
                raise ValueError("Camera must be given if xyz_ray_dir_in_camera_coords")

            # pyre-fixme[58]: `@` is not supported for operand types `Tensor` and
            #  `Union[Tensor, Module]`.
            directions = ray_bundle.directions @ camera.R
        else:
            directions = ray_bundle.directions

        # NNs operate on the flattenned rays; reshaping to the correct spatial size
        features = features.reshape(*raymarch_features.shape[:-1], -1)

        # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
        raw_densities = self._density_layer(features)

        rays_colors = self._get_colors(features, directions)

        return raw_densities, rays_colors


class SRNRaymarchHyperNet(Configurable, torch.nn.Module):
    """
    This is a raymarching function which has a forward like SRNRaymarchFunction
    but instead of the weights being parameters of the module, they
    are the output of another network, the hypernet, which takes the global_code
    as input. All the dataclass members of SRNRaymarchFunction are here with the
    same meaning. In addition, there are members with names ending `_hypernet`
    which affect the hypernet.

    Because this class may be called repeatedly for the same global_code, the
    output of the hypernet is cached in self.cached_srn_raymarch_function.
    This member must be manually set to None whenever the global_code changes.
    """

    n_harmonic_functions: int = 3  # 0 means raw 3D coord inputs
    n_hidden_units: int = 256
    n_layers: int = 2
    n_hidden_units_hypernet: int = 256
    n_layers_hypernet: int = 1
    in_features: int = 3
    out_features: int = 256
    latent_dim_hypernet: int = 0
    latent_dim: int = 0
    xyz_in_camera_coords: bool = False

    def __post_init__(self):
        raymarch_input_embedding_dim = (
            HarmonicEmbedding.get_output_dim_static(
                self.in_features,
                self.n_harmonic_functions,
                True,
            )
            + self.latent_dim
        )

        self._hypernet = hyperlayers.HyperFC(
            hyper_in_ch=self.latent_dim_hypernet,
            hyper_num_hidden_layers=self.n_layers_hypernet,
            hyper_hidden_ch=self.n_hidden_units_hypernet,
            hidden_ch=self.n_hidden_units,
            num_hidden_layers=self.n_layers,
            in_ch=raymarch_input_embedding_dim,
            out_ch=self.n_hidden_units,
        )

        self.cached_srn_raymarch_function: Optional[Tuple[SRNRaymarchFunction]] = None

    def _run_hypernet(self, global_code: torch.Tensor) -> Tuple[SRNRaymarchFunction]:
        """
        Runs the hypernet and returns a 1-tuple containing the generated
        srn_raymarch_function.
        """

        # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
        net = self._hypernet(global_code)

        # use the hyper-net generated network to instantiate the raymarch module
        srn_raymarch_function = SRNRaymarchFunction(
            n_harmonic_functions=self.n_harmonic_functions,
            n_hidden_units=self.n_hidden_units,
            n_layers=self.n_layers,
            in_features=self.in_features,
            out_features=self.out_features,
            latent_dim=self.latent_dim,
            xyz_in_camera_coords=self.xyz_in_camera_coords,
            raymarch_function=(net,),
        )

        # move the generated raymarch function to the correct device
        srn_raymarch_function.to(global_code.device)

        return (srn_raymarch_function,)

    def forward(
        self,
        ray_bundle: ImplicitronRayBundle,
        fun_viewpool=None,
        camera: Optional[CamerasBase] = None,
        global_code=None,
        **kwargs,
    ):
        if global_code is None:
            raise ValueError("SRN Hypernetwork requires a non-trivial global code.")

        # The raymarching network is cached in case the function is called repeatedly
        # across LSTM iterations for the same global_code.
        if self.cached_srn_raymarch_function is None:
            # generate the raymarching network from the hypernet
            # pyre-fixme[16]: `SRNRaymarchHyperNet` has no attribute
            #  `cached_srn_raymarch_function`.
            self.cached_srn_raymarch_function = self._run_hypernet(global_code)
        (srn_raymarch_function,) = cast(
            Tuple[SRNRaymarchFunction], self.cached_srn_raymarch_function
        )

        return srn_raymarch_function(
            ray_bundle=ray_bundle,
            fun_viewpool=fun_viewpool,
            camera=camera,
            global_code=None,  # the hypernetwork takes the global code
        )


@registry.register
class SRNImplicitFunction(ImplicitFunctionBase, torch.nn.Module):
    latent_dim: int = 0
    # pyre-fixme[13]: Attribute `raymarch_function` is never initialized.
    raymarch_function: SRNRaymarchFunction
    # pyre-fixme[13]: Attribute `pixel_generator` is never initialized.
    pixel_generator: SRNPixelGenerator

    def __post_init__(self):
        run_auto_creation(self)

    def create_raymarch_function(self) -> None:
        self.raymarch_function = SRNRaymarchFunction(
            latent_dim=self.latent_dim,
            # pyre-fixme[32]: Keyword argument must be a mapping with string keys.
            **self.raymarch_function_args,
        )

    @classmethod
    def raymarch_function_tweak_args(cls, type, args: DictConfig) -> None:
        args.pop("latent_dim", None)

    def forward(
        self,
        *,
        ray_bundle: ImplicitronRayBundle,
        fun_viewpool=None,
        camera: Optional[CamerasBase] = None,
        global_code=None,
        raymarch_features: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        predict_colors = raymarch_features is not None
        if predict_colors:
            return self.pixel_generator(
                raymarch_features=raymarch_features,
                ray_bundle=ray_bundle,
                camera=camera,
                **kwargs,
            )
        else:
            return self.raymarch_function(
                ray_bundle=ray_bundle,
                fun_viewpool=fun_viewpool,
                camera=camera,
                global_code=global_code,
                **kwargs,
            )


@registry.register
class SRNHyperNetImplicitFunction(ImplicitFunctionBase, torch.nn.Module):
    """
    This implicit function uses a hypernetwork to generate the
    SRNRaymarchingFunction, and this is cached. Whenever the
    global_code changes, `on_bind_args` must be called to clear
    the cache.
    """

    latent_dim_hypernet: int = 0
    latent_dim: int = 0
    # pyre-fixme[13]: Attribute `hypernet` is never initialized.
    hypernet: SRNRaymarchHyperNet
    # pyre-fixme[13]: Attribute `pixel_generator` is never initialized.
    pixel_generator: SRNPixelGenerator

    def __post_init__(self):
        run_auto_creation(self)

    def create_hypernet(self) -> None:
        self.hypernet = SRNRaymarchHyperNet(
            latent_dim=self.latent_dim,
            latent_dim_hypernet=self.latent_dim_hypernet,
            # pyre-fixme[32]: Keyword argument must be a mapping with string keys.
            **self.hypernet_args,
        )

    @classmethod
    def hypernet_tweak_args(cls, type, args: DictConfig) -> None:
        args.pop("latent_dim", None)
        args.pop("latent_dim_hypernet", None)

    def forward(
        self,
        *,
        ray_bundle: ImplicitronRayBundle,
        fun_viewpool=None,
        camera: Optional[CamerasBase] = None,
        global_code=None,
        raymarch_features: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        predict_colors = raymarch_features is not None
        if predict_colors:
            return self.pixel_generator(
                raymarch_features=raymarch_features,
                ray_bundle=ray_bundle,
                camera=camera,
                **kwargs,
            )
        else:
            return self.hypernet(
                ray_bundle=ray_bundle,
                fun_viewpool=fun_viewpool,
                camera=camera,
                global_code=global_code,
                **kwargs,
            )

    def on_bind_args(self):
        """
        The global_code may have changed, so we reset the hypernet.
        """
        self.hypernet.cached_srn_raymarch_function = None
