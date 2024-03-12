# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Callable, Tuple, Union

import torch

from ...ops.utils import eyes
from ...structures import Volumes
from ...transforms import Transform3d
from ..cameras import CamerasBase
from .raysampling import HeterogeneousRayBundle, RayBundle
from .utils import _validate_ray_bundle_variables, ray_bundle_variables_to_ray_points


# The implicit renderer class should be initialized with a
# function for raysampling and a function for raymarching.

# During the forward pass:
# 1) The raysampler:
#     - samples rays from input cameras
#     - transforms the rays to world coordinates
# 2) The volumetric_function (which is a callable argument of the forward pass)
#    evaluates ray_densities and ray_features at the sampled ray-points.
# 3) The raymarcher takes ray_densities and ray_features and uses a raymarching
#    algorithm to render each ray.


class ImplicitRenderer(torch.nn.Module):
    """
    A class for rendering a batch of implicit surfaces. The class should
    be initialized with a raysampler and raymarcher class which both have
    to be a `Callable`.

    VOLUMETRIC_FUNCTION

    The `forward` function of the renderer accepts as input the rendering cameras
    as well as the `volumetric_function` `Callable`, which defines a field of opacity
    and feature vectors over the 3D domain of the scene.

    A standard `volumetric_function` has the following signature::

        def volumetric_function(
            ray_bundle: Union[RayBundle, HeterogeneousRayBundle],
            **kwargs,
        ) -> Tuple[torch.Tensor, torch.Tensor]

    With the following arguments:
        `ray_bundle`: A RayBundle or HeterogeneousRayBundle object
            containing the following variables:

            `origins`: A tensor of shape `(minibatch, ..., 3)` denoting
                the origins of the rendering rays.
            `directions`: A tensor of shape `(minibatch, ..., 3)`
                containing the direction vectors of rendering rays.
            `lengths`: A tensor of shape
                `(minibatch, ..., num_points_per_ray)`containing the
                lengths at which the ray points are sampled.
            `xys`: A tensor of shape
                `(minibatch, ..., 2)` containing the
                xy locations of each ray's pixel in the screen space.
    Calling `volumetric_function` then returns the following:
        `rays_densities`: A tensor of shape
            `(minibatch, ..., num_points_per_ray, opacity_dim)` containing
            the an opacity vector for each ray point.
        `rays_features`: A tensor of shape
            `(minibatch, ..., num_points_per_ray, feature_dim)` containing
            the an feature vector for each ray point.

    Note that, in order to increase flexibility of the API, we allow multiple
    other arguments to enter the volumetric function via additional
    (optional) keyword arguments `**kwargs`.
    A typical use-case is passing a `CamerasBase` object as an additional
    keyword argument, which can allow the volumetric function to adjust its
    outputs based on the directions of the projection rays.

    Example:
        A simple volumetric function of a 0-centered
        RGB sphere with a unit diameter is defined as follows::

            def volumetric_function(
                ray_bundle: Union[RayBundle, HeterogeneousRayBundle],
                **kwargs,
            ) -> Tuple[torch.Tensor, torch.Tensor]:

                # first convert the ray origins, directions and lengths
                # to 3D ray point locations in world coords
                rays_points_world = ray_bundle_to_ray_points(ray_bundle)

                # set the densities as an inverse sigmoid of the
                # ray point distance from the sphere centroid
                rays_densities = torch.sigmoid(
                    -100.0 * rays_points_world.norm(dim=-1, keepdim=True)
                )

                # set the ray features to RGB colors proportional
                # to the 3D location of the projection of ray points
                # on the sphere surface
                rays_features = torch.nn.functional.normalize(
                    rays_points_world, dim=-1
                ) * 0.5 + 0.5

                return rays_densities, rays_features

    """

    def __init__(self, raysampler: Callable, raymarcher: Callable) -> None:
        """
        Args:
            raysampler: A `Callable` that takes as input scene cameras
                (an instance of `CamerasBase`) and returns a
                RayBundle or HeterogeneousRayBundle, that
                describes the rays emitted from the cameras.
            raymarcher: A `Callable` that receives the response of the
                `volumetric_function` (an input to `self.forward`) evaluated
                along the sampled rays, and renders the rays with a
                ray-marching algorithm.
        """
        super().__init__()

        if not callable(raysampler):
            raise ValueError('"raysampler" has to be a "Callable" object.')
        if not callable(raymarcher):
            raise ValueError('"raymarcher" has to be a "Callable" object.')

        self.raysampler = raysampler
        self.raymarcher = raymarcher

    def forward(
        self, cameras: CamerasBase, volumetric_function: Callable, **kwargs
    ) -> Tuple[torch.Tensor, Union[RayBundle, HeterogeneousRayBundle]]:
        """
        Render a batch of images using a volumetric function
        represented as a callable (e.g. a Pytorch module).

        Args:
            cameras: A batch of cameras that render the scene. A `self.raysampler`
                takes the cameras as input and samples rays that pass through the
                domain of the volumetric function.
            volumetric_function: A `Callable` that accepts the parametrizations
                of the rendering rays and returns the densities and features
                at the respective 3D of the rendering rays. Please refer to
                the main class documentation for details.

        Returns:
            images: A tensor of shape `(minibatch, ..., feature_dim + opacity_dim)`
                containing the result of the rendering.
            ray_bundle: A `Union[RayBundle, HeterogeneousRayBundle]` containing
                the parametrizations of the sampled rendering rays.
        """

        if not callable(volumetric_function):
            raise ValueError('"volumetric_function" has to be a "Callable" object.')

        # first call the ray sampler that returns the RayBundle or HeterogeneousRayBundle
        # parametrizing the rendering rays.
        ray_bundle = self.raysampler(
            cameras=cameras, volumetric_function=volumetric_function, **kwargs
        )
        # ray_bundle.origins - minibatch x ... x 3
        # ray_bundle.directions - minibatch x ... x 3
        # ray_bundle.lengths - minibatch x ... x n_pts_per_ray
        # ray_bundle.xys - minibatch x ... x 2

        # given sampled rays, call the volumetric function that
        # evaluates the densities and features at the locations of the
        # ray points
        # pyre-fixme[23]: Unable to unpack `object` into 2 values.
        rays_densities, rays_features = volumetric_function(
            ray_bundle=ray_bundle, cameras=cameras, **kwargs
        )
        # ray_densities - minibatch x ... x n_pts_per_ray x density_dim
        # ray_features - minibatch x ... x n_pts_per_ray x feature_dim

        # finally, march along the sampled rays to obtain the renders
        images = self.raymarcher(
            rays_densities=rays_densities,
            rays_features=rays_features,
            ray_bundle=ray_bundle,
            **kwargs,
        )
        # images - minibatch x ... x (feature_dim + opacity_dim)

        return images, ray_bundle


# The volume renderer class should be initialized with a
# function for raysampling and a function for raymarching.

# During the forward pass:
# 1) The raysampler:
#     - samples rays from input cameras
#     - transforms the rays to world coordinates
# 2) The scene volumes (which are an argument of the forward function)
#    are then sampled at the locations of the ray-points to generate
#    ray_densities and ray_features.
# 3) The raymarcher takes ray_densities and ray_features and uses a raymarching
#    algorithm to render each ray.


class VolumeRenderer(torch.nn.Module):
    """
    A class for rendering a batch of Volumes. The class should
    be initialized with a raysampler and a raymarcher class which both have
    to be a `Callable`.
    """

    def __init__(
        self, raysampler: Callable, raymarcher: Callable, sample_mode: str = "bilinear"
    ) -> None:
        """
        Args:
            raysampler: A `Callable` that takes as input scene cameras
                (an instance of `CamerasBase`) and returns a
                `Union[RayBundle, HeterogeneousRayBundle],` that
                describes the rays emitted from the cameras.
            raymarcher: A `Callable` that receives the `volumes`
                (an instance of `Volumes` input to `self.forward`)
                sampled at the ray-points, and renders the rays with a
                ray-marching algorithm.
            sample_mode: Defines the algorithm used to sample the volumetric
                voxel grid. Can be either "bilinear" or "nearest".
        """
        super().__init__()

        self.renderer = ImplicitRenderer(raysampler, raymarcher)
        self._sample_mode = sample_mode

    def forward(
        self, cameras: CamerasBase, volumes: Volumes, **kwargs
    ) -> Tuple[torch.Tensor, Union[RayBundle, HeterogeneousRayBundle]]:
        """
        Render a batch of images using raymarching over rays cast through
        input `Volumes`.

        Args:
            cameras: A batch of cameras that render the scene. A `self.raysampler`
                takes the cameras as input and samples rays that pass through the
                domain of the volumetric function.
            volumes: An instance of the `Volumes` class representing a
                batch of volumes that are being rendered.

        Returns:
            images: A tensor of shape `(minibatch, ..., (feature_dim + opacity_dim)`
                containing the result of the rendering.
            ray_bundle: A `RayBundle` or `HeterogeneousRayBundle` containing the
                parametrizations of the sampled rendering rays.
        """
        volumetric_function = VolumeSampler(volumes, sample_mode=self._sample_mode)
        return self.renderer(
            cameras=cameras, volumetric_function=volumetric_function, **kwargs
        )


class VolumeSampler(torch.nn.Module):
    """
    A module to sample a batch of volumes `Volumes`
    at 3D points sampled along projection rays.
    """

    def __init__(
        self,
        volumes: Volumes,
        sample_mode: str = "bilinear",
        padding_mode: str = "zeros",
    ) -> None:
        """
        Args:
            volumes: An instance of the `Volumes` class representing a
                batch of volumes that are being rendered.
            sample_mode: Defines the algorithm used to sample the volumetric
                voxel grid. Can be either "bilinear" or "nearest".
            padding_mode: How to handle values outside of the volume.
                One of: zeros, border, reflection
                See torch.nn.functional.grid_sample for more information.
        """
        super().__init__()
        if not isinstance(volumes, Volumes):
            raise ValueError("'volumes' have to be an instance of the 'Volumes' class.")
        self._volumes = volumes
        self._sample_mode = sample_mode
        self._padding_mode = padding_mode

    def _get_ray_directions_transform(self):
        """
        Compose the ray-directions transform by removing the translation component
        from the volume global-to-local coords transform.
        """
        world2local = self._volumes.get_world_to_local_coords_transform().get_matrix()
        directions_transform_matrix = eyes(
            4,
            N=world2local.shape[0],
            device=world2local.device,
            dtype=world2local.dtype,
        )
        directions_transform_matrix[:, :3, :3] = world2local[:, :3, :3]
        directions_transform = Transform3d(matrix=directions_transform_matrix)
        return directions_transform

    def forward(
        self, ray_bundle: Union[RayBundle, HeterogeneousRayBundle], **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given an input ray parametrization, the forward function samples
        `self._volumes` at the respective 3D ray-points.
        Can also accept ImplicitronRayBundle as argument for ray_bundle.

        Args:
            ray_bundle: A RayBundle or HeterogeneousRayBundle object with the following fields:
                rays_origins_world: A tensor of shape `(minibatch, ..., 3)` denoting the
                    origins of the sampling rays in world coords.
                rays_directions_world: A tensor of shape `(minibatch, ..., 3)`
                    containing the direction vectors of sampling rays in world coords.
                rays_lengths: A tensor of shape `(minibatch, ..., num_points_per_ray)`
                    containing the lengths at which the rays are sampled.

        Returns:
            rays_densities: A tensor of shape
                `(minibatch, ..., num_points_per_ray, opacity_dim)` containing the
                density vectors sampled from the volume at the locations of
                the ray points.
            rays_features: A tensor of shape
                `(minibatch, ..., num_points_per_ray, feature_dim)` containing the
                feature vectors sampled from the volume at the locations of
                the ray points.
        """

        # take out the interesting parts of ray_bundle
        rays_origins_world = ray_bundle.origins
        rays_directions_world = ray_bundle.directions
        rays_lengths = ray_bundle.lengths

        # validate the inputs
        _validate_ray_bundle_variables(
            rays_origins_world, rays_directions_world, rays_lengths
        )
        if self._volumes.densities().shape[0] != rays_origins_world.shape[0]:
            raise ValueError("Input volumes have to have the same batch size as rays.")

        #########################################################
        # 1) convert the origins/directions to the local coords #
        #########################################################

        # origins are mapped with the world_to_local transform of the volumes
        rays_origins_local = self._volumes.world_to_local_coords(rays_origins_world)

        # obtain the Transform3d object that transforms ray directions to local coords
        directions_transform = self._get_ray_directions_transform()

        # transform the directions to the local coords
        rays_directions_local = directions_transform.transform_points(
            rays_directions_world.view(rays_lengths.shape[0], -1, 3)
        ).view(rays_directions_world.shape)

        ############################
        # 2) obtain the ray points #
        ############################

        # this op produces a fairly big tensor (minibatch, ..., n_samples_per_ray, 3)
        rays_points_local = ray_bundle_variables_to_ray_points(
            rays_origins_local, rays_directions_local, rays_lengths
        )

        ########################
        # 3) sample the volume #
        ########################

        # generate the tensor for sampling
        volumes_densities = self._volumes.densities()
        dim_density = volumes_densities.shape[1]
        volumes_features = self._volumes.features()

        # reshape to a size which grid_sample likes
        rays_points_local_flat = rays_points_local.view(
            rays_points_local.shape[0], -1, 1, 1, 3
        )

        # run the grid sampler on the volumes densities
        rays_densities = torch.nn.functional.grid_sample(
            volumes_densities,
            rays_points_local_flat,
            mode=self._sample_mode,
            padding_mode=self._padding_mode,
            align_corners=self._volumes.get_align_corners(),
        )

        # permute the dimensions & reshape densities after sampling
        rays_densities = rays_densities.permute(0, 2, 3, 4, 1).view(
            *rays_points_local.shape[:-1], volumes_densities.shape[1]
        )

        # if features exist, run grid sampler again on the features densities
        if volumes_features is None:
            dim_feature = 0
            _, rays_features = rays_densities.split([dim_density, dim_feature], dim=-1)
        else:
            rays_features = torch.nn.functional.grid_sample(
                volumes_features,
                rays_points_local_flat,
                mode=self._sample_mode,
                padding_mode=self._padding_mode,
                align_corners=self._volumes.get_align_corners(),
            )

            # permute the dimensions & reshape features after sampling
            rays_features = rays_features.permute(0, 2, 3, 4, 1).view(
                *rays_points_local.shape[:-1], volumes_features.shape[1]
            )

        return rays_densities, rays_features
