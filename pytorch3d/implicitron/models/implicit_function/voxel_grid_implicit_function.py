# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import warnings
from dataclasses import fields
from typing import Callable, Dict, Optional, Tuple

import torch

from omegaconf import DictConfig

from pytorch3d.implicitron.models.implicit_function.base import ImplicitFunctionBase
from pytorch3d.implicitron.models.implicit_function.decoding_functions import (
    DecoderFunctionBase,
)
from pytorch3d.implicitron.models.implicit_function.voxel_grid import VoxelGridModule
from pytorch3d.implicitron.models.renderer.base import ImplicitronRayBundle
from pytorch3d.implicitron.tools.config import (
    enable_get_default_args,
    get_default_args_field,
    registry,
    run_auto_creation,
)
from pytorch3d.renderer import ray_bundle_to_ray_points
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.renderer.implicit import HarmonicEmbedding

logger = logging.getLogger(__name__)


enable_get_default_args(HarmonicEmbedding)


@registry.register
# pyre-ignore[13]
class VoxelGridImplicitFunction(ImplicitFunctionBase, torch.nn.Module):
    """
    This implicit function consists of two streams, one for the density calculation and one
    for the color calculation. Each of these streams has three main parts:
        1) Voxel grids:
            They take the (x, y, z) position and return the embedding of that point.
            These components are replaceable, you can make your own or choose one of
            several options.
        2) Harmonic embeddings:
            Convert each feature into series of 'harmonic features', feature is passed through
            sine and cosine functions. Input is of shape [minibatch, ..., D] output
            [minibatch, ..., (n_harmonic_functions * 2 + int(append_input)) * D]. Appends
            input by default. If you want it to behave like identity, put n_harmonic_functions=0
            and append_input=True.
        3) Decoding functions:
            The decoder is an instance of the DecoderFunctionBase and converts the embedding
            of a spatial location to density/color. Examples are Identity which returns its
            input and the MLP which uses fully connected nerual network to transform the input.
            These components are replaceable, you can make your own or choose from
            several options.

    Calculating density is done in three steps:
        1) Evaluating the voxel grid on points
        2) Embedding the outputs with harmonic embedding
        3) Passing through the Density decoder

    To calculate the color we need the embedding and the viewing direction, it has five steps:
        1) Transforming the viewing direction with camera
        2) Evaluating the voxel grid on points
        3) Embedding the outputs with harmonic embedding
        4) Embedding the normalized direction with harmonic embedding
        5) Passing everything through the Color decoder

    If using the Implicitron configuration system the input_dim to the decoding functions will
    be set to the output_dim of the Harmonic embeddings.

    A speed up comes from using the scaffold, a low resolution voxel grid.
    The scaffold is referenced as "binary occupancy grid mask" in TensoRF paper and "AlphaMask"
    in official TensoRF implementation.
    The scaffold is used in:
        1) filtering points in empty space
            - controlled by `scaffold_filter_points` boolean. If set to True, points for which
                scaffold predicts that are in empty space will return 0 density and
                (0, 0, 0) color.
        2) calculating the bounding box of an object and cropping the voxel grids
            - controlled by `volume_cropping_epochs`.
            - at those epochs the implicit function will find the bounding box of an object
                inside it and crop density and color grids. Cropping of the voxel grids means
                preserving only voxel values that are inside the bounding box and changing the
                resolution to match the original, while preserving the new cropped location in
                world coordinates.

    The scaffold has to exist before attempting filtering and cropping, and is created on
    `scaffold_calculating_epochs`. Each voxel in the scaffold is labeled as having density 1 if
    the point in the center of it evaluates to greater than `scaffold_empty_space_threshold`.
    3D max pooling is performed on the densities of the points in 3D.
    Scaffold features are off by default.

    Members:
        voxel_grid_density (VoxelGridBase): voxel grid to use for density estimation
        voxel_grid_color   (VoxelGridBase): voxel grid to use for color   estimation

        harmonic_embedder_xyz_density (HarmonicEmbedder): Function to transform the outputs of
            the voxel_grid_density
        harmonic_embedder_xyz_color (HarmonicEmbedder): Function to transform the outputs of
            the voxel_grid_color for density
        harmonic_embedder_dir_color (HarmonicEmbedder): Function to transform the outputs of
            the voxel_grid_color for color

        decoder_density (DecoderFunctionBase): decoder function to use for density estimation
        color_density   (DecoderFunctionBase): decoder function to use for color   estimation

        use_multiple_streams (bool): if you want the density and color calculations to run on
            different cuda streams set this to True. Default True.
        xyz_ray_dir_in_camera_coords (bool): This is true if the directions are given in
            camera coordinates. Default False.

        voxel_grid_scaffold (VoxelGridModule): which holds the scaffold. Extents and
            translation of it are set to those of voxel_grid_density.
        scaffold_calculating_epochs (Tuple[int, ...]): at which epochs to recalculate the
            scaffold. (The scaffold will be created automatically at the beginning of
            the calculation.)
        scaffold_resolution (Tuple[int, int, int]): (width, height, depth) of the underlying
            voxel grid which stores scaffold
        scaffold_empty_space_threshold (float): if `self._get_density` evaluates to less than
            this it will be considered as empty space and the scaffold at that point would
            evaluate as empty space.
        scaffold_occupancy_chunk_size (str or int): Number of xy scaffold planes to calculate
            at the same time. To calculate the scaffold we need to query `_get_density()` at
            every voxel, this calculation can be split into scaffold depth number of xy plane
            calculations if you want the lowest memory usage, one calculation to calculate the
            whole scaffold, but with higher memory footprint or any other number of planes.
            Setting to a non-positive number calculates all planes at the same time.
            Defaults to '-1' (=calculating all planes).
        scaffold_max_pool_kernel_size (int): Size of the pooling region to use when
            calculating the scaffold. Defaults to 3.
        scaffold_filter_points (bool): If set to True the points will be filtered using
            `self.voxel_grid_scaffold`. Filtered points will be predicted as having 0 density
            and (0, 0, 0) color. The points which were not evaluated as empty space will be
            passed through the steps outlined above.
        volume_cropping_epochs: on which epochs to crop the voxel grids to fit the object's
            bounding box. Scaffold has to be calculated before cropping.
    """

    # ---- voxel grid for density
    voxel_grid_density: VoxelGridModule

    # ---- voxel grid for color
    voxel_grid_color: VoxelGridModule

    # ---- harmonic embeddings density
    harmonic_embedder_xyz_density_args: DictConfig = get_default_args_field(
        HarmonicEmbedding
    )
    harmonic_embedder_xyz_color_args: DictConfig = get_default_args_field(
        HarmonicEmbedding
    )
    harmonic_embedder_dir_color_args: DictConfig = get_default_args_field(
        HarmonicEmbedding
    )

    # ---- decoder function for density
    decoder_density_class_type: str = "MLPDecoder"
    decoder_density: DecoderFunctionBase

    # ---- decoder function for color
    decoder_color_class_type: str = "MLPDecoder"
    decoder_color: DecoderFunctionBase

    # ---- cuda streams
    use_multiple_streams: bool = True

    # ---- camera
    xyz_ray_dir_in_camera_coords: bool = False

    # --- scaffold
    # voxel_grid_scaffold: VoxelGridModule
    scaffold_calculating_epochs: Tuple[int, ...] = ()
    scaffold_resolution: Tuple[int, int, int] = (128, 128, 128)
    scaffold_empty_space_threshold: float = 0.001
    scaffold_occupancy_chunk_size: int = -1
    scaffold_max_pool_kernel_size: int = 3
    scaffold_filter_points: bool = True

    # --- cropping
    volume_cropping_epochs: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        run_auto_creation(self)
        self.voxel_grid_scaffold = self._create_voxel_grid_scaffold()
        self.harmonic_embedder_xyz_density = HarmonicEmbedding(
            **self.harmonic_embedder_xyz_density_args
        )
        self.harmonic_embedder_xyz_color = HarmonicEmbedding(
            **self.harmonic_embedder_xyz_color_args
        )
        self.harmonic_embedder_dir_color = HarmonicEmbedding(
            **self.harmonic_embedder_dir_color_args
        )
        self._scaffold_ready = False

    def forward(
        self,
        ray_bundle: ImplicitronRayBundle,
        fun_viewpool=None,
        camera: Optional[CamerasBase] = None,
        global_code=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        The forward function accepts the parametrizations of 3D points sampled along
        projection rays. The forward pass is responsible for attaching a 3D vector
        and a 1D scalar representing the point's RGB color and opacity respectively.

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
            camera: A camera model which will be used to transform the viewing
                directions

        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacitiy of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.
        """
        # ########## convert the ray parametrizations to world coordinates ########## #
        # points.shape = [minibatch x n_rays_width x n_rays_height x pts_per_ray x 3]
        # pyre-ignore[6]
        points = ray_bundle_to_ray_points(ray_bundle)
        directions = ray_bundle.directions.reshape(-1, 3)
        input_shape = points.shape
        num_points_per_ray = input_shape[-2]
        points = points.view(-1, 3)
        non_empty_points = None

        # ########## filter the points using the scaffold ########## #
        if self._scaffold_ready and self.scaffold_filter_points:
            with torch.no_grad():
                non_empty_points = self.voxel_grid_scaffold(points)[..., 0] > 0
            points = points[non_empty_points]
            if len(points) == 0:
                warnings.warn(
                    "The scaffold has filtered all the points."
                    "The voxel grids and decoding functions will not be run."
                )
                return (
                    points.new_zeros((*input_shape[:-1], 1)),
                    points.new_zeros((*input_shape[:-1], 3)),
                    {},
                )

        # ########## calculate color and density ########## #
        rays_densities, rays_colors = self._calculate_density_and_color(
            points, directions, camera, non_empty_points, num_points_per_ray
        )

        if not (self._scaffold_ready and self.scaffold_filter_points):
            return (
                rays_densities.view((*input_shape[:-1], rays_densities.shape[-1])),
                rays_colors.view((*input_shape[:-1], rays_colors.shape[-1])),
                {},
            )

        # ########## merge scaffold calculated points ########## #
        # Create a zeroed tensor corresponding to a point with density=0 and fill it
        # with calculated density for points which are not in empty space. Do the
        # same for color
        rays_densities_combined = rays_densities.new_zeros(
            (math.prod(input_shape[:-1]), rays_densities.shape[-1])
        )
        rays_colors_combined = rays_colors.new_zeros(
            (math.prod(input_shape[:-1]), rays_colors.shape[-1])
        )
        assert non_empty_points is not None
        rays_densities_combined[non_empty_points] = rays_densities
        rays_colors_combined[non_empty_points] = rays_colors

        return (
            rays_densities_combined.view((*input_shape[:-1], rays_densities.shape[-1])),
            rays_colors_combined.view((*input_shape[:-1], rays_colors.shape[-1])),
            {},
        )

    def _calculate_density_and_color(
        self,
        points: torch.Tensor,
        directions: torch.Tensor,
        camera: Optional[CamerasBase],
        non_empty_points: Optional[torch.Tensor],
        num_points_per_ray: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates density and color at `points`.
        If enabled use cuda streams.

        Args:
            points: points at which to calculate density and color.
                Tensor of shape [n_points, 3].
            directions: from which directions are the points viewed.
                One per ray. Tensor of shape [n_rays, 3].
            camera: A camera model which will be used to transform the viewing
                directions
            non_empty_points: indices of points which weren't filtered out;
                used for expanding directions
            num_points_per_ray: number of points per ray, needed to expand directions.
        Returns:
               Tuple of color (tensor of shape [..., 3]) and density
                (tensor of shape [..., 1])
        """
        if self.use_multiple_streams and points.is_cuda:
            current_stream = torch.cuda.current_stream(points.device)
            other_stream = torch.cuda.Stream(points.device)
            other_stream.wait_stream(current_stream)

            with torch.cuda.stream(other_stream):
                # rays_densities.shape =
                # [minibatch x n_rays_width x n_rays_height x pts_per_ray x density_dim]
                rays_densities = self._get_density(points)

            # rays_colors.shape =
            # [minibatch x n_rays_width x n_rays_height x pts_per_ray x color_dim]
            rays_colors = self._get_color(
                points, camera, directions, non_empty_points, num_points_per_ray
            )

            current_stream.wait_stream(other_stream)
        else:
            # Same calculation as above, just serial.
            rays_densities = self._get_density(points)
            rays_colors = self._get_color(
                points, camera, directions, non_empty_points, num_points_per_ray
            )
        return rays_densities, rays_colors

    def _get_density(self, points: torch.Tensor) -> torch.Tensor:
        """
        Calculates density at points:
            1) Evaluates the voxel grid on points
            2) Embeds the outputs with harmonic embedding
            3) Passes everything through the Density decoder

        Args:
            points: tensor of shape [..., 3]
                where the last dimension is the points in the (x, y, z)
        Returns:
            calculated densities of shape [..., density_dim], `density_dim` is the
                feature dimensionality which `decoder_density` returns
        """
        embeds_density = self.voxel_grid_density(points)
        harmonic_embedding_density = self.harmonic_embedder_xyz_density(embeds_density)
        # shape = [..., density_dim]
        return self.decoder_density(harmonic_embedding_density)

    def _get_color(
        self,
        points: torch.Tensor,
        camera: Optional[CamerasBase],
        directions: torch.Tensor,
        non_empty_points: Optional[torch.Tensor],
        num_points_per_ray: int,
    ) -> torch.Tensor:
        """
        Calculates color at points using the viewing direction:
            1) Transforms the viewing direction with camera
            2) Evaluates the voxel grid on points
            3) Embeds the outputs with harmonic embedding
            4) Embeds the normalized direction with harmonic embedding
            5) Passes everything through the Color decoder
        Args:
            points: tensor of shape (..., 3)
                where the last dimension is the points in the (x, y, z)
            camera: A camera model which will be used to transform the viewing
                directions
            directions: A tensor of shape `(..., 3)`
                containing the direction vectors of sampling rays in world coords.
            non_empty_points: indices of points which weren't filtered out;
                used for expanding directions
            num_points_per_ray: number of points per ray, needed to expand directions.
        """
        # ########## transform direction ########## #
        if self.xyz_ray_dir_in_camera_coords:
            if camera is None:
                raise ValueError("Camera must be given if xyz_ray_dir_in_camera_coords")
            directions = directions @ camera.R

        # ########## get voxel grid output ########## #
        # embeds_color.shape = [..., pts_per_ray, n_features]
        embeds_color = self.voxel_grid_color(points)

        # ########## embed with the harmonic function ########## #
        # Obtain the harmonic embedding of the voxel grid output.
        harmonic_embedding_color = self.harmonic_embedder_xyz_color(embeds_color)

        # Normalize the ray_directions to unit l2 norm.
        rays_directions_normed = torch.nn.functional.normalize(directions, dim=-1)
        # Obtain the harmonic embedding of the normalized ray directions.
        harmonic_embedding_dir = self.harmonic_embedder_dir_color(
            rays_directions_normed
        )

        harmonic_embedding_dir = torch.repeat_interleave(
            harmonic_embedding_dir, num_points_per_ray, dim=0
        )
        if non_empty_points is not None:
            harmonic_embedding_dir = harmonic_embedding_dir[non_empty_points]

        # total color embedding is concatenation of the harmonic embedding of voxel grid
        # output and harmonic embedding of the normalized direction
        total_color_embedding = torch.cat(
            (harmonic_embedding_color, harmonic_embedding_dir), dim=-1
        )

        # ########## evaluate color with the decoding function ########## #
        # rays_colors.shape = [..., pts_per_ray, 3] in [0-1]
        return self.decoder_color(total_color_embedding)

    @staticmethod
    def allows_multiple_passes() -> bool:
        """
        Returns True as this implicit function allows
        multiple passes. Overridden from ImplicitFunctionBase.
        """
        return True

    def subscribe_to_epochs(self) -> Tuple[Tuple[int, ...], Callable[[int], bool]]:
        """
        Method which expresses interest in subscribing to optimization epoch updates.
        This implicit function subscribes to epochs to calculate the scaffold and to
        crop voxel grids, so this method combines wanted epochs and wraps their callbacks.

        Returns:
            list of epochs on which to call a callable and callable to be called on
                particular epoch. The callable returns True if parameter change has
                happened else False and it must be supplied with one argument, epoch.
        """

        def callback(epoch) -> bool:
            change = False
            if epoch in self.scaffold_calculating_epochs:
                change = self._get_scaffold(epoch)
            if epoch in self.volume_cropping_epochs:
                change = self._crop(epoch) or change
            return change

        # remove duplicates
        call_epochs = list(
            set(self.scaffold_calculating_epochs) | set(self.volume_cropping_epochs)
        )
        return call_epochs, callback

    def _crop(self, epoch: int) -> bool:
        """
        Finds the bounding box of an object represented in the scaffold and crops
        density and color voxel grids to match that bounding box. If density of the
        scaffold is 0 everywhere (there is no object in it) no change will
        happen.

        Args:
            epoch: ignored
        Returns:
            True (indicating that parameter change has happened) if there is
            an object inside, else False.
        """
        # find bounding box
        points = self.voxel_grid_scaffold.get_grid_points(epoch=epoch)
        assert self._scaffold_ready, "Scaffold has to be calculated before cropping."
        occupancy = self.voxel_grid_scaffold(points)[..., 0] > 0
        non_zero_idxs = torch.nonzero(occupancy)
        if len(non_zero_idxs) == 0:
            return False
        min_indices = tuple(torch.min(non_zero_idxs, dim=0)[0])
        max_indices = tuple(torch.max(non_zero_idxs, dim=0)[0])
        min_point, max_point = points[min_indices], points[max_indices]

        logger.info(
            f"Cropping at epoch {epoch} to bounding box "
            f"[{min_point.tolist()}, {max_point.tolist()}]."
        )

        # crop the voxel grids
        self.voxel_grid_density.crop_self(min_point, max_point)
        self.voxel_grid_color.crop_self(min_point, max_point)
        return True

    @torch.no_grad()
    def _get_scaffold(self, epoch: int) -> bool:
        """
        Creates a low resolution grid which is used to filter points that are in empty
        space.

        Args:
            epoch: epoch on which it is called, ignored inside method
        Returns:
             Always False: Modifies `self.voxel_grid_scaffold` member.
        """

        planes = []
        points = self.voxel_grid_scaffold.get_grid_points(epoch=epoch)

        chunk_size = (
            self.scaffold_occupancy_chunk_size
            if self.scaffold_occupancy_chunk_size > 0
            else points.shape[-1]
        )
        for k in range(0, points.shape[-1], chunk_size):
            points_in_planes = points[..., k : k + chunk_size]
            planes.append(self._get_density(points_in_planes)[..., 0])

        density_cube = torch.cat(planes, dim=-1)
        density_cube = torch.nn.functional.max_pool3d(
            density_cube[None, None],
            kernel_size=self.scaffold_max_pool_kernel_size,
            padding=self.scaffold_max_pool_kernel_size // 2,
            stride=1,
        )
        occupancy_cube = density_cube > self.scaffold_empty_space_threshold
        self.voxel_grid_scaffold.params["voxel_grid"] = occupancy_cube.float()
        self._scaffold_ready = True

        return False

    @classmethod
    def decoder_density_tweak_args(cls, type_, args: DictConfig) -> None:
        args.pop("input_dim", None)

    def create_decoder_density_impl(self, type_, args: DictConfig) -> None:
        """
        Decoding functions come after harmonic embedding and voxel grid. In order to not
        calculate the input dimension of the decoder in the config file this function
        calculates the required input dimension and sets the input dimension of the
        decoding function to this value.
        """
        grid_args = self.voxel_grid_density_args
        grid_output_dim = VoxelGridModule.get_output_dim(grid_args)

        embedder_args = self.harmonic_embedder_xyz_density_args
        input_dim = HarmonicEmbedding.get_output_dim_static(
            grid_output_dim,
            embedder_args["n_harmonic_functions"],
            embedder_args["append_input"],
        )

        cls = registry.get(DecoderFunctionBase, type_)
        need_input_dim = any(field.name == "input_dim" for field in fields(cls))
        if need_input_dim:
            self.decoder_density = cls(input_dim=input_dim, **args)
        else:
            self.decoder_density = cls(**args)

    @classmethod
    def decoder_color_tweak_args(cls, type_, args: DictConfig) -> None:
        args.pop("input_dim", None)

    def create_decoder_color_impl(self, type_, args: DictConfig) -> None:
        """
        Decoding functions come after harmonic embedding and voxel grid. In order to not
        calculate the input dimension of the decoder in the config file this function
        calculates the required input dimension and sets the input dimension of the
        decoding function to this value.
        """
        grid_args = self.voxel_grid_color_args
        grid_output_dim = VoxelGridModule.get_output_dim(grid_args)

        embedder_args = self.harmonic_embedder_xyz_color_args
        input_dim0 = HarmonicEmbedding.get_output_dim_static(
            grid_output_dim,
            embedder_args["n_harmonic_functions"],
            embedder_args["append_input"],
        )

        dir_dim = 3
        embedder_args = self.harmonic_embedder_dir_color_args
        input_dim1 = HarmonicEmbedding.get_output_dim_static(
            dir_dim,
            embedder_args["n_harmonic_functions"],
            embedder_args["append_input"],
        )

        input_dim = input_dim0 + input_dim1

        cls = registry.get(DecoderFunctionBase, type_)
        need_input_dim = any(field.name == "input_dim" for field in fields(cls))
        if need_input_dim:
            self.decoder_color = cls(input_dim=input_dim, **args)
        else:
            self.decoder_color = cls(**args)

    def _create_voxel_grid_scaffold(self) -> VoxelGridModule:
        """
        Creates object to become self.voxel_grid_scaffold:
            -  makes `self.voxel_grid_scaffold` have same world to local mapping as
                    `self.voxel_grid_density`
        """
        return VoxelGridModule(
            extents=self.voxel_grid_density_args["extents"],
            translation=self.voxel_grid_density_args["translation"],
            voxel_grid_class_type="FullResolutionVoxelGrid",
            hold_voxel_grid_as_parameters=False,
            voxel_grid_FullResolutionVoxelGrid_args={
                "resolution_changes": {0: self.scaffold_resolution},
                "padding": "zeros",
                "align_corners": True,
                "mode": "trilinear",
            },
        )
