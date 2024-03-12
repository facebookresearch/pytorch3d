# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

import torch
from pytorch3d.renderer import ImplicitRenderer, ray_bundle_to_ray_points
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import plot_scene
from visdom import Visdom

from .implicit_function import NeuralRadianceField
from .raymarcher import EmissionAbsorptionNeRFRaymarcher
from .raysampler import NeRFRaysampler, ProbabilisticRaysampler
from .utils import calc_mse, calc_psnr, sample_images_at_mc_locs


class RadianceFieldRenderer(torch.nn.Module):
    """
    Implements a renderer of a Neural Radiance Field.

    This class holds pointers to the fine and coarse renderer objects, which are
    instances of `pytorch3d.renderer.ImplicitRenderer`, and pointers to the
    neural networks representing the fine and coarse Neural Radiance Fields,
    which are instances of `NeuralRadianceField`.

    The rendering forward pass proceeds as follows:
        1) For a given input camera, rendering rays are generated with the
            `NeRFRaysampler` object of `self._renderer['coarse']`.
            In the training mode (`self.training==True`), the rays are a set
                of `n_rays_per_image` random 2D locations of the image grid.
            In the evaluation mode (`self.training==False`), the rays correspond
                to the full image grid. The rays are further split to
                `chunk_size_test`-sized chunks to prevent out-of-memory errors.
        2) For each ray point, the coarse `NeuralRadianceField` MLP is evaluated.
            The pointer to this MLP is stored in `self._implicit_function['coarse']`
        3) The coarse radiance field is rendered with the
            `EmissionAbsorptionNeRFRaymarcher` object of `self._renderer['coarse']`.
        4) The coarse raymarcher outputs a probability distribution that guides
            the importance raysampling of the fine rendering pass. The
            `ProbabilisticRaysampler` stored in `self._renderer['fine'].raysampler`
            implements the importance ray-sampling.
        5) Similar to 2) the fine MLP in `self._implicit_function['fine']`
            labels the ray points with occupancies and colors.
        6) self._renderer['fine'].raymarcher` generates the final fine render.
        7) The fine and coarse renders are compared to the ground truth input image
            with PSNR and MSE metrics.
    """

    def __init__(
        self,
        image_size: Tuple[int, int],
        n_pts_per_ray: int,
        n_pts_per_ray_fine: int,
        n_rays_per_image: int,
        min_depth: float,
        max_depth: float,
        stratified: bool,
        stratified_test: bool,
        chunk_size_test: int,
        n_harmonic_functions_xyz: int = 6,
        n_harmonic_functions_dir: int = 4,
        n_hidden_neurons_xyz: int = 256,
        n_hidden_neurons_dir: int = 128,
        n_layers_xyz: int = 8,
        append_xyz: Tuple[int, ...] = (5,),
        density_noise_std: float = 0.0,
        visualization: bool = False,
    ):
        """
        Args:
            image_size: The size of the rendered image (`[height, width]`).
            n_pts_per_ray: The number of points sampled along each ray for the
                coarse rendering pass.
            n_pts_per_ray_fine: The number of points sampled along each ray for the
                fine rendering pass.
            n_rays_per_image: Number of Monte Carlo ray samples when training
                (`self.training==True`).
            min_depth: The minimum depth of a sampled ray-point for the coarse rendering.
            max_depth: The maximum depth of a sampled ray-point for the coarse rendering.
            stratified: If `True`, stratifies (=randomly offsets) the depths
                of each ray point during training (`self.training==True`).
            stratified_test: If `True`, stratifies (=randomly offsets) the depths
                of each ray point during evaluation (`self.training==False`).
            chunk_size_test: The number of rays in each chunk of image rays.
                Active only when `self.training==True`.
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
                Prior to evaluating the skip layers, the tensor which was input to MLP
                is appended to the skip layer input.
            density_noise_std: The standard deviation of the random normal noise
                added to the output of the occupancy MLP.
                Active only when `self.training==True`.
            visualization: whether to store extra output for visualization.
        """

        super().__init__()

        # The renderers and implicit functions are stored under the fine/coarse
        # keys in ModuleDict PyTorch modules.
        self._renderer = torch.nn.ModuleDict()
        self._implicit_function = torch.nn.ModuleDict()

        # Init the EA raymarcher used by both passes.
        raymarcher = EmissionAbsorptionNeRFRaymarcher()

        # Parse out image dimensions.
        image_height, image_width = image_size

        for render_pass in ("coarse", "fine"):
            if render_pass == "coarse":
                # Initialize the coarse raysampler.
                raysampler = NeRFRaysampler(
                    n_pts_per_ray=n_pts_per_ray,
                    min_depth=min_depth,
                    max_depth=max_depth,
                    stratified=stratified,
                    stratified_test=stratified_test,
                    n_rays_per_image=n_rays_per_image,
                    image_height=image_height,
                    image_width=image_width,
                )
            elif render_pass == "fine":
                # Initialize the fine raysampler.
                raysampler = ProbabilisticRaysampler(
                    n_pts_per_ray=n_pts_per_ray_fine,
                    stratified=stratified,
                    stratified_test=stratified_test,
                )
            else:
                raise ValueError(f"No such rendering pass {render_pass}")

            # Initialize the fine/coarse renderer.
            self._renderer[render_pass] = ImplicitRenderer(
                raysampler=raysampler,
                raymarcher=raymarcher,
            )

            # Instantiate the fine/coarse NeuralRadianceField module.
            self._implicit_function[render_pass] = NeuralRadianceField(
                n_harmonic_functions_xyz=n_harmonic_functions_xyz,
                n_harmonic_functions_dir=n_harmonic_functions_dir,
                n_hidden_neurons_xyz=n_hidden_neurons_xyz,
                n_hidden_neurons_dir=n_hidden_neurons_dir,
                n_layers_xyz=n_layers_xyz,
                append_xyz=append_xyz,
            )

        self._density_noise_std = density_noise_std
        self._chunk_size_test = chunk_size_test
        self._image_size = image_size
        self.visualization = visualization

    def precache_rays(
        self,
        cache_cameras: List[CamerasBase],
        cache_camera_hashes: List[str],
    ):
        """
        Precaches the rays emitted from the list of cameras `cache_cameras`,
        where each camera is uniquely identified with the corresponding hash
        from `cache_camera_hashes`.

        The cached rays are moved to cpu and stored in
        `self._renderer['coarse']._ray_cache`.

        Raises `ValueError` when caching two cameras with the same hash.

        Args:
            cache_cameras: A list of `N` cameras for which the rays are pre-cached.
            cache_camera_hashes: A list of `N` unique identifiers for each
                camera from `cameras`.
        """
        self._renderer["coarse"].raysampler.precache_rays(
            cache_cameras,
            cache_camera_hashes,
        )

    def _process_ray_chunk(
        self,
        camera_hash: Optional[str],
        camera: CamerasBase,
        image: torch.Tensor,
        chunk_idx: int,
    ) -> dict:
        """
        Samples and renders a chunk of rays.

        Args:
            camera_hash: A unique identifier of a pre-cached camera.
                If `None`, the cache is not searched and the sampled rays are
                calculated from scratch.
            camera: A batch of cameras from which the scene is rendered.
            image: A batch of corresponding ground truth images of shape
                ('batch_size', ·, ·, 3).
            chunk_idx: The index of the currently rendered ray chunk.
        Returns:
            out: `dict` containing the outputs of the rendering:
                `rgb_coarse`: The result of the coarse rendering pass.
                `rgb_fine`: The result of the fine rendering pass.
                `rgb_gt`: The corresponding ground-truth RGB values.
        """
        # Initialize the outputs of the coarse rendering to None.
        coarse_ray_bundle = None
        coarse_weights = None

        # First evaluate the coarse rendering pass, then the fine one.
        for renderer_pass in ("coarse", "fine"):
            (rgb, weights), ray_bundle_out = self._renderer[renderer_pass](
                cameras=camera,
                volumetric_function=self._implicit_function[renderer_pass],
                chunksize=self._chunk_size_test,
                chunk_idx=chunk_idx,
                density_noise_std=(self._density_noise_std if self.training else 0.0),
                input_ray_bundle=coarse_ray_bundle,
                ray_weights=coarse_weights,
                camera_hash=camera_hash,
            )

            if renderer_pass == "coarse":
                rgb_coarse = rgb
                # Store the weights and the rays of the first rendering pass
                # for the ensuing importance ray-sampling of the fine render.
                coarse_ray_bundle = ray_bundle_out
                coarse_weights = weights
                if image is not None:
                    # Sample the ground truth images at the xy locations of the
                    # rendering ray pixels.
                    rgb_gt = sample_images_at_mc_locs(
                        image[..., :3][None],
                        ray_bundle_out.xys,
                    )
                else:
                    rgb_gt = None

            elif renderer_pass == "fine":
                rgb_fine = rgb

            else:
                raise ValueError(f"No such rendering pass {renderer_pass}")

        out = {"rgb_fine": rgb_fine, "rgb_coarse": rgb_coarse, "rgb_gt": rgb_gt}
        if self.visualization:
            # Store the coarse rays/weights only for visualization purposes.
            out["coarse_ray_bundle"] = type(coarse_ray_bundle)(
                *[v.detach().cpu() for k, v in coarse_ray_bundle._asdict().items()]
            )
            out["coarse_weights"] = coarse_weights.detach().cpu()

        return out

    def forward(
        self,
        camera_hash: Optional[str],
        camera: CamerasBase,
        image: torch.Tensor,
    ) -> Tuple[dict, dict]:
        """
        Performs the coarse and fine rendering passes of the radiance field
        from the viewpoint of the input `camera`.
        Afterwards, both renders are compared to the input ground truth `image`
        by evaluating the peak signal-to-noise ratio and the mean-squared error.

        The rendering result depends on the `self.training` flag:
            - In the training mode (`self.training==True`), the function renders
              a random subset of image rays (Monte Carlo rendering).
            - In evaluation mode (`self.training==False`), the function renders
              the full image. In order to prevent out-of-memory errors,
              when `self.training==False`, the rays are sampled and rendered
              in batches of size `chunksize`.

        Args:
            camera_hash: A unique identifier of a pre-cached camera.
                If `None`, the cache is not searched and the sampled rays are
                calculated from scratch.
            camera: A batch of cameras from which the scene is rendered.
            image: A batch of corresponding ground truth images of shape
                ('batch_size', ·, ·, 3).
        Returns:
            out: `dict` containing the outputs of the rendering:
                `rgb_coarse`: The result of the coarse rendering pass.
                `rgb_fine`: The result of the fine rendering pass.
                `rgb_gt`: The corresponding ground-truth RGB values.

                The shape of `rgb_coarse`, `rgb_fine`, `rgb_gt` depends on the
                `self.training` flag:
                    If `==True`, all 3 tensors are of shape
                    `(batch_size, n_rays_per_image, 3)` and contain the result
                    of the Monte Carlo training rendering pass.
                    If `==False`, all 3 tensors are of shape
                    `(batch_size, image_size[0], image_size[1], 3)` and contain
                    the result of the full image rendering pass.
            metrics: `dict` containing the error metrics comparing the fine and
                coarse renders to the ground truth:
                `mse_coarse`: Mean-squared error between the coarse render and
                    the input `image`
                `mse_fine`: Mean-squared error between the fine render and
                    the input `image`
                `psnr_coarse`: Peak signal-to-noise ratio between the coarse render and
                    the input `image`
                `psnr_fine`: Peak signal-to-noise ratio between the fine render and
                    the input `image`
        """
        if not self.training:
            # Full evaluation pass.
            n_chunks = self._renderer["coarse"].raysampler.get_n_chunks(
                self._chunk_size_test,
                camera.R.shape[0],
            )
        else:
            # MonteCarlo ray sampling.
            n_chunks = 1

        # Process the chunks of rays.
        chunk_outputs = [
            self._process_ray_chunk(
                camera_hash,
                camera,
                image,
                chunk_idx,
            )
            for chunk_idx in range(n_chunks)
        ]

        if not self.training:
            # For a full render pass concatenate the output chunks,
            # and reshape to image size.
            out = {
                k: (
                    torch.cat(
                        [ch_o[k] for ch_o in chunk_outputs],
                        dim=1,
                    ).view(-1, *self._image_size, 3)
                    if chunk_outputs[0][k] is not None
                    else None
                )
                for k in ("rgb_fine", "rgb_coarse", "rgb_gt")
            }
        else:
            out = chunk_outputs[0]

        # Calc the error metrics.
        metrics = {}
        if image is not None:
            for render_pass in ("coarse", "fine"):
                for metric_name, metric_fun in zip(
                    ("mse", "psnr"), (calc_mse, calc_psnr)
                ):
                    metrics[f"{metric_name}_{render_pass}"] = metric_fun(
                        out["rgb_" + render_pass][..., :3],
                        out["rgb_gt"][..., :3],
                    )

        return out, metrics


def visualize_nerf_outputs(
    nerf_out: dict, output_cache: List, viz: Visdom, visdom_env: str
):
    """
    Visualizes the outputs of the `RadianceFieldRenderer`.

    Args:
        nerf_out: An output of the validation rendering pass.
        output_cache: A list with outputs of several training render passes.
        viz: A visdom connection object.
        visdom_env: The name of visdom environment for visualization.
    """

    # Show the training images.
    ims = torch.stack([o["image"] for o in output_cache])
    ims = torch.cat(list(ims), dim=1)
    viz.image(
        ims.permute(2, 0, 1),
        env=visdom_env,
        win="images",
        opts={"title": "train_images"},
    )

    # Show the coarse and fine renders together with the ground truth images.
    ims_full = torch.cat(
        [
            nerf_out[imvar][0].permute(2, 0, 1).detach().cpu().clamp(0.0, 1.0)
            for imvar in ("rgb_coarse", "rgb_fine", "rgb_gt")
        ],
        dim=2,
    )
    viz.image(
        ims_full,
        env=visdom_env,
        win="images_full",
        opts={"title": "coarse | fine | target"},
    )

    # Make a 3D plot of training cameras and their emitted rays.
    camera_trace = {
        f"camera_{ci:03d}": o["camera"].cpu() for ci, o in enumerate(output_cache)
    }
    ray_pts_trace = {
        f"ray_pts_{ci:03d}": Pointclouds(
            ray_bundle_to_ray_points(o["coarse_ray_bundle"])
            .detach()
            .cpu()
            .view(1, -1, 3)
        )
        for ci, o in enumerate(output_cache)
    }
    plotly_plot = plot_scene(
        {
            "training_scene": {
                **camera_trace,
                **ray_pts_trace,
            },
        },
        pointcloud_max_points=5000,
        pointcloud_marker_size=1,
        camera_scale=0.3,
    )
    viz.plotlyplot(plotly_plot, env=visdom_env, win="scenes")
