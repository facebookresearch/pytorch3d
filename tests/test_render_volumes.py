# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Optional, Tuple

import numpy as np
import torch
from pytorch3d.ops import knn_points
from pytorch3d.renderer import (
    AbsorptionOnlyRaymarcher,
    AlphaCompositor,
    EmissionAbsorptionRaymarcher,
    MonteCarloRaysampler,
    MultinomialRaysampler,
    NDCMultinomialRaysampler,
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    RayBundle,
    VolumeRenderer,
    VolumeSampler,
)
from pytorch3d.renderer.implicit.utils import _validate_ray_bundle_variables
from pytorch3d.structures import Pointclouds, Volumes

from .common_testing import TestCaseMixin
from .test_points_to_volumes import init_uniform_y_rotations


DEBUG = False
if DEBUG:
    import os
    import tempfile

    from PIL import Image


ZERO_TRANSLATION = torch.zeros(1, 3)


def init_boundary_volume(
    batch_size: int,
    volume_size: Tuple[int, int, int],
    border_offset: int = 2,
    shape: str = "cube",
    volume_translation: torch.Tensor = ZERO_TRANSLATION,
):
    """
    Generate a volume with sides colored with distinct colors.
    """

    device = torch.device("cuda")

    # first center the volume for the purpose of generating the canonical shape
    volume_translation_tmp = (0.0, 0.0, 0.0)

    # set the voxel size to 1 / (volume_size-1)
    volume_voxel_size = 1 / (volume_size[0] - 1.0)

    # colors of the sides of the cube
    clr_sides = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )

    # get the coord grid of the volume
    coord_grid = Volumes(
        densities=torch.zeros(1, 1, *volume_size, device=device),
        voxel_size=volume_voxel_size,
        volume_translation=volume_translation_tmp,
    ).get_coord_grid()[0]

    # extract the boundary points and their colors of the cube
    if shape == "cube":
        boundary_points, boundary_colors = [], []
        for side, clr_side in enumerate(clr_sides):
            first = side % 2
            dim = side // 2
            slices = [slice(border_offset, -border_offset, 1)] * 3
            slices[dim] = int(border_offset * (2 * first - 1))
            slices.append(slice(0, 3, 1))
            boundary_points_ = coord_grid[slices].reshape(-1, 3)
            boundary_points.append(boundary_points_)
            boundary_colors.append(clr_side[None].expand_as(boundary_points_))
        # set the internal part of the volume to be completely opaque
        volume_densities = torch.zeros(*volume_size, device=device)
        volume_densities[[slice(border_offset, -border_offset, 1)] * 3] = 1.0
        boundary_points, boundary_colors = [
            torch.cat(p, dim=0) for p in [boundary_points, boundary_colors]
        ]
        # color the volume voxels with the nearest boundary points' color
        _, idx, _ = knn_points(
            coord_grid.view(1, -1, 3), boundary_points.view(1, -1, 3)
        )
        volume_colors = (
            boundary_colors[idx.view(-1)].view(*volume_size, 3).permute(3, 0, 1, 2)
        )

    elif shape == "sphere":
        # set all voxels within a certain distance from the origin to be opaque
        volume_densities = (
            coord_grid.norm(dim=-1)
            <= 0.5 * volume_voxel_size * (volume_size[0] - border_offset)
        ).float()
        # color each voxel with the standrd spherical color
        volume_colors = (
            (torch.nn.functional.normalize(coord_grid, dim=-1) + 1.0) * 0.5
        ).permute(3, 0, 1, 2)

    else:
        raise ValueError(shape)

    volume_voxel_size = torch.ones((batch_size, 1), device=device) * volume_voxel_size
    volume_translation = volume_translation.expand(batch_size, 3)
    volumes = Volumes(
        densities=volume_densities[None, None].expand(batch_size, 1, *volume_size),
        features=volume_colors[None].expand(batch_size, 3, *volume_size),
        voxel_size=volume_voxel_size,
        volume_translation=volume_translation,
    )

    return volumes, volume_voxel_size, volume_translation


def init_cameras(
    batch_size: int = 10,
    image_size: Optional[Tuple[int, int]] = (50, 50),
    ndc: bool = False,
):
    """
    Initialize a batch of cameras whose extrinsics rotate the cameras around
    the world's y axis.
    Depending on whether we want an NDC-space (`ndc==True`) or a screen-space camera,
    the camera's focal length and principal point are initialized accordingly:
        For `ndc==False`, p0=focal_length=image_size/2.
        For `ndc==True`, focal_length=1.0, p0 = 0.0.
    The the z-coordinate of the translation vector of each camera is fixed to 1.5.
    """
    device = torch.device("cuda:0")

    # trivial rotations
    R = init_uniform_y_rotations(batch_size=batch_size, device=device)

    # move camera 1.5 m away from the scene center
    T = torch.zeros((batch_size, 3), device=device)
    T[:, 2] = 1.5

    if ndc:
        p0 = torch.zeros(batch_size, 2, device=device)
        focal = torch.ones(batch_size, device=device)
    else:
        p0 = torch.ones(batch_size, 2, device=device)
        p0[:, 0] *= image_size[1] * 0.5
        p0[:, 1] *= image_size[0] * 0.5
        focal = max(*image_size) * torch.ones(batch_size, device=device)

    # convert to a Camera object
    cameras = PerspectiveCameras(focal, p0, R=R, T=T, device=device)
    return cameras


class TestRenderVolumes(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(42)
        np.random.seed(42)

    @staticmethod
    def renderer(
        volume_size=(25, 25, 25),
        batch_size=10,
        shape="sphere",
        raymarcher_type=EmissionAbsorptionRaymarcher,
        n_rays_per_image=10,
        n_pts_per_ray=10,
    ):
        # get the volumes
        volumes = init_boundary_volume(
            volume_size=volume_size, batch_size=batch_size, shape=shape
        )[0]

        # init the mc raysampler
        raysampler = MonteCarloRaysampler(
            min_x=-1.0,
            max_x=1.0,
            min_y=-1.0,
            max_y=1.0,
            n_rays_per_image=n_rays_per_image,
            n_pts_per_ray=n_pts_per_ray,
            min_depth=0.1,
            max_depth=2.0,
        ).to(volumes.device)

        # get the raymarcher
        raymarcher = raymarcher_type()

        renderer = VolumeRenderer(
            raysampler=raysampler, raymarcher=raymarcher, sample_mode="bilinear"
        )

        # generate NDC camera extrinsics and intrinsics
        cameras = init_cameras(batch_size, image_size=None, ndc=True)

        def run_renderer():
            renderer(cameras=cameras, volumes=volumes)

        return run_renderer

    def test_input_types(self, batch_size: int = 10):
        """
        Check that ValueErrors are thrown where expected.
        """
        # check the constructor
        for bad_raysampler in (None, 5, []):
            for bad_raymarcher in (None, 5, []):
                with self.assertRaises(ValueError):
                    VolumeRenderer(raysampler=bad_raysampler, raymarcher=bad_raymarcher)

        raysampler = NDCMultinomialRaysampler(
            image_width=100,
            image_height=100,
            n_pts_per_ray=10,
            min_depth=0.1,
            max_depth=1.0,
        )

        # init a trivial renderer
        renderer = VolumeRenderer(
            raysampler=raysampler, raymarcher=EmissionAbsorptionRaymarcher()
        )

        # get cameras
        cameras = init_cameras(batch_size=batch_size)

        # get volumes
        volumes = init_boundary_volume(volume_size=(10, 10, 10), batch_size=batch_size)[
            0
        ]

        # different batch sizes for cameras / volumes
        with self.assertRaises(ValueError):
            renderer(cameras=cameras, volumes=volumes[:-1])

        # ray checks for VolumeSampler
        volume_sampler = VolumeSampler(volumes=volumes)
        n_rays = 100
        for bad_ray_bundle in (
            (
                torch.rand(batch_size, n_rays, 3),
                torch.rand(batch_size, n_rays + 1, 3),
                torch.rand(batch_size, n_rays, 10),
            ),
            (
                torch.rand(batch_size + 1, n_rays, 3),
                torch.rand(batch_size, n_rays, 3),
                torch.rand(batch_size, n_rays, 10),
            ),
            (
                torch.rand(batch_size, n_rays, 3),
                torch.rand(batch_size, n_rays, 2),
                torch.rand(batch_size, n_rays, 10),
            ),
            (
                torch.rand(batch_size, n_rays, 3),
                torch.rand(batch_size, n_rays, 3),
                torch.rand(batch_size, n_rays),
            ),
        ):
            ray_bundle = RayBundle(
                **dict(
                    zip(
                        ("origins", "directions", "lengths"),
                        [r.to(cameras.device) for r in bad_ray_bundle],
                    )
                ),
                xys=None,
            )
            with self.assertRaises(ValueError):
                volume_sampler(ray_bundle)

            # check also explicitly the ray bundle validation function
            with self.assertRaises(ValueError):
                _validate_ray_bundle_variables(*bad_ray_bundle)

    def test_compare_with_pointclouds_renderer(
        self, batch_size=11, volume_size=(30, 30, 30), image_size=(200, 250)
    ):
        """
        Generate a volume and its corresponding point cloud and check whether
        PointsRenderer returns the same images as the corresponding VolumeRenderer.
        """

        # generate NDC camera extrinsics and intrinsics
        cameras = init_cameras(batch_size, image_size=image_size, ndc=True)

        # init the boundary volume
        for shape in ("sphere", "cube"):
            if not DEBUG and shape == "cube":
                # do not run numeric checks for the cube as the
                # differences in rendering equations make the renders incomparable
                continue

            # get rand offset of the volume
            volume_translation = torch.randn(batch_size, 3) * 0.1
            # volume_translation[2] = 0.1
            volumes = init_boundary_volume(
                volume_size=volume_size,
                batch_size=batch_size,
                shape=shape,
                volume_translation=volume_translation,
            )[0]

            # convert the volumes to a pointcloud
            points = []
            points_features = []
            for densities_one, features_one, grid_one in zip(
                volumes.densities(),
                volumes.features(),
                volumes.get_coord_grid(world_coordinates=True),
            ):
                opaque = densities_one.view(-1) > 1e-4
                points.append(grid_one.view(-1, 3)[opaque])
                points_features.append(features_one.reshape(3, -1).t()[opaque])
            pointclouds = Pointclouds(points, features=points_features)

            # init the grid raysampler with the ndc grid
            coord_range = 1.0
            half_pix_size = coord_range / max(*image_size)
            raysampler = NDCMultinomialRaysampler(
                image_width=image_size[1],
                image_height=image_size[0],
                n_pts_per_ray=256,
                min_depth=0.1,
                max_depth=2.0,
            )

            # get the EA raymarcher
            raymarcher = EmissionAbsorptionRaymarcher()

            # jitter the camera intrinsics a bit for each render
            cameras_randomized = cameras.clone()
            cameras_randomized.principal_point = (
                torch.randn_like(cameras.principal_point) * 0.3
            )
            cameras_randomized.focal_length = (
                cameras.focal_length + torch.randn_like(cameras.focal_length) * 0.2
            )

            # get the volumetric render
            images = VolumeRenderer(
                raysampler=raysampler, raymarcher=raymarcher, sample_mode="bilinear"
            )(cameras=cameras_randomized, volumes=volumes)[0][..., :3]

            # instantiate the points renderer
            point_radius = 6 * half_pix_size
            points_renderer = PointsRenderer(
                rasterizer=PointsRasterizer(
                    cameras=cameras_randomized,
                    raster_settings=PointsRasterizationSettings(
                        image_size=image_size, radius=point_radius, points_per_pixel=10
                    ),
                ),
                compositor=AlphaCompositor(),
            )

            # get the point render
            images_pts = points_renderer(pointclouds)

            if shape == "sphere":
                diff = (images - images_pts).abs().mean(dim=-1)
                mu_diff = diff.mean(dim=(1, 2))
                std_diff = diff.std(dim=(1, 2))
                self.assertClose(mu_diff, torch.zeros_like(mu_diff), atol=3e-2)
                self.assertClose(std_diff, torch.zeros_like(std_diff), atol=6e-2)

            if DEBUG:
                outdir = tempfile.gettempdir() + "/test_volume_vs_pts_renderer"
                os.makedirs(outdir, exist_ok=True)

                frames = []
                for image, image_pts in zip(images, images_pts):
                    diff_image = (
                        ((image - image_pts) * 0.5 + 0.5)
                        .mean(dim=2, keepdim=True)
                        .repeat(1, 1, 3)
                    )
                    image_pil = Image.fromarray(
                        (
                            torch.cat((image, image_pts, diff_image), dim=1)
                            .detach()
                            .cpu()
                            .numpy()
                            * 255.0
                        ).astype(np.uint8)
                    )
                    frames.append(image_pil)

                # export gif
                outfile = os.path.join(outdir, f"volume_vs_pts_render_{shape}.gif")
                frames[0].save(
                    outfile,
                    save_all=True,
                    append_images=frames[1:],
                    duration=batch_size // 15,
                    loop=0,
                )
                print(f"exported {outfile}")

                # export concatenated frames
                outfile_cat = os.path.join(outdir, f"volume_vs_pts_render_{shape}.png")
                Image.fromarray(
                    np.concatenate([np.array(f) for f in frames], axis=0)
                ).save(outfile_cat)
                print(f"exported {outfile_cat}")

    def test_monte_carlo_rendering(
        self, n_frames=20, volume_size=(30, 30, 30), image_size=(40, 50)
    ):
        """
        Tests that rendering with the MonteCarloRaysampler matches the
        rendering with MultinomialRaysampler sampled at the corresponding
        MonteCarlo locations.
        """
        volumes = init_boundary_volume(
            volume_size=volume_size, batch_size=n_frames, shape="sphere"
        )[0]

        # generate camera extrinsics and intrinsics
        cameras = init_cameras(n_frames, image_size=image_size)

        # init the grid raysampler
        raysampler_multinomial = MultinomialRaysampler(
            min_x=0.5,
            max_x=image_size[1] - 0.5,
            min_y=0.5,
            max_y=image_size[0] - 0.5,
            image_width=image_size[1],
            image_height=image_size[0],
            n_pts_per_ray=256,
            min_depth=0.5,
            max_depth=2.0,
        )

        # init the mc raysampler
        raysampler_mc = MonteCarloRaysampler(
            min_x=0.5,
            max_x=image_size[1] - 0.5,
            min_y=0.5,
            max_y=image_size[0] - 0.5,
            n_rays_per_image=3000,
            n_pts_per_ray=256,
            min_depth=0.5,
            max_depth=2.0,
        )

        # get the EA raymarcher
        raymarcher = EmissionAbsorptionRaymarcher()

        # get both mc and grid renders
        (
            (images_opacities_mc, ray_bundle_mc),
            (images_opacities_grid, ray_bundle_grid),
        ) = [
            VolumeRenderer(
                raysampler=raysampler_multinomial,
                raymarcher=raymarcher,
                sample_mode="bilinear",
            )(cameras=cameras, volumes=volumes)
            for raysampler in (raysampler_mc, raysampler_multinomial)
        ]

        # convert the mc sampling locations to [-1, 1]
        sample_loc = ray_bundle_mc.xys.clone()
        sample_loc[..., 0] = 2 * (sample_loc[..., 0] / image_size[1]) - 1
        sample_loc[..., 1] = 2 * (sample_loc[..., 1] / image_size[0]) - 1

        # sample the grid render at the mc locations
        images_opacities_mc_ = torch.nn.functional.grid_sample(
            images_opacities_grid.permute(0, 3, 1, 2), sample_loc, align_corners=False
        )

        # check that the samples are the same
        self.assertClose(
            images_opacities_mc.permute(0, 3, 1, 2), images_opacities_mc_, atol=1e-4
        )

    def test_rotating_gif(self):
        self._rotating_gif(image_size=(200, 100))
        self._rotating_gif(image_size=(100, 200))

    def _rotating_gif(
        self, image_size, n_frames=50, fps=15, volume_size=(100, 100, 100)
    ):
        """
        Render a gif animation of a rotating cube/sphere (runs only if `DEBUG==True`).
        """

        if not DEBUG:
            # do not run this if debug is False
            return

        for shape in ("sphere", "cube"):
            for sample_mode in ("bilinear", "nearest"):
                volumes = init_boundary_volume(
                    volume_size=volume_size, batch_size=n_frames, shape=shape
                )[0]

                # generate camera extrinsics and intrinsics
                cameras = init_cameras(n_frames, image_size=image_size)

                # init the grid raysampler
                raysampler = MultinomialRaysampler(
                    min_x=0.5,
                    max_x=image_size[1] - 0.5,
                    min_y=0.5,
                    max_y=image_size[0] - 0.5,
                    image_width=image_size[1],
                    image_height=image_size[0],
                    n_pts_per_ray=256,
                    min_depth=0.5,
                    max_depth=2.0,
                )

                # get the EA raymarcher
                raymarcher = EmissionAbsorptionRaymarcher()

                # initialize the renderer
                renderer = VolumeRenderer(
                    raysampler=raysampler,
                    raymarcher=raymarcher,
                    sample_mode=sample_mode,
                )

                # run the renderer
                images_opacities = renderer(cameras=cameras, volumes=volumes)[0]

                # split output to the alpha channel and rendered images
                images, opacities = images_opacities[..., :3], images_opacities[..., 3]

                # export the gif
                outdir = tempfile.gettempdir() + "/test_volume_renderer_gifs"
                os.makedirs(outdir, exist_ok=True)
                frames = []
                for image, opacity in zip(images, opacities):
                    image_pil = Image.fromarray(
                        (
                            torch.cat(
                                (image, opacity[..., None].repeat(1, 1, 3)), dim=1
                            )
                            .detach()
                            .cpu()
                            .numpy()
                            * 255.0
                        ).astype(np.uint8)
                    )
                    frames.append(image_pil)
                outfile = os.path.join(outdir, f"{shape}_{sample_mode}.gif")
                frames[0].save(
                    outfile,
                    save_all=True,
                    append_images=frames[1:],
                    duration=n_frames // fps,
                    loop=0,
                )
                print(f"exported {outfile}")

    def test_rotating_cube_volume_render(self):
        """
        Generates 4 renders of 4 sides of a volume representing a 3D cube.
        Since each side of the cube is homogeneously colored with
        a different color, this should result in 4 images of homogeneous color
        with the depth of each pixel equal to a constant.
        """

        # batch_size = 4 sides of the cube
        batch_size = 4
        image_size = (50, 40)

        for volume_size in ([25, 25, 25],):
            for sample_mode in ("bilinear", "nearest"):
                volume_translation = torch.zeros(4, 3)
                volume_translation.requires_grad = True
                volumes, volume_voxel_size, _ = init_boundary_volume(
                    volume_size=volume_size,
                    batch_size=batch_size,
                    shape="cube",
                    volume_translation=volume_translation,
                )

                # generate camera extrinsics and intrinsics
                cameras = init_cameras(batch_size, image_size=image_size)

                # enable the gradient caching for the camera variables
                # the list of differentiable camera vars
                cam_vars = ("R", "T", "focal_length", "principal_point")
                for cam_var in cam_vars:
                    getattr(cameras, cam_var).requires_grad = True
                # enable the grad for volume vars as well
                volumes.features().requires_grad = True
                volumes.densities().requires_grad = True

                raysampler = MultinomialRaysampler(
                    min_x=0.5,
                    max_x=image_size[1] - 0.5,
                    min_y=0.5,
                    max_y=image_size[0] - 0.5,
                    image_width=image_size[1],
                    image_height=image_size[0],
                    n_pts_per_ray=128,
                    min_depth=0.01,
                    max_depth=3.0,
                )

                raymarcher = EmissionAbsorptionRaymarcher()
                renderer = VolumeRenderer(
                    raysampler=raysampler,
                    raymarcher=raymarcher,
                    sample_mode=sample_mode,
                )
                images_opacities = renderer(cameras=cameras, volumes=volumes)[0]
                images, opacities = images_opacities[..., :3], images_opacities[..., 3]

                # check that the renderer does not erase gradients
                loss = images_opacities.sum()
                loss.backward()
                for check_var in (
                    *[getattr(cameras, cam_var) for cam_var in cam_vars],
                    volumes.features(),
                    volumes.densities(),
                    volume_translation,
                ):
                    self.assertIsNotNone(check_var.grad)

                # ao opacities should be exactly the same as the ea ones
                # we can further get the ea opacities from a feature-less
                # version of our volumes
                raymarcher_ao = AbsorptionOnlyRaymarcher()
                renderer_ao = VolumeRenderer(
                    raysampler=raysampler,
                    raymarcher=raymarcher_ao,
                    sample_mode=sample_mode,
                )
                volumes_featureless = Volumes(
                    densities=volumes.densities(),
                    volume_translation=volume_translation,
                    voxel_size=volume_voxel_size,
                )
                opacities_ao = renderer_ao(
                    cameras=cameras, volumes=volumes_featureless
                )[0][..., 0]
                self.assertClose(opacities, opacities_ao)

                # colors of the sides of the cube
                gt_clr_sides = torch.tensor(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [0.0, 1.0, 0.0],
                    ],
                    dtype=torch.float32,
                    device=images.device,
                )

                if DEBUG:
                    outdir = tempfile.gettempdir() + "/test_volume_renderer"
                    os.makedirs(outdir, exist_ok=True)
                    for imidx, (image, opacity) in enumerate(zip(images, opacities)):
                        for image_ in (image, opacity):
                            image_pil = Image.fromarray(
                                (image_.detach().cpu().numpy() * 255.0).astype(np.uint8)
                            )
                            outfile = (
                                outdir
                                + f"/rgb_{sample_mode}"
                                + f"_{str(volume_size).replace(' ', '')}"
                                + f"_{imidx:003d}"
                            )
                            if image_ is image:
                                outfile += "_rgb.png"
                            else:
                                outfile += "_opacity.png"
                            image_pil.save(outfile)
                            print(f"exported {outfile}")

                border = 10
                for image, opacity, gt_color in zip(images, opacities, gt_clr_sides):
                    image_crop = image[border:-border, border:-border]
                    opacity_crop = opacity[border:-border, border:-border]

                    # check mean and std difference from gt
                    err = (
                        (image_crop - gt_color[None, None].expand_as(image_crop))
                        .abs()
                        .mean(dim=-1)
                    )
                    zero = err.new_zeros(1)[0]
                    self.assertClose(err.mean(), zero, atol=1e-2)
                    self.assertClose(err.std(), zero, atol=1e-2)

                    err_opacity = (opacity_crop - 1.0).abs()
                    self.assertClose(err_opacity.mean(), zero, atol=1e-2)
                    self.assertClose(err_opacity.std(), zero, atol=1e-2)
