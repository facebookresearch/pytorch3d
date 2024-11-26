# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch
from pytorch3d.renderer import (
    BlendParams,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer,
    Materials,
    MeshRasterizer,
    MeshRenderer,
    MonteCarloRaysampler,
    MultinomialRaysampler,
    NDCMultinomialRaysampler,
    PointLights,
    RasterizationSettings,
    ray_bundle_to_ray_points,
    RayBundle,
    SoftPhongShader,
    TexturesVertex,
)
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere

from .common_testing import TestCaseMixin
from .test_render_volumes import init_cameras


DEBUG = False
if DEBUG:
    import os
    import tempfile

    from PIL import Image


def spherical_volumetric_function(
    ray_bundle: RayBundle,
    sphere_centroid: torch.Tensor,
    sphere_diameter: float,
    **kwargs,
):
    """
    Volumetric function of a simple RGB sphere with diameter `sphere_diameter`
    and centroid `sphere_centroid`.
    """
    # convert the ray bundle to world points
    rays_points_world = ray_bundle_to_ray_points(ray_bundle)
    batch_size = rays_points_world.shape[0]

    # surface_vectors = vectors from world coords towards the sphere centroid
    surface_vectors = (
        rays_points_world.view(batch_size, -1, 3) - sphere_centroid[:, None]
    )

    # the squared distance of each ray point to the centroid of the sphere
    surface_dist = (
        (surface_vectors**2)
        .sum(-1, keepdim=True)
        .view(*rays_points_world.shape[:-1], 1)
    )

    # set all ray densities within the sphere_diameter distance from the centroid to 1
    rays_densities = torch.sigmoid(-100.0 * (surface_dist - sphere_diameter**2))

    # ray colors are proportional to the normalized surface_vectors
    rays_features = (
        torch.nn.functional.normalize(
            surface_vectors.view(rays_points_world.shape), dim=-1
        )
        * 0.5
        + 0.5
    )

    return rays_densities, rays_features


class TestRenderImplicit(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(42)
        np.random.seed(42)

    @staticmethod
    def renderer(
        batch_size=10,
        raymarcher_type=EmissionAbsorptionRaymarcher,
        n_rays_per_image=10,
        n_pts_per_ray=10,
        sphere_diameter=0.75,
    ):
        # generate NDC camera extrinsics and intrinsics
        cameras = init_cameras(batch_size, image_size=None, ndc=True)

        # get rand offset of the volume
        sphere_centroid = torch.randn(batch_size, 3, device=cameras.device) * 0.1

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
        ).to(cameras.device)

        # get the raymarcher
        raymarcher = raymarcher_type()

        # get the implicit renderer
        renderer = ImplicitRenderer(raysampler=raysampler, raymarcher=raymarcher)

        def run_renderer():
            renderer(
                cameras=cameras,
                volumetric_function=spherical_volumetric_function,
                sphere_centroid=sphere_centroid,
                sphere_diameter=sphere_diameter,
            )

        return run_renderer

    def test_input_types(self):
        """
        Check that ValueErrors are thrown where expected.
        """
        # check the constructor
        for bad_raysampler in (None, 5, []):
            for bad_raymarcher in (None, 5, []):
                with self.assertRaises(ValueError):
                    ImplicitRenderer(
                        raysampler=bad_raysampler, raymarcher=bad_raymarcher
                    )

        # init a trivial renderer
        renderer = ImplicitRenderer(
            raysampler=NDCMultinomialRaysampler(
                image_width=100,
                image_height=100,
                n_pts_per_ray=10,
                min_depth=0.1,
                max_depth=1.0,
            ),
            raymarcher=EmissionAbsorptionRaymarcher(),
        )

        # get default cameras
        cameras = init_cameras()

        for bad_volumetric_function in (None, 5, []):
            with self.assertRaises(ValueError):
                renderer(cameras=cameras, volumetric_function=bad_volumetric_function)

    def test_compare_with_meshes_renderer(self):
        self._compare_with_meshes_renderer(image_size=(200, 100))
        self._compare_with_meshes_renderer(image_size=(100, 200))

    def _compare_with_meshes_renderer(
        self, image_size, batch_size=11, sphere_diameter=0.6
    ):
        """
        Generate a spherical RGB volumetric function and its corresponding mesh
        and check whether MeshesRenderer returns the same images as the
        corresponding ImplicitRenderer.
        """

        # generate NDC camera extrinsics and intrinsics
        cameras = init_cameras(batch_size, image_size=image_size, ndc=True)

        # get rand offset of the volume
        sphere_centroid = torch.randn(batch_size, 3, device=cameras.device) * 0.1
        sphere_centroid.requires_grad = True

        # init the grid raysampler with the ndc grid
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

        # the list of differentiable camera vars
        cam_vars = ("R", "T", "focal_length", "principal_point")
        # enable the gradient caching for the camera variables
        for cam_var in cam_vars:
            getattr(cameras_randomized, cam_var).requires_grad = True

        # get the implicit renderer
        images_opacities = ImplicitRenderer(
            raysampler=raysampler, raymarcher=raymarcher
        )(
            cameras=cameras_randomized,
            volumetric_function=spherical_volumetric_function,
            sphere_centroid=sphere_centroid,
            sphere_diameter=sphere_diameter,
        )[0]

        # check that the renderer does not erase gradients
        loss = images_opacities.sum()
        loss.backward()
        for check_var in (
            *[getattr(cameras_randomized, cam_var) for cam_var in cam_vars],
            sphere_centroid,
        ):
            self.assertIsNotNone(check_var.grad)

        # instantiate the corresponding spherical mesh
        ico = ico_sphere(level=4, device=cameras.device).extend(batch_size)
        verts = (
            torch.nn.functional.normalize(ico.verts_padded(), dim=-1) * sphere_diameter
            + sphere_centroid[:, None]
        )
        meshes = Meshes(
            verts=verts,
            faces=ico.faces_padded(),
            textures=TexturesVertex(
                verts_features=(
                    torch.nn.functional.normalize(verts, dim=-1) * 0.5 + 0.5
                )
            ),
        )

        # instantiate the corresponding mesh renderer
        lights = PointLights(device=cameras.device, location=[[0.0, 0.0, 0.0]])
        renderer_textured = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras_randomized,
                raster_settings=RasterizationSettings(
                    image_size=image_size,
                    blur_radius=1e-3,
                    faces_per_pixel=10,
                    z_clip_value=None,
                    perspective_correct=False,
                ),
            ),
            shader=SoftPhongShader(
                device=cameras.device,
                cameras=cameras_randomized,
                lights=lights,
                materials=Materials(
                    ambient_color=((2.0, 2.0, 2.0),),
                    diffuse_color=((0.0, 0.0, 0.0),),
                    specular_color=((0.0, 0.0, 0.0),),
                    shininess=64,
                    device=cameras.device,
                ),
                blend_params=BlendParams(
                    sigma=1e-3, gamma=1e-4, background_color=(0.0, 0.0, 0.0)
                ),
            ),
        )

        # get the mesh render
        images_opacities_meshes = renderer_textured(
            meshes, cameras=cameras_randomized, lights=lights
        )

        if DEBUG:
            outdir = tempfile.gettempdir() + "/test_implicit_vs_mesh_renderer"
            os.makedirs(outdir, exist_ok=True)

            frames = []
            for image_opacity, image_opacity_mesh in zip(
                images_opacities, images_opacities_meshes
            ):
                image, opacity = image_opacity.split([3, 1], dim=-1)
                image_mesh, opacity_mesh = image_opacity_mesh.split([3, 1], dim=-1)
                diff_image = (
                    ((image - image_mesh) * 0.5 + 0.5)
                    .mean(dim=2, keepdim=True)
                    .repeat(1, 1, 3)
                )
                image_pil = Image.fromarray(
                    (
                        torch.cat(
                            (
                                image,
                                image_mesh,
                                diff_image,
                                opacity.repeat(1, 1, 3),
                                opacity_mesh.repeat(1, 1, 3),
                            ),
                            dim=1,
                        )
                        .detach()
                        .cpu()
                        .numpy()
                        * 255.0
                    ).astype(np.uint8)
                )
                frames.append(image_pil)

            # export gif
            outfile = os.path.join(outdir, "implicit_vs_mesh_render.gif")
            frames[0].save(
                outfile,
                save_all=True,
                append_images=frames[1:],
                duration=batch_size // 15,
                loop=0,
            )
            print(f"exported {outfile}")

            # export concatenated frames
            outfile_cat = os.path.join(outdir, "implicit_vs_mesh_render.png")
            Image.fromarray(np.concatenate([np.array(f) for f in frames], axis=0)).save(
                outfile_cat
            )
            print(f"exported {outfile_cat}")

        # compare the renders
        diff = (images_opacities - images_opacities_meshes).abs().mean(dim=-1)
        mu_diff = diff.mean(dim=(1, 2))
        std_diff = diff.std(dim=(1, 2))
        self.assertClose(mu_diff, torch.zeros_like(mu_diff), atol=5e-2)
        self.assertClose(std_diff, torch.zeros_like(std_diff), atol=6e-2)

    def test_rotating_gif(self):
        self._rotating_gif(image_size=(200, 100))
        self._rotating_gif(image_size=(100, 200))

    def _rotating_gif(self, image_size, n_frames=50, fps=15, sphere_diameter=0.5):
        """
        Render a gif animation of a rotating sphere (runs only if `DEBUG==True`).
        """

        if not DEBUG:
            # do not run this if debug is False
            return

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
            min_depth=0.1,
            max_depth=2.0,
        )

        # get the EA raymarcher
        raymarcher = EmissionAbsorptionRaymarcher()

        # get the implicit render
        renderer = ImplicitRenderer(raysampler=raysampler, raymarcher=raymarcher)

        # get the (0) centroid of the sphere
        sphere_centroid = torch.zeros(n_frames, 3, device=cameras.device) * 0.1

        # run the renderer
        images_opacities = renderer(
            cameras=cameras,
            volumetric_function=spherical_volumetric_function,
            sphere_centroid=sphere_centroid,
            sphere_diameter=sphere_diameter,
        )[0]

        # split output to the alpha channel and rendered images
        images, opacities = images_opacities[..., :3], images_opacities[..., 3]

        # export the gif
        outdir = tempfile.gettempdir() + "/test_implicit_renderer_gifs"
        os.makedirs(outdir, exist_ok=True)
        frames = []
        for image, opacity in zip(images, opacities):
            image_pil = Image.fromarray(
                (
                    torch.cat(
                        (image, opacity[..., None].clamp(0.0, 1.0).repeat(1, 1, 3)),
                        dim=1,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    * 255.0
                ).astype(np.uint8)
            )
            frames.append(image_pil)
        outfile = os.path.join(outdir, "rotating_sphere.gif")
        frames[0].save(
            outfile,
            save_all=True,
            append_images=frames[1:],
            duration=n_frames // fps,
            loop=0,
        )
        print(f"exported {outfile}")
