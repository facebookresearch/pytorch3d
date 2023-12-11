# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""
Sanity checks for output images from the renderer.
"""
import os
import unittest
from collections import namedtuple

from itertools import product

import numpy as np
import torch
from PIL import Image
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.renderer import (
    AmbientLights,
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    look_at_view_transform,
    Materials,
    MeshRasterizer,
    MeshRenderer,
    MeshRendererWithFragments,
    OrthographicCameras,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    TexturesAtlas,
    TexturesUV,
    TexturesVertex,
)
from pytorch3d.renderer.fisheyecameras import FishEyeCameras
from pytorch3d.renderer.mesh.shader import (
    BlendParams,
    HardFlatShader,
    HardGouraudShader,
    HardPhongShader,
    SoftPhongShader,
    SoftSilhouetteShader,
    SplatterPhongShader,
    TexturedSoftPhongShader,
)
from pytorch3d.renderer.opengl import MeshRasterizerOpenGL
from pytorch3d.structures.meshes import (
    join_meshes_as_batch,
    join_meshes_as_scene,
    Meshes,
)
from pytorch3d.utils.ico_sphere import ico_sphere
from pytorch3d.utils.torus import torus

from .common_testing import (
    get_pytorch3d_dir,
    get_tests_dir,
    load_rgb_image,
    TestCaseMixin,
)

# If DEBUG=True, save out images generated in the tests for debugging.
# All saved images have prefix DEBUG_
DEBUG = False
DATA_DIR = get_tests_dir() / "data"
TUTORIAL_DATA_DIR = get_pytorch3d_dir() / "docs/tutorials/data"

RasterizerTest = namedtuple(
    "RasterizerTest", ["rasterizer", "shader", "reference_name", "debug_name"]
)


class TestRenderMeshes(TestCaseMixin, unittest.TestCase):
    def test_simple_sphere(self, elevated_camera=False, check_depth=False):
        """
        Test output of phong and gouraud shading matches a reference image using
        the default values for the light sources.

        Args:
            elevated_camera: Defines whether the camera observing the scene should
                           have an elevation of 45 degrees.
        """
        device = torch.device("cuda:0")

        # Init mesh
        sphere_mesh = ico_sphere(5, device)
        verts_padded = sphere_mesh.verts_padded()
        faces_padded = sphere_mesh.faces_padded()
        feats = torch.ones_like(verts_padded, device=device)
        textures = TexturesVertex(verts_features=feats)
        sphere_mesh = Meshes(verts=verts_padded, faces=faces_padded, textures=textures)

        # Init rasterizer settings
        if elevated_camera:
            # Elevated and rotated camera
            R, T = look_at_view_transform(dist=2.7, elev=45.0, azim=45.0)
            postfix = "_elevated_"
            # If y axis is up, the spot of light should
            # be on the bottom left of the sphere.
        else:
            # No elevation or azimuth rotation
            R, T = look_at_view_transform(2.7, 0.0, 0.0)
            postfix = "_"
        for cam_type in (
            FoVPerspectiveCameras,
            FoVOrthographicCameras,
            PerspectiveCameras,
            OrthographicCameras,
            FishEyeCameras,
        ):
            if cam_type == FishEyeCameras:
                cam_kwargs = {
                    "radial_params": torch.tensor(
                        [
                            [-1, -2, -3, 0, 0, 1],
                        ],
                        dtype=torch.float32,
                    ),
                    "tangential_params": torch.tensor(
                        [[0.7002747019, -0.4005228974]], dtype=torch.float32
                    ),
                    "thin_prism_params": torch.tensor(
                        [
                            [-1.000134884, -1.000084822, -1.0009420014, -1.0001276838],
                        ],
                        dtype=torch.float32,
                    ),
                }
                cameras = cam_type(
                    device=device,
                    R=R,
                    T=T,
                    use_tangential=True,
                    use_radial=True,
                    use_thin_prism=True,
                    world_coordinates=True,
                    **cam_kwargs,
                )
            else:
                cameras = cam_type(device=device, R=R, T=T)

            # Init shader settings
            materials = Materials(device=device)
            lights = PointLights(device=device)
            lights.location = torch.tensor([0.0, 0.0, +2.0], device=device)[None]

            raster_settings = RasterizationSettings(
                image_size=512, blur_radius=0.0, faces_per_pixel=1
            )
            blend_params = BlendParams(0.5, 1e-4, (0, 0, 0))

            # Test several shaders
            rasterizer_tests = [
                RasterizerTest(MeshRasterizer, HardPhongShader, "phong", "hard_phong"),
                RasterizerTest(
                    MeshRasterizer, HardGouraudShader, "gouraud", "hard_gouraud"
                ),
                RasterizerTest(MeshRasterizer, HardFlatShader, "flat", "hard_flat"),
                RasterizerTest(
                    MeshRasterizerOpenGL,
                    SplatterPhongShader,
                    "splatter",
                    "splatter_phong",
                ),
            ]
            for test in rasterizer_tests:
                shader = test.shader(
                    lights=lights,
                    cameras=cameras,
                    materials=materials,
                    blend_params=blend_params,
                )
                if test.rasterizer == MeshRasterizer:
                    rasterizer = test.rasterizer(
                        cameras=cameras, raster_settings=raster_settings
                    )
                elif test.rasterizer == MeshRasterizerOpenGL:
                    if type(cameras) in [
                        PerspectiveCameras,
                        OrthographicCameras,
                        FishEyeCameras,
                    ]:
                        # MeshRasterizerOpenGL is only compatible with FoV cameras.
                        continue
                    rasterizer = test.rasterizer(
                        cameras=cameras,
                        raster_settings=raster_settings,
                    )

                if check_depth:
                    renderer = MeshRendererWithFragments(
                        rasterizer=rasterizer, shader=shader
                    )
                    images, fragments = renderer(sphere_mesh)
                    self.assertClose(fragments.zbuf, rasterizer(sphere_mesh).zbuf)
                    # Check the alpha channel is the mask. For soft rasterizers, the
                    # boundary will not match exactly so we use quantiles to compare.
                    self.assertLess(
                        (
                            images[..., -1]
                            - (fragments.pix_to_face[..., 0] >= 0).float()
                        ).quantile(0.99),
                        0.005,
                    )
                else:
                    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
                    images = renderer(sphere_mesh)

                rgb = images[0, ..., :3].squeeze().cpu()
                filename = "simple_sphere_light_%s%s%s.png" % (
                    test.reference_name,
                    postfix,
                    cam_type.__name__,
                )

                image_ref = load_rgb_image("test_%s" % filename, DATA_DIR)
                if DEBUG:
                    debug_filename = "simple_sphere_light_%s%s%s.png" % (
                        test.debug_name,
                        postfix,
                        cam_type.__name__,
                    )
                    filename = "DEBUG_%s" % debug_filename
                    Image.fromarray((rgb.numpy() * 255).astype(np.uint8)).save(
                        DATA_DIR / filename
                    )
                self.assertClose(rgb, image_ref, atol=0.05)

            ########################################################
            # Move the light to the +z axis in world space so it is
            # behind the sphere. Note that +Z is in, +Y up,
            # +X left for both world and camera space.
            ########################################################
            lights.location[..., 2] = -2.0
            phong_shader = HardPhongShader(
                lights=lights,
                cameras=cameras,
                materials=materials,
                blend_params=blend_params,
            )
            if check_depth:
                phong_renderer = MeshRendererWithFragments(
                    rasterizer=rasterizer, shader=phong_shader
                )
                images, fragments = phong_renderer(sphere_mesh, lights=lights)
                self.assertClose(
                    fragments.zbuf, rasterizer(sphere_mesh, lights=lights).zbuf
                )
                # Check the alpha channel is the mask
                self.assertLess(
                    (
                        images[..., -1] - (fragments.pix_to_face[..., 0] >= 0).float()
                    ).quantile(0.99),
                    0.005,
                )
            else:
                phong_renderer = MeshRenderer(
                    rasterizer=rasterizer, shader=phong_shader
                )
                images = phong_renderer(sphere_mesh, lights=lights)
            rgb = images[0, ..., :3].squeeze().cpu()
            if DEBUG:
                filename = "DEBUG_simple_sphere_dark%s%s.png" % (
                    postfix,
                    cam_type.__name__,
                )
                Image.fromarray((rgb.numpy() * 255).astype(np.uint8)).save(
                    DATA_DIR / filename
                )

            image_ref_phong_dark = load_rgb_image(
                "test_simple_sphere_dark%s%s.png" % (postfix, cam_type.__name__),
                DATA_DIR,
            )
            # Soft shaders (SplatterPhong) will have a different boundary than hard
            # ones, but should be identical otherwise.
            self.assertLess((rgb - image_ref_phong_dark).quantile(0.99), 0.005)

    def test_simple_sphere_elevated_camera(self):
        """
        Test output of phong and gouraud shading matches a reference image using
        the default values for the light sources.

        The rendering is performed with a camera that has non-zero elevation.
        """
        self.test_simple_sphere(elevated_camera=True)

    def test_simple_sphere_depth(self):
        """
        Test output of phong and gouraud shading matches a reference image using
        the default values for the light sources.

        The rendering is performed with a camera that has non-zero elevation.
        """
        self.test_simple_sphere(check_depth=True)

    def test_simple_sphere_screen(self):

        """
        Test output when rendering with PerspectiveCameras & OrthographicCameras
        in NDC vs screen space.
        """
        device = torch.device("cuda:0")

        # Init mesh
        sphere_mesh = ico_sphere(5, device)
        verts_padded = sphere_mesh.verts_padded()
        faces_padded = sphere_mesh.faces_padded()
        feats = torch.ones_like(verts_padded, device=device)
        textures = TexturesVertex(verts_features=feats)
        sphere_mesh = Meshes(verts=verts_padded, faces=faces_padded, textures=textures)

        R, T = look_at_view_transform(2.7, 0.0, 0.0)

        # Init shader settings
        materials = Materials(device=device)
        lights = PointLights(device=device)
        lights.location = torch.tensor([0.0, 0.0, +2.0], device=device)[None]

        raster_settings = RasterizationSettings(
            image_size=512, blur_radius=0.0, faces_per_pixel=1
        )
        half_half = (512.0 / 2.0, 512.0 / 2.0)
        for cam_type in (PerspectiveCameras, OrthographicCameras):
            cameras = cam_type(
                device=device,
                R=R,
                T=T,
                principal_point=(half_half,),
                focal_length=(half_half,),
                image_size=((512, 512),),
                in_ndc=False,
            )
            rasterizer = MeshRasterizer(
                cameras=cameras, raster_settings=raster_settings
            )
            blend_params = BlendParams(1e-4, 1e-4, (0, 0, 0))

            shader = HardPhongShader(
                lights=lights,
                cameras=cameras,
                materials=materials,
                blend_params=blend_params,
            )
            renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
            images = renderer(sphere_mesh)
            rgb = images[0, ..., :3].squeeze().cpu()
            filename = "test_simple_sphere_light_phong_%s.png" % cam_type.__name__
            if DEBUG:
                Image.fromarray((rgb.numpy() * 255).astype(np.uint8)).save(
                    DATA_DIR / f"{filename}_.png"
                )

            image_ref = load_rgb_image(filename, DATA_DIR)
            self.assertClose(rgb, image_ref, atol=0.05)

    def test_simple_sphere_batched(self):
        """
        Test a mesh with vertex textures can be extended to form a batch, and
        is rendered correctly with Phong, Gouraud and Flat Shaders with batched
        lighting and hard and soft blending.
        """
        batch_size = 3
        device = torch.device("cuda:0")

        # Init mesh with vertex textures.
        sphere_meshes = ico_sphere(3, device).extend(batch_size)
        verts_padded = sphere_meshes.verts_padded()
        faces_padded = sphere_meshes.faces_padded()
        feats = torch.ones_like(verts_padded, device=device)
        textures = TexturesVertex(verts_features=feats)
        sphere_meshes = Meshes(
            verts=verts_padded, faces=faces_padded, textures=textures
        )

        # Init rasterizer settings
        dist = torch.tensor([2, 4, 6]).to(device)
        elev = torch.zeros_like(dist)
        azim = torch.zeros_like(dist)
        R, T = look_at_view_transform(dist, elev, azim)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        raster_settings = RasterizationSettings(
            image_size=512, blur_radius=0.0, faces_per_pixel=4
        )

        # Init shader settings
        materials = Materials(device=device)
        lights_location = torch.tensor([0.0, 0.0, +2.0], device=device)
        lights_location = lights_location[None].expand(batch_size, -1)
        lights = PointLights(device=device, location=lights_location)
        blend_params = BlendParams(0.5, 1e-4, (0, 0, 0))

        # Init renderer
        rasterizer_tests = [
            RasterizerTest(MeshRasterizer, HardPhongShader, "phong", "hard_phong"),
            RasterizerTest(
                MeshRasterizer, HardGouraudShader, "gouraud", "hard_gouraud"
            ),
            RasterizerTest(MeshRasterizer, HardFlatShader, "flat", "hard_flat"),
            RasterizerTest(
                MeshRasterizerOpenGL,
                SplatterPhongShader,
                "splatter",
                "splatter_phong",
            ),
        ]
        for test in rasterizer_tests:
            reference_name = test.reference_name
            debug_name = test.debug_name
            rasterizer = test.rasterizer(
                cameras=cameras, raster_settings=raster_settings
            )

            shader = test.shader(
                lights=lights,
                cameras=cameras,
                materials=materials,
                blend_params=blend_params,
            )
            renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
            images = renderer(sphere_meshes)
            for i in range(batch_size):
                image_ref = load_rgb_image(
                    "test_simple_sphere_batched_%s_%s_%s.png"
                    % (reference_name, type(cameras).__name__, i),
                    DATA_DIR,
                )
                rgb = images[i, ..., :3].squeeze().cpu()
                if DEBUG:
                    filename = "DEBUG_simple_sphere_batched_%s_%s_%s.png" % (
                        debug_name,
                        type(cameras).__name__,
                        i,
                    )
                    Image.fromarray((rgb.numpy() * 255).astype(np.uint8)).save(
                        DATA_DIR / filename
                    )
                self.assertClose(rgb, image_ref, atol=0.05)

    def test_silhouette_with_grad(self):
        """
        Test silhouette blending. Also check that gradient calculation works.
        """
        device = torch.device("cuda:0")
        sphere_mesh = ico_sphere(5, device)
        verts, faces = sphere_mesh.get_mesh_verts_faces(0)
        sphere_mesh = Meshes(verts=[verts], faces=[faces])

        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend_params.sigma,
            faces_per_pixel=80,
            clip_barycentric_coords=True,
        )

        # Init rasterizer settings
        R, T = look_at_view_transform(2.7, 0, 0)
        for cam_type in (
            FoVPerspectiveCameras,
            FoVOrthographicCameras,
            PerspectiveCameras,
            OrthographicCameras,
            FishEyeCameras,
        ):
            if cam_type == FishEyeCameras:
                cameras = cam_type(
                    device=device,
                    R=R,
                    T=T,
                    use_tangential=False,
                    use_radial=False,
                    use_thin_prism=False,
                    world_coordinates=True,
                )
            else:
                cameras = cam_type(device=device, R=R, T=T)

            # Init renderer
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras, raster_settings=raster_settings
                ),
                shader=SoftSilhouetteShader(blend_params=blend_params),
            )
            images = renderer(sphere_mesh)
            alpha = images[0, ..., 3].squeeze().cpu()
            if DEBUG:
                filename = os.path.join(
                    DATA_DIR, "DEBUG_%s_silhouette.png" % (cam_type.__name__)
                )
                Image.fromarray((alpha.detach().numpy() * 255).astype(np.uint8)).save(
                    filename
                )

            ref_filename = "test_%s_silhouette.png" % (cam_type.__name__)
            image_ref_filename = DATA_DIR / ref_filename
            with Image.open(image_ref_filename) as raw_image_ref:
                image_ref = torch.from_numpy(np.array(raw_image_ref))

            image_ref = image_ref.to(dtype=torch.float32) / 255.0
            self.assertClose(alpha, image_ref, atol=0.055)

            # Check grad exist
            verts.requires_grad = True
            sphere_mesh = Meshes(verts=[verts], faces=[faces])
            images = renderer(sphere_mesh)
            images[0, ...].sum().backward()
            self.assertIsNotNone(verts.grad)

    def test_texture_map(self):
        """
        Test a mesh with a texture map is loaded and rendered correctly.
        The pupils in the eyes of the cow should always be looking to the left.
        """
        self._texture_map_per_rasterizer(MeshRasterizer)

    def test_texture_map_opengl(self):
        """
        Test a mesh with a texture map is loaded and rendered correctly.
        The pupils in the eyes of the cow should always be looking to the left.
        """
        self._texture_map_per_rasterizer(MeshRasterizerOpenGL)

    def _texture_map_per_rasterizer(self, rasterizer_type):
        device = torch.device("cuda:0")

        obj_filename = TUTORIAL_DATA_DIR / "cow_mesh/cow.obj"

        # Load mesh + texture
        verts, faces, aux = load_obj(
            obj_filename, device=device, load_textures=True, texture_wrap=None
        )
        tex_map = list(aux.texture_images.values())[0]
        tex_map = tex_map[None, ...].to(faces.textures_idx.device)
        textures = TexturesUV(
            maps=tex_map, faces_uvs=[faces.textures_idx], verts_uvs=[aux.verts_uvs]
        )
        mesh = Meshes(verts=[verts], faces=[faces.verts_idx], textures=textures)

        # Init rasterizer settings
        R, T = look_at_view_transform(2.7, 0, 0)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

        raster_settings = RasterizationSettings(
            image_size=512, blur_radius=0.0, faces_per_pixel=1
        )

        # Init shader settings
        materials = Materials(device=device)
        lights = PointLights(device=device)

        # Place light behind the cow in world space. The front of
        # the cow is facing the -z direction.
        lights.location = torch.tensor([0.0, 0.0, 2.0], device=device)[None]

        blend_params = BlendParams(
            sigma=1e-1 if rasterizer_type == MeshRasterizer else 0.5,
            gamma=1e-4,
            background_color=torch.tensor([1.0, 1.0, 1.0], device=device),
        )
        # Init renderer
        rasterizer = rasterizer_type(cameras=cameras, raster_settings=raster_settings)
        if rasterizer_type == MeshRasterizer:
            shader = TexturedSoftPhongShader(
                lights=lights,
                cameras=cameras,
                materials=materials,
                blend_params=blend_params,
            )
        elif rasterizer_type == MeshRasterizerOpenGL:
            shader = SplatterPhongShader(
                lights=lights,
                cameras=cameras,
                materials=materials,
                blend_params=blend_params,
            )
        renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

        # Load reference image
        image_ref = load_rgb_image(
            f"test_texture_map_back_{rasterizer_type.__name__}.png", DATA_DIR
        )

        for bin_size in [0, None]:
            if rasterizer_type == MeshRasterizerOpenGL and bin_size == 0:
                # MeshRasterizerOpenGL does not use this parameter.
                continue
            # Check both naive and coarse to fine produce the same output.
            renderer.rasterizer.raster_settings.bin_size = bin_size
            images = renderer(mesh)
            rgb = images[0, ..., :3].squeeze().cpu()

            if DEBUG:
                Image.fromarray((rgb.numpy() * 255).astype(np.uint8)).save(
                    DATA_DIR / f"DEBUG_texture_map_back_{rasterizer_type.__name__}.png"
                )

            # NOTE some pixels can be flaky and will not lead to
            # `cond1` being true. Add `cond2` and check `cond1 or cond2`
            cond1 = torch.allclose(rgb, image_ref, atol=0.05)
            cond2 = ((rgb - image_ref).abs() > 0.05).sum() < 5
            # self.assertTrue(cond1 or cond2)

        # Check grad exists
        [verts] = mesh.verts_list()
        verts.requires_grad = True
        mesh2 = Meshes(verts=[verts], faces=mesh.faces_list(), textures=mesh.textures)
        images = renderer(mesh2)
        images[0, ...].sum().backward()
        self.assertIsNotNone(verts.grad)

        ##########################################
        # Check rendering of the front of the cow
        ##########################################

        R, T = look_at_view_transform(2.7, 0, 180)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

        # Move light to the front of the cow in world space
        lights.location = torch.tensor([0.0, 0.0, -2.0], device=device)[None]

        # Load reference image
        image_ref = load_rgb_image(
            f"test_texture_map_front_{rasterizer_type.__name__}.png", DATA_DIR
        )

        for bin_size in [0, None]:
            if rasterizer == MeshRasterizerOpenGL and bin_size == 0:
                # MeshRasterizerOpenGL does not use this parameter.
                continue
            # Check both naive and coarse to fine produce the same output.
            renderer.rasterizer.raster_settings.bin_size = bin_size

            images = renderer(mesh, cameras=cameras, lights=lights)
            rgb = images[0, ..., :3].squeeze().cpu()

            if DEBUG:
                Image.fromarray((rgb.numpy() * 255).astype(np.uint8)).save(
                    DATA_DIR / f"DEBUG_texture_map_front_{rasterizer_type.__name__}.png"
                )

            # NOTE some pixels can be flaky and will not lead to
            # `cond1` being true. Add `cond2` and check `cond1 or cond2`
            cond1 = torch.allclose(rgb, image_ref, atol=0.05)
            cond2 = ((rgb - image_ref).abs() > 0.05).sum() < 5
            self.assertTrue(cond1 or cond2)

        #################################
        # Add blurring to rasterization
        #################################
        if rasterizer_type == MeshRasterizer:
            # Note that MeshRasterizer can blur the images arbitrarily, however
            # MeshRasterizerOpenGL is limited by its kernel size (currently 3 px^2),
            # so this test only makes sense for MeshRasterizer.
            R, T = look_at_view_transform(2.7, 0, 180)
            cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
            # For MeshRasterizer, blurring is controlled by blur_radius. For
            # MeshRasterizerOpenGL, by sigma.
            blend_params = BlendParams(sigma=5e-4, gamma=1e-4)
            raster_settings = RasterizationSettings(
                image_size=512,
                blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend_params.sigma,
                faces_per_pixel=100,
                clip_barycentric_coords=True,
                perspective_correct=rasterizer_type.__name__ == "MeshRasterizerOpenGL",
            )

            # Load reference image
            image_ref = load_rgb_image("test_blurry_textured_rendering.png", DATA_DIR)

            for bin_size in [0, None]:
                # Check both naive and coarse to fine produce the same output.
                renderer.rasterizer.raster_settings.bin_size = bin_size

                images = renderer(
                    mesh.clone(),
                    cameras=cameras,
                    raster_settings=raster_settings,
                    blend_params=blend_params,
                )
                rgb = images[0, ..., :3].squeeze().cpu()

                if DEBUG:
                    Image.fromarray((rgb.numpy() * 255).astype(np.uint8)).save(
                        DATA_DIR / "DEBUG_blurry_textured_rendering.png"
                    )

                self.assertClose(rgb, image_ref, atol=0.05)

    def test_batch_uvs(self):
        self._batch_uvs(MeshRasterizer)

    def test_batch_uvs_opengl(self):
        self._batch_uvs(MeshRasterizer)

    def _batch_uvs(self, rasterizer_type):
        """Test that two random tori with TexturesUV render the same as each individually."""
        torch.manual_seed(1)
        device = torch.device("cuda:0")

        plain_torus = torus(r=1, R=4, sides=10, rings=10, device=device)
        [verts] = plain_torus.verts_list()
        [faces] = plain_torus.faces_list()
        nocolor = torch.zeros((100, 100), device=device)
        color_gradient = torch.linspace(0, 1, steps=100, device=device)
        color_gradient1 = color_gradient[None].expand_as(nocolor)
        color_gradient2 = color_gradient[:, None].expand_as(nocolor)
        colors1 = torch.stack([nocolor, color_gradient1, color_gradient2], dim=2)
        colors2 = torch.stack([color_gradient1, color_gradient2, nocolor], dim=2)
        verts_uvs1 = torch.rand(size=(verts.shape[0], 2), device=device)
        verts_uvs2 = torch.rand(size=(verts.shape[0], 2), device=device)

        textures1 = TexturesUV(
            maps=[colors1], faces_uvs=[faces], verts_uvs=[verts_uvs1]
        )
        textures2 = TexturesUV(
            maps=[colors2], faces_uvs=[faces], verts_uvs=[verts_uvs2]
        )
        mesh1 = Meshes(verts=[verts], faces=[faces], textures=textures1)
        mesh2 = Meshes(verts=[verts], faces=[faces], textures=textures2)
        mesh_both = join_meshes_as_batch([mesh1, mesh2])

        R, T = look_at_view_transform(10, 10, 0)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

        raster_settings = RasterizationSettings(
            image_size=128, blur_radius=0.0, faces_per_pixel=1
        )

        # Init shader settings
        lights = PointLights(device=device)
        lights.location = torch.tensor([0.0, 0.0, 2.0], device=device)[None]

        blend_params = BlendParams(
            sigma=0.5,
            gamma=1e-4,
            background_color=torch.tensor([1.0, 1.0, 1.0], device=device),
        )
        # Init renderer
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        if rasterizer_type == MeshRasterizer:
            shader = HardPhongShader(
                device=device, lights=lights, cameras=cameras, blend_params=blend_params
            )
        else:
            shader = SplatterPhongShader(
                device=device, lights=lights, cameras=cameras, blend_params=blend_params
            )

        renderer = MeshRenderer(rasterizer, shader)

        outputs = []
        for meshes in [mesh_both, mesh1, mesh2]:
            outputs.append(renderer(meshes))

        if DEBUG:
            Image.fromarray(
                (outputs[0][0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
            ).save(DATA_DIR / "test_batch_uvs0.png")
            Image.fromarray(
                (outputs[1][0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
            ).save(DATA_DIR / "test_batch_uvs1.png")
            Image.fromarray(
                (outputs[0][1, ..., :3].cpu().numpy() * 255).astype(np.uint8)
            ).save(DATA_DIR / "test_batch_uvs2.png")
            Image.fromarray(
                (outputs[2][0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
            ).save(DATA_DIR / "test_batch_uvs3.png")

            diff = torch.abs(outputs[0][0, ..., :3] - outputs[1][0, ..., :3])
            Image.fromarray(((diff > 1e-5).cpu().numpy().astype(np.uint8) * 255)).save(
                DATA_DIR / "test_batch_uvs01.png"
            )
            diff = torch.abs(outputs[0][1, ..., :3] - outputs[2][0, ..., :3])
            Image.fromarray(((diff > 1e-5).cpu().numpy().astype(np.uint8) * 255)).save(
                DATA_DIR / "test_batch_uvs23.png"
            )

        self.assertClose(outputs[0][0, ..., :3], outputs[1][0, ..., :3], atol=1e-5)
        self.assertClose(outputs[0][1, ..., :3], outputs[2][0, ..., :3], atol=1e-5)

    def test_join_uvs(self):
        self._join_uvs(MeshRasterizer)

    def test_join_uvs_opengl(self):
        self._join_uvs(MeshRasterizerOpenGL)

    def _join_uvs(self, rasterizer_type):
        """Meshes with TexturesUV joined into a scene"""
        # Test the result of rendering three tori with separate textures.
        # The expected result is consistent with rendering them each alone.
        # This tests TexturesUV.join_scene with rectangle flipping,
        # and we check the form of the merged map as well.
        torch.manual_seed(1)
        device = torch.device("cuda:0")

        R, T = look_at_view_transform(18, 0, 0)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

        raster_settings = RasterizationSettings(
            image_size=256, blur_radius=0.0, faces_per_pixel=1
        )

        lights = AmbientLights(device=device)
        blend_params = BlendParams(
            sigma=0.5,
            gamma=1e-4,
            background_color=torch.tensor([1.0, 1.0, 1.0], device=device),
        )
        rasterizer = rasterizer_type(cameras=cameras, raster_settings=raster_settings)
        if rasterizer_type == MeshRasterizer:
            shader = HardPhongShader(
                device=device, blend_params=blend_params, cameras=cameras, lights=lights
            )
        else:
            shader = SplatterPhongShader(
                device=device, blend_params=blend_params, cameras=cameras, lights=lights
            )
        renderer = MeshRenderer(rasterizer, shader)

        plain_torus = torus(r=1, R=4, sides=5, rings=6, device=device)
        [verts] = plain_torus.verts_list()
        verts_shifted1 = verts.clone()
        verts_shifted1 *= 0.5
        verts_shifted1[:, 1] += 7
        verts_shifted2 = verts.clone()
        verts_shifted2 *= 0.5
        verts_shifted2[:, 1] -= 7
        verts_shifted3 = verts.clone()
        verts_shifted3 *= 0.5
        verts_shifted3[:, 1] -= 700

        [faces] = plain_torus.faces_list()
        nocolor = torch.zeros((100, 100), device=device)
        color_gradient = torch.linspace(0, 1, steps=100, device=device)
        color_gradient1 = color_gradient[None].expand_as(nocolor)
        color_gradient2 = color_gradient[:, None].expand_as(nocolor)
        colors1 = torch.stack([nocolor, color_gradient1, color_gradient2], dim=2)
        colors2 = torch.stack([color_gradient1, color_gradient2, nocolor], dim=2)
        verts_uvs1 = torch.rand(size=(verts.shape[0], 2), device=device)
        verts_uvs2 = torch.rand(size=(verts.shape[0], 2), device=device)

        for i, align_corners, padding_mode in [
            (0, True, "border"),
            (1, False, "border"),
            (2, False, "zeros"),
        ]:
            textures1 = TexturesUV(
                maps=[colors1],
                faces_uvs=[faces],
                verts_uvs=[verts_uvs1],
                align_corners=align_corners,
                padding_mode=padding_mode,
            )

            # These downsamplings of colors2 are chosen to ensure a flip and a non flip
            # when the maps are merged.
            # We have maps of size (100, 100), (50, 99) and (99, 50).
            textures2 = TexturesUV(
                maps=[colors2[::2, :-1]],
                faces_uvs=[faces],
                verts_uvs=[verts_uvs2],
                align_corners=align_corners,
                padding_mode=padding_mode,
            )
            offset = torch.tensor([0, 0, 0.5], device=device)
            textures3 = TexturesUV(
                maps=[colors2[:-1, ::2] + offset],
                faces_uvs=[faces],
                verts_uvs=[verts_uvs2],
                align_corners=align_corners,
                padding_mode=padding_mode,
            )
            mesh1 = Meshes(verts=[verts], faces=[faces], textures=textures1)
            mesh2 = Meshes(verts=[verts_shifted1], faces=[faces], textures=textures2)
            mesh3 = Meshes(verts=[verts_shifted2], faces=[faces], textures=textures3)
            # mesh4 is like mesh1 but outside the field of view. It is here to test
            # that having another texture with the same map doesn't produce
            # two copies in the joined map.
            mesh4 = Meshes(verts=[verts_shifted3], faces=[faces], textures=textures1)
            mesh = join_meshes_as_scene([mesh1, mesh2, mesh3, mesh4])

            output = renderer(mesh)[0, ..., :3].cpu()
            output1 = renderer(mesh1)[0, ..., :3].cpu()
            output2 = renderer(mesh2)[0, ..., :3].cpu()
            output3 = renderer(mesh3)[0, ..., :3].cpu()
            # The background color is white and the objects do not overlap, so we can
            # predict the merged image by taking the minimum over every channel
            merged = torch.min(torch.min(output1, output2), output3)

            image_ref = load_rgb_image(
                f"test_joinuvs{i}_{rasterizer_type.__name__}_final.png", DATA_DIR
            )
            map_ref = load_rgb_image(f"test_joinuvs{i}_map.png", DATA_DIR)

            if DEBUG:
                Image.fromarray((output.numpy() * 255).astype(np.uint8)).save(
                    DATA_DIR
                    / f"DEBUG_test_joinuvs{i}_{rasterizer_type.__name__}_final.png"
                )
                Image.fromarray((merged.numpy() * 255).astype(np.uint8)).save(
                    DATA_DIR
                    / f"DEBUG_test_joinuvs{i}_{rasterizer_type.__name__}_merged.png"
                )

                Image.fromarray((output1.numpy() * 255).astype(np.uint8)).save(
                    DATA_DIR / f"DEBUG_test_joinuvs{i}_{rasterizer_type.__name__}_1.png"
                )
                Image.fromarray((output2.numpy() * 255).astype(np.uint8)).save(
                    DATA_DIR / f"DEBUG_test_joinuvs{i}_{rasterizer_type.__name__}_2.png"
                )
                Image.fromarray((output3.numpy() * 255).astype(np.uint8)).save(
                    DATA_DIR / f"DEBUG_test_joinuvs{i}_{rasterizer_type.__name__}_3.png"
                )
                Image.fromarray(
                    (mesh.textures.maps_padded()[0].cpu().numpy() * 255).astype(
                        np.uint8
                    )
                ).save(DATA_DIR / f"DEBUG_test_joinuvs{i}_map.png")
                Image.fromarray(
                    (mesh2.textures.maps_padded()[0].cpu().numpy() * 255).astype(
                        np.uint8
                    )
                ).save(DATA_DIR / f"DEBUG_test_joinuvs{i}_map2.png")
                Image.fromarray(
                    (mesh3.textures.maps_padded()[0].cpu().numpy() * 255).astype(
                        np.uint8
                    )
                ).save(DATA_DIR / f"DEBUG_test_joinuvs{i}_map3.png")

            self.assertClose(output, merged, atol=0.005)
            self.assertClose(output, image_ref, atol=0.005)
            self.assertClose(mesh.textures.maps_padded()[0].cpu(), map_ref, atol=0.05)

    def test_join_uvs_simple(self):
        # Example from issue #826
        a = TexturesUV(
            maps=torch.full((1, 4000, 4000, 3), 0.8),
            faces_uvs=torch.arange(300).reshape(1, 100, 3),
            verts_uvs=torch.rand(1, 300, 2) * 0.4 + 0.1,
        )
        b = TexturesUV(
            maps=torch.full((1, 2000, 2000, 3), 0.7),
            faces_uvs=torch.arange(150).reshape(1, 50, 3),
            verts_uvs=torch.rand(1, 150, 2) * 0.2 + 0.3,
        )
        self.assertEqual(a._num_faces_per_mesh, [100])
        self.assertEqual(b._num_faces_per_mesh, [50])
        c = a.join_batch([b]).join_scene()
        self.assertEqual(a._num_faces_per_mesh, [100])
        self.assertEqual(b._num_faces_per_mesh, [50])
        self.assertEqual(c._num_faces_per_mesh, [150])

        color = c.faces_verts_textures_packed()
        color1 = color[:100, :, 0].flatten()
        color2 = color[100:, :, 0].flatten()
        expect1 = color1.new_tensor(0.8)
        expect2 = color2.new_tensor(0.7)
        self.assertClose(color1.min(), expect1)
        self.assertClose(color1.max(), expect1)
        self.assertClose(color2.min(), expect2)
        self.assertClose(color2.max(), expect2)

        if DEBUG:
            from pytorch3d.vis.texture_vis import texturesuv_image_PIL as PI

            PI(a, radius=5).save(DATA_DIR / "test_join_uvs_simple_a.png")
            PI(b, radius=5).save(DATA_DIR / "test_join_uvs_simple_b.png")
            PI(c, radius=5).save(DATA_DIR / "test_join_uvs_simple_c.png")

    def test_join_verts(self):
        self._join_verts(MeshRasterizer)

    def test_join_verts_opengl(self):
        self._join_verts(MeshRasterizerOpenGL)

    def _join_verts(self, rasterizer_type):
        """Meshes with TexturesVertex joined into a scene"""
        # Test the result of rendering two tori with separate textures.
        # The expected result is consistent with rendering them each alone.
        torch.manual_seed(1)
        device = torch.device("cuda:0")

        plain_torus = torus(r=1, R=4, sides=5, rings=6, device=device)
        [verts] = plain_torus.verts_list()
        verts_shifted1 = verts.clone()
        verts_shifted1 *= 0.5
        verts_shifted1[:, 1] += 7

        faces = plain_torus.faces_list()
        textures1 = TexturesVertex(verts_features=[torch.rand_like(verts)])
        textures2 = TexturesVertex(verts_features=[torch.rand_like(verts)])
        mesh1 = Meshes(verts=[verts], faces=faces, textures=textures1)
        mesh2 = Meshes(verts=[verts_shifted1], faces=faces, textures=textures2)
        mesh = join_meshes_as_scene([mesh1, mesh2])

        R, T = look_at_view_transform(18, 0, 0)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

        raster_settings = RasterizationSettings(
            image_size=256, blur_radius=0.0, faces_per_pixel=1
        )

        lights = AmbientLights(device=device)
        blend_params = BlendParams(
            sigma=0.5,
            gamma=1e-4,
            background_color=torch.tensor([1.0, 1.0, 1.0], device=device),
        )
        rasterizer = rasterizer_type(cameras=cameras, raster_settings=raster_settings)
        if rasterizer_type == MeshRasterizer:
            shader = HardPhongShader(
                device=device, blend_params=blend_params, cameras=cameras, lights=lights
            )
        else:
            shader = SplatterPhongShader(
                device=device, blend_params=blend_params, cameras=cameras, lights=lights
            )

        renderer = MeshRenderer(rasterizer, shader)

        output = renderer(mesh)

        image_ref = load_rgb_image(
            f"test_joinverts_final_{rasterizer_type.__name__}.png", DATA_DIR
        )

        if DEBUG:
            debugging_outputs = []
            for mesh_ in [mesh1, mesh2]:
                debugging_outputs.append(renderer(mesh_))
            Image.fromarray(
                (output[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
            ).save(
                DATA_DIR / f"DEBUG_test_joinverts_final_{rasterizer_type.__name__}.png"
            )
            Image.fromarray(
                (debugging_outputs[0][0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
            ).save(DATA_DIR / "DEBUG_test_joinverts_1.png")
            Image.fromarray(
                (debugging_outputs[1][0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
            ).save(DATA_DIR / "DEBUG_test_joinverts_2.png")

        result = output[0, ..., :3].cpu()
        self.assertClose(result, image_ref, atol=0.05)

    def test_join_atlas(self):
        self._join_atlas(MeshRasterizer)

    def test_join_atlas_opengl(self):
        self._join_atlas(MeshRasterizerOpenGL)

    def _join_atlas(self, rasterizer_type):
        """Meshes with TexturesAtlas joined into a scene"""
        # Test the result of rendering two tori with separate textures.
        # The expected result is consistent with rendering them each alone.
        torch.manual_seed(1)
        device = torch.device("cuda:0")

        plain_torus = torus(r=1, R=4, sides=5, rings=6, device=device)
        [verts] = plain_torus.verts_list()
        verts_shifted1 = verts.clone()
        verts_shifted1 *= 1.2
        verts_shifted1[:, 0] += 4
        verts_shifted1[:, 1] += 5
        verts[:, 0] -= 4
        verts[:, 1] -= 4

        [faces] = plain_torus.faces_list()
        map_size = 3
        # Two random atlases.
        # The averaging of the random numbers here is not consistent with the
        # meaning of the atlases, but makes each face a bit smoother than
        # if everything had a random color.
        atlas1 = torch.rand(size=(faces.shape[0], map_size, map_size, 3), device=device)
        atlas1[:, 1] = 0.5 * atlas1[:, 0] + 0.5 * atlas1[:, 2]
        atlas1[:, :, 1] = 0.5 * atlas1[:, :, 0] + 0.5 * atlas1[:, :, 2]
        atlas2 = torch.rand(size=(faces.shape[0], map_size, map_size, 3), device=device)
        atlas2[:, 1] = 0.5 * atlas2[:, 0] + 0.5 * atlas2[:, 2]
        atlas2[:, :, 1] = 0.5 * atlas2[:, :, 0] + 0.5 * atlas2[:, :, 2]

        textures1 = TexturesAtlas(atlas=[atlas1])
        textures2 = TexturesAtlas(atlas=[atlas2])
        mesh1 = Meshes(verts=[verts], faces=[faces], textures=textures1)
        mesh2 = Meshes(verts=[verts_shifted1], faces=[faces], textures=textures2)
        self.assertEqual(textures1._num_faces_per_mesh, [len(faces)])
        self.assertEqual(textures2._num_faces_per_mesh, [len(faces)])
        mesh_joined = join_meshes_as_scene([mesh1, mesh2])
        self.assertEqual(textures1._num_faces_per_mesh, [len(faces)])
        self.assertEqual(textures2._num_faces_per_mesh, [len(faces)])
        self.assertEqual(mesh_joined.textures._num_faces_per_mesh, [len(faces) * 2])

        R, T = look_at_view_transform(18, 0, 0)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

        raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=rasterizer_type.__name__ == "MeshRasterizerOpenGL",
        )

        lights = AmbientLights(device=device)
        blend_params = BlendParams(
            sigma=0.5,
            gamma=1e-4,
            background_color=torch.tensor([1.0, 1.0, 1.0], device=device),
        )

        rasterizer = rasterizer_type(cameras=cameras, raster_settings=raster_settings)
        if rasterizer_type == MeshRasterizer:
            shader = HardPhongShader(
                device=device, blend_params=blend_params, cameras=cameras, lights=lights
            )
        else:
            shader = SplatterPhongShader(
                device=device, blend_params=blend_params, cameras=cameras, lights=lights
            )

        renderer = MeshRenderer(rasterizer, shader)

        output = renderer(mesh_joined)

        image_ref = load_rgb_image(
            f"test_joinatlas_final_{rasterizer_type.__name__}.png", DATA_DIR
        )

        if DEBUG:
            debugging_outputs = []
            for mesh_ in [mesh1, mesh2]:
                debugging_outputs.append(renderer(mesh_))
            Image.fromarray(
                (output[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
            ).save(
                DATA_DIR / f"DEBUG_test_joinatlas_final_{rasterizer_type.__name__}.png"
            )
            Image.fromarray(
                (debugging_outputs[0][0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
            ).save(DATA_DIR / f"test_joinatlas_1_{rasterizer_type.__name__}.png")
            Image.fromarray(
                (debugging_outputs[1][0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
            ).save(DATA_DIR / f"test_joinatlas_2_{rasterizer_type.__name__}.png")

        result = output[0, ..., :3].cpu()
        self.assertClose(result, image_ref, atol=0.05)

    def test_joined_spheres(self):
        self._joined_spheres(MeshRasterizer)

    def test_joined_spheres_opengl(self):
        self._joined_spheres(MeshRasterizerOpenGL)

    def _joined_spheres(self, rasterizer_type):
        """
        Test a list of Meshes can be joined as a single mesh and
        the single mesh is rendered correctly with Phong, Gouraud
        and Flat Shaders.
        """
        device = torch.device("cuda:0")

        # Init mesh with vertex textures.
        # Initialize a list containing two ico spheres of different sizes.
        sphere_list = [ico_sphere(3, device), ico_sphere(4, device)]
        # [(42 verts, 80 faces), (162 verts, 320 faces)]
        # The scale the vertices need to be set at to resize the spheres
        scales = [0.25, 1]
        # The distance the spheres ought to be offset horizontally to prevent overlap.
        offsets = [1.2, -0.3]
        # Initialize a list containing the adjusted sphere meshes.
        sphere_mesh_list = []
        for i in range(len(sphere_list)):
            verts = sphere_list[i].verts_padded() * scales[i]
            verts[0, :, 0] += offsets[i]
            sphere_mesh_list.append(
                Meshes(verts=verts, faces=sphere_list[i].faces_padded())
            )
        joined_sphere_mesh = join_meshes_as_scene(sphere_mesh_list)
        joined_sphere_mesh.textures = TexturesVertex(
            verts_features=torch.ones_like(joined_sphere_mesh.verts_padded())
        )

        # Init rasterizer settings
        R, T = look_at_view_transform(2.7, 0.0, 0.0)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=rasterizer_type.__name__ == "MeshRasterizerOpenGL",
        )

        # Init shader settings
        materials = Materials(device=device)
        lights = PointLights(device=device)
        lights.location = torch.tensor([0.0, 0.0, +2.0], device=device)[None]
        blend_params = BlendParams(0.5, 1e-4, (0, 0, 0))

        # Init renderer
        rasterizer = rasterizer_type(cameras=cameras, raster_settings=raster_settings)
        shaders = {
            "phong": HardPhongShader,
            "gouraud": HardGouraudShader,
            "flat": HardFlatShader,
            "splatter": SplatterPhongShader,
        }
        for (name, shader_init) in shaders.items():
            if rasterizer_type == MeshRasterizerOpenGL and name != "splatter":
                continue
            if rasterizer_type == MeshRasterizer and name == "splatter":
                continue

            shader = shader_init(
                lights=lights,
                cameras=cameras,
                materials=materials,
                blend_params=blend_params,
            )
            renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
            image = renderer(joined_sphere_mesh)
            rgb = image[..., :3].squeeze().cpu()
            if DEBUG:
                file_name = "DEBUG_joined_spheres_%s.png" % name
                Image.fromarray((rgb.numpy() * 255).astype(np.uint8)).save(
                    DATA_DIR / file_name
                )
            image_ref = load_rgb_image("test_joined_spheres_%s.png" % name, DATA_DIR)
            self.assertClose(rgb, image_ref, atol=0.05)

    def test_texture_map_atlas(self):
        self._texture_map_atlas(MeshRasterizer)

    def test_texture_map_atlas_opengl(self):
        self._texture_map_atlas(MeshRasterizerOpenGL)

    def _texture_map_atlas(self, rasterizer_type):
        """
        Test a mesh with a texture map as a per face atlas is loaded and rendered correctly.
        Also check that the backward pass for texture atlas rendering is differentiable.
        """
        device = torch.device("cuda:0")

        obj_filename = TUTORIAL_DATA_DIR / "cow_mesh/cow.obj"

        # Load mesh and texture as a per face texture atlas.
        verts, faces, aux = load_obj(
            obj_filename,
            device=device,
            load_textures=True,
            create_texture_atlas=True,
            texture_atlas_size=8,
            texture_wrap=None,
        )
        atlas = aux.texture_atlas
        mesh = Meshes(
            verts=[verts],
            faces=[faces.verts_idx],
            textures=TexturesAtlas(atlas=[atlas]),
        )

        # Init rasterizer settings
        R, T = look_at_view_transform(2.7, 0, 0)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

        raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=0.0,
            faces_per_pixel=1,
            cull_backfaces=True,
            perspective_correct=rasterizer_type.__name__ == "MeshRasterizerOpenGL",
        )

        # Init shader settings
        materials = Materials(device=device, specular_color=((0, 0, 0),), shininess=0.0)
        blend_params = BlendParams(0.5, 1e-4, (1.0, 1.0, 1.0))
        lights = PointLights(device=device)

        # Place light behind the cow in world space. The front of
        # the cow is facing the -z direction.
        lights.location = torch.tensor([0.0, 0.0, 2.0], device=device)[None]

        # The HardPhongShader can be used directly with atlas textures.
        rasterizer = rasterizer_type(cameras=cameras, raster_settings=raster_settings)
        if rasterizer_type == MeshRasterizer:
            shader = HardPhongShader(
                device=device,
                blend_params=blend_params,
                cameras=cameras,
                lights=lights,
                materials=materials,
            )
        else:
            shader = SplatterPhongShader(
                device=device,
                blend_params=blend_params,
                cameras=cameras,
                lights=lights,
                materials=materials,
            )

        renderer = MeshRenderer(rasterizer, shader)

        images = renderer(mesh)
        rgb = images[0, ..., :3].squeeze()

        # Load reference image
        image_ref = load_rgb_image(
            f"test_texture_atlas_8x8_back_{rasterizer_type.__name__}.png", DATA_DIR
        )

        if DEBUG:
            Image.fromarray((rgb.detach().cpu().numpy() * 255).astype(np.uint8)).save(
                DATA_DIR
                / f"DEBUG_texture_atlas_8x8_back_{rasterizer_type.__name__}.png"
            )

        self.assertClose(rgb.cpu(), image_ref, atol=0.05)

        # Check gradients are propagated
        # correctly back to the texture atlas.
        # Because of how texture sampling is implemented
        # for the texture atlas it is not possible to get
        # gradients back to the vertices.
        atlas.requires_grad = True
        mesh = Meshes(
            verts=[verts],
            faces=[faces.verts_idx],
            textures=TexturesAtlas(atlas=[atlas]),
        )
        raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=0.0001,
            faces_per_pixel=5 if rasterizer_type.__name__ == "MeshRasterizer" else 1,
            cull_backfaces=rasterizer_type.__name__ == "MeshRasterizer",
            clip_barycentric_coords=True,
        )
        images = renderer(mesh, raster_settings=raster_settings)
        images[0, ...].sum().backward()

        fragments = rasterizer(mesh, raster_settings=raster_settings)
        if rasterizer_type == MeshRasterizer:
            # Some of the bary coordinates are outside the
            # [0, 1] range as expected because the blur is > 0.
            self.assertTrue(fragments.bary_coords.ge(1.0).any())
        self.assertIsNotNone(atlas.grad)
        self.assertTrue(atlas.grad.sum().abs() > 0.0)

    def test_simple_sphere_outside_zfar(self):
        self._simple_sphere_outside_zfar(MeshRasterizer)

    def test_simple_sphere_outside_zfar_opengl(self):
        self._simple_sphere_outside_zfar(MeshRasterizerOpenGL)

    def _simple_sphere_outside_zfar(self, rasterizer_type):
        """
        Test output when rendering a sphere that is beyond zfar with a SoftPhongShader.
        This renders a sphere of radius 500, with the camera at x=1500 for different
        settings of zfar.  This is intended to check 1) setting cameras.zfar propagates
        to the blender and that the rendered sphere is (soft) clipped if it is beyond
        zfar, 2) make sure there are no numerical precision/overflow errors associated
        with larger world coordinates
        """
        device = torch.device("cuda:0")

        # Init mesh
        sphere_mesh = ico_sphere(5, device)
        verts_padded = sphere_mesh.verts_padded() * 500
        faces_padded = sphere_mesh.faces_padded()
        feats = torch.ones_like(verts_padded, device=device)
        textures = TexturesVertex(verts_features=feats)
        sphere_mesh = Meshes(verts=verts_padded, faces=faces_padded, textures=textures)

        R, T = look_at_view_transform(1500, 0.0, 0.0)

        # Init shader settings
        materials = Materials(device=device)
        lights = PointLights(device=device)
        lights.location = torch.tensor([0.0, 0.0, +1000.0], device=device)[None]

        raster_settings = RasterizationSettings(
            image_size=256, blur_radius=0.0, faces_per_pixel=1
        )
        for zfar in (10000.0, 100.0):
            cameras = FoVPerspectiveCameras(
                device=device, R=R, T=T, aspect_ratio=1.0, fov=60.0, zfar=zfar
            )
            blend_params = BlendParams(
                1e-4 if rasterizer_type == MeshRasterizer else 0.5, 1e-4, (0, 0, 1.0)
            )
            rasterizer = rasterizer_type(
                cameras=cameras, raster_settings=raster_settings
            )
            if rasterizer_type == MeshRasterizer:
                shader = SoftPhongShader(
                    blend_params=blend_params,
                    cameras=cameras,
                    lights=lights,
                    materials=materials,
                )
            else:
                shader = SplatterPhongShader(
                    device=device,
                    blend_params=blend_params,
                    cameras=cameras,
                    lights=lights,
                    materials=materials,
                )
            renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
            images = renderer(sphere_mesh)
            rgb = images[0, ..., :3].squeeze().cpu()

            filename = (
                "test_simple_sphere_outside_zfar_"
                f"{int(zfar)}_{rasterizer_type.__name__}.png"
            )

            # Load reference image
            image_ref = load_rgb_image(filename, DATA_DIR)

            if DEBUG:
                Image.fromarray((rgb.numpy() * 255).astype(np.uint8)).save(
                    DATA_DIR / ("DEBUG_" + filename)
                )

            self.assertClose(rgb, image_ref, atol=0.05)

    def test_cameras_kwarg(self):
        """
        Test that when cameras are passed in as a kwarg the rendering
        works as expected
        """
        device = torch.device("cuda:0")

        # Init mesh
        sphere_mesh = ico_sphere(5, device)
        verts_padded = sphere_mesh.verts_padded()
        faces_padded = sphere_mesh.faces_padded()
        feats = torch.ones_like(verts_padded, device=device)
        textures = TexturesVertex(verts_features=feats)
        sphere_mesh = Meshes(verts=verts_padded, faces=faces_padded, textures=textures)

        # No elevation or azimuth rotation
        rasterizer_tests = [
            RasterizerTest(MeshRasterizer, HardPhongShader, "phong", "hard_phong"),
            RasterizerTest(
                MeshRasterizerOpenGL,
                SplatterPhongShader,
                "splatter",
                "splatter_phong",
            ),
        ]
        R, T = look_at_view_transform(2.7, 0.0, 0.0)
        for cam_type in (
            FoVPerspectiveCameras,
            FoVOrthographicCameras,
            PerspectiveCameras,
            OrthographicCameras,
        ):
            for test in rasterizer_tests:
                if test.rasterizer == MeshRasterizerOpenGL and cam_type in [
                    PerspectiveCameras,
                    OrthographicCameras,
                ]:
                    # MeshRasterizerOpenGL only works with FoV cameras.
                    continue

                cameras = cam_type(device=device, R=R, T=T)

                # Init shader settings
                materials = Materials(device=device)
                lights = PointLights(device=device)
                lights.location = torch.tensor([0.0, 0.0, +2.0], device=device)[None]

                raster_settings = RasterizationSettings(
                    image_size=512, blur_radius=0.0, faces_per_pixel=1
                )
                rasterizer = test.rasterizer(raster_settings=raster_settings)
                blend_params = BlendParams(0.5, 1e-4, (0, 0, 0))
                shader = test.shader(
                    lights=lights, materials=materials, blend_params=blend_params
                )
                renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

                # Cameras can be passed into the renderer in the forward pass
                images = renderer(sphere_mesh, cameras=cameras)
                rgb = images.squeeze()[..., :3].cpu().numpy()
                image_ref = load_rgb_image(
                    f"test_simple_sphere_light_{test.reference_name}_{cam_type.__name__}.png",
                    DATA_DIR,
                )
                self.assertClose(rgb, image_ref, atol=0.05)

    def test_nd_sphere(self):
        """
        Test that the render can handle textures with more than 3 channels and
        not just 3 channel RGB.
        """
        torch.manual_seed(1)
        device = torch.device("cuda:0")
        C = 5
        WHITE = ((1.0,) * C,)
        BLACK = ((0.0,) * C,)

        # Init mesh
        sphere_mesh = ico_sphere(5, device)
        verts_padded = sphere_mesh.verts_padded()
        faces_padded = sphere_mesh.faces_padded()
        feats = torch.ones(*verts_padded.shape[:-1], C, device=device)
        n_verts = feats.shape[1]
        # make some non-uniform pattern
        feats *= torch.arange(0, 10, step=10 / n_verts, device=device).unsqueeze(1)
        textures = TexturesVertex(verts_features=feats)
        sphere_mesh = Meshes(verts=verts_padded, faces=faces_padded, textures=textures)

        # No elevation or azimuth rotation
        R, T = look_at_view_transform(2.7, 0.0, 0.0)

        cameras = PerspectiveCameras(device=device, R=R, T=T)

        # Init shader settings
        materials = Materials(
            device=device,
            ambient_color=WHITE,
            diffuse_color=WHITE,
            specular_color=WHITE,
        )
        lights = AmbientLights(
            device=device,
            ambient_color=WHITE,
        )
        lights.location = torch.tensor([0.0, 0.0, +2.0], device=device)[None]

        raster_settings = RasterizationSettings(
            image_size=512, blur_radius=0.0, faces_per_pixel=1
        )
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        blend_params = BlendParams(
            1e-4,
            1e-4,
            background_color=BLACK[0],
        )

        # only test HardFlatShader since that's the only one that makes
        # sense for classification
        shader = HardFlatShader(
            lights=lights,
            cameras=cameras,
            materials=materials,
            blend_params=blend_params,
        )
        renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
        images = renderer(sphere_mesh)

        self.assertEqual(images.shape[-1], C + 1)
        self.assertClose(images.amax(), torch.tensor(10.0), atol=0.01)
        self.assertClose(images.amin(), torch.tensor(0.0), atol=0.01)

        # grab last 3 color channels
        rgb = (images[0, ..., C - 3 : C] / 10).squeeze().cpu()
        filename = "test_nd_sphere.png"

        if DEBUG:
            debug_filename = "DEBUG_%s" % filename
            Image.fromarray((rgb.numpy() * 255).astype(np.uint8)).save(
                DATA_DIR / debug_filename
            )

        image_ref = load_rgb_image(filename, DATA_DIR)
        self.assertClose(rgb, image_ref, atol=0.05)

    def test_simple_sphere_fisheye_params(self):
        """
        Test output of phong and gouraud shading matches a reference image using
        the default values for the light sources.

        """
        device = torch.device("cuda:0")

        # Init mesh
        sphere_mesh = ico_sphere(5, device)
        verts_padded = sphere_mesh.verts_padded()
        faces_padded = sphere_mesh.faces_padded()
        feats = torch.ones_like(verts_padded, device=device)
        textures = TexturesVertex(verts_features=feats)
        sphere_mesh = Meshes(verts=verts_padded, faces=faces_padded, textures=textures)

        # Init rasterizer settings
        R, T = look_at_view_transform(2.7, 0.0, 0.0)
        postfix = "_"

        cam_kwargs = [
            {
                "radial_params": torch.tensor(
                    [
                        [-1, -2, -3, 0, 0, 1],
                    ],
                    dtype=torch.float32,
                ),
            },
            {
                "tangential_params": torch.tensor(
                    [[0.7002747019, -0.4005228974]], dtype=torch.float32
                ),
            },
            {
                "thin_prism_params": torch.tensor(
                    [
                        [
                            -1.000134884,
                            -1.000084822,
                            -1.0009420014,
                            -1.0001276838,
                        ],
                    ],
                    dtype=torch.float32,
                ),
            },
        ]
        variants = ["radial", "tangential", "prism"]
        for test_case, variant in zip(cam_kwargs, variants):
            cameras = FishEyeCameras(
                device=device,
                R=R,
                T=T,
                use_tangential=True,
                use_radial=True,
                use_thin_prism=True,
                world_coordinates=True,
                **test_case,
            )

            # Init shader settings
            materials = Materials(device=device)
            lights = PointLights(device=device)
            lights.location = torch.tensor([0.0, 0.0, +2.0], device=device)[None]

            raster_settings = RasterizationSettings(
                image_size=512, blur_radius=0.0, faces_per_pixel=1
            )
            blend_params = BlendParams(0.5, 1e-4, (0, 0, 0))

            # Test several shaders
            rasterizer_tests = [
                RasterizerTest(
                    MeshRasterizer, HardPhongShader, "hard_phong", "hard_phong"
                ),
                RasterizerTest(
                    MeshRasterizer, HardGouraudShader, "hard_gouraud", "hard_gouraud"
                ),
                RasterizerTest(
                    MeshRasterizer, HardFlatShader, "hard_flat", "hard_flat"
                ),
            ]
            for test in rasterizer_tests:
                shader = test.shader(
                    lights=lights,
                    cameras=cameras,
                    materials=materials,
                    blend_params=blend_params,
                )
                if test.rasterizer == MeshRasterizer:
                    rasterizer = test.rasterizer(
                        cameras=cameras, raster_settings=raster_settings
                    )

                renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
                images = renderer(sphere_mesh)

                rgb = images[0, ..., :3].squeeze().cpu()
                filename = "simple_sphere_light_%s%s%s%s%s.png" % (
                    test.reference_name,
                    postfix,
                    variant,
                    postfix,
                    FishEyeCameras.__name__,
                )

                image_ref = load_rgb_image("test_%s" % filename, DATA_DIR)
                if DEBUG:
                    debug_filename = "simple_sphere_light_%s%s%s%s%s.png" % (
                        test.debug_name,
                        postfix,
                        variant,
                        postfix,
                        FishEyeCameras.__name__,
                    )
                    filename = "DEBUG_%s" % debug_filename
                    Image.fromarray((rgb.numpy() * 255).astype(np.uint8)).save(
                        DATA_DIR / filename
                    )
                self.assertClose(rgb, image_ref, atol=0.05)

            ########################################################
            # Move the light to the +z axis in world space so it is
            # behind the sphere. Note that +Z is in, +Y up,
            # +X left for both world and camera space.
            ########################################################
            lights.location[..., 2] = -2.0
            phong_shader = HardPhongShader(
                lights=lights,
                cameras=cameras,
                materials=materials,
                blend_params=blend_params,
            )

            phong_renderer = MeshRenderer(rasterizer=rasterizer, shader=phong_shader)
            images = phong_renderer(sphere_mesh, lights=lights)
            rgb = images[0, ..., :3].squeeze().cpu()
            if DEBUG:
                filename = "DEBUG_simple_sphere_dark%s%s%s%s.png" % (
                    postfix,
                    variant,
                    postfix,
                    FishEyeCameras.__name__,
                )
                Image.fromarray((rgb.numpy() * 255).astype(np.uint8)).save(
                    DATA_DIR / filename
                )

            image_ref_phong_dark = load_rgb_image(
                "test_simple_sphere_dark%s%s%s%s.png"
                % (postfix, variant, postfix, FishEyeCameras.__name__),
                DATA_DIR,
            )
            # Soft shaders (SplatterPhong) will have a different boundary than hard
            # ones, but should be identical otherwise.
            self.assertLess((rgb - image_ref_phong_dark).quantile(0.99), 0.005)

    def test_fisheye_cow_mesh(self):
        """
        Test FishEye Camera distortions on real meshes
        """
        device = torch.device("cuda:0")
        obj_filename = os.path.join(DATA_DIR, "missing_usemtl/cow.obj")
        mesh = load_objs_as_meshes([obj_filename], device=device)
        R, T = look_at_view_transform(2.7, 0, 180)
        radial_params = torch.tensor([[-1.0, 1.0, 1.0, 0.0, 0.0, -1.0]])
        tangential_params = torch.tensor([[0.5, 0.5]])
        thin_prism_params = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        combinations = product([False, True], repeat=3)
        for combination in combinations:
            cameras = FishEyeCameras(
                device=device,
                R=R,
                T=T,
                world_coordinates=True,
                use_radial=combination[0],
                use_tangential=combination[1],
                use_thin_prism=combination[2],
                radial_params=radial_params,
                tangential_params=tangential_params,
                thin_prism_params=thin_prism_params,
            )
            raster_settings = RasterizationSettings(
                image_size=512,
                blur_radius=0.0,
                faces_per_pixel=1,
            )
            lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras, raster_settings=raster_settings
                ),
                shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
            )
            images = renderer(mesh)
            rgb = images[0, ..., :3].squeeze().cpu()
            filename = "test_cow_mesh_%s_radial_%s_tangential_%s_prism_%s.png" % (
                FishEyeCameras.__name__,
                combination[0],
                combination[1],
                combination[2],
            )
            image_ref = load_rgb_image(filename, DATA_DIR)
            if DEBUG:
                filename = filename.replace("test", "DEBUG")
                Image.fromarray((rgb.numpy() * 255).astype(np.uint8)).save(
                    DATA_DIR / filename
                )
            self.assertClose(rgb, image_ref, atol=0.05)
