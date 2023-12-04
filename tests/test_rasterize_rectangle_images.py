# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
import unittest
from itertools import product

import numpy as np
import torch
from PIL import Image
from pytorch3d.io import load_obj
from pytorch3d.renderer import (
    BlendParams,
    FoVPerspectiveCameras,
    look_at_view_transform,
    Materials,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    SplatterPhongShader,
    TexturesUV,
)
from pytorch3d.renderer.mesh.rasterize_meshes import (
    rasterize_meshes,
    rasterize_meshes_python,
)
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.opengl import MeshRasterizerOpenGL
from pytorch3d.renderer.points import (
    AlphaCompositor,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
)
from pytorch3d.renderer.points.rasterize_points import (
    rasterize_points,
    rasterize_points_python,
)
from pytorch3d.renderer.points.rasterizer import PointFragments
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.transforms.transform3d import Transform3d
from pytorch3d.utils import torus

from .common_testing import (
    get_pytorch3d_dir,
    get_tests_dir,
    load_rgb_image,
    TestCaseMixin,
)


DEBUG = False
DATA_DIR = get_tests_dir() / "data"

# Verts/Faces for a simple mesh with two faces.
verts0 = torch.tensor(
    [
        [-0.7, -0.70, 1.0],
        [0.0, -0.1, 1.0],
        [0.7, -0.7, 1.0],
        [-0.7, 0.1, 1.0],
        [0.0, 0.7, 1.0],
        [0.7, 0.1, 1.0],
    ],
    dtype=torch.float32,
)
faces0 = torch.tensor([[1, 0, 2], [4, 3, 5]], dtype=torch.int64)

# Points for a simple point cloud. Get the vertices from a
# torus and apply rotations such that the points are no longer
# symmerical in X/Y.
torus_mesh = torus(r=0.25, R=1.0, sides=5, rings=2 * 5)
t = (
    Transform3d()
    .rotate_axis_angle(angle=90, axis="Y")
    .rotate_axis_angle(angle=45, axis="Z")
    .scale(0.3)
)
torus_points = t.transform_points(torus_mesh.verts_padded()).squeeze()


def _save_debug_image(idx, image_size, bin_size, blur):
    """
    Save a mask image from the rasterization output for debugging.
    """
    H, W = image_size
    # Save out the last image for debugging
    rgb = (idx[-1, ..., :3].cpu() > -1).squeeze()
    suffix = "square" if H == W else "non_square"
    filename = "%s_bin_size_%s_blur_%.3f_%dx%d.png"
    filename = filename % (suffix, str(bin_size), blur, H, W)
    if DEBUG:
        filename = "DEBUG_%s" % filename
        Image.fromarray((rgb.numpy() * 255).astype(np.uint8)).save(DATA_DIR / filename)


class TestRasterizeRectangleImagesErrors(TestCaseMixin, unittest.TestCase):
    def test_mesh_image_size_arg(self):
        meshes = Meshes(verts=[verts0], faces=[faces0])

        with self.assertRaisesRegex(ValueError, re.escape("tuple/list of (H, W)")):
            rasterize_meshes(
                meshes,
                (100, 200, 3),
                0.0001,
                faces_per_pixel=1,
            )

        with self.assertRaisesRegex(ValueError, "sizes must be greater than 0"):
            rasterize_meshes(
                meshes,
                (0, 10),
                0.0001,
                faces_per_pixel=1,
            )

        with self.assertRaisesRegex(ValueError, "sizes must be integers"):
            rasterize_meshes(
                meshes,
                (100.5, 120.5),
                0.0001,
                faces_per_pixel=1,
            )

    def test_points_image_size_arg(self):
        points = Pointclouds([verts0])

        with self.assertRaisesRegex(ValueError, re.escape("tuple/list of (H, W)")):
            rasterize_points(
                points,
                (100, 200, 3),
                0.0001,
                points_per_pixel=1,
            )

        with self.assertRaisesRegex(ValueError, "sizes must be greater than 0"):
            rasterize_points(
                points,
                (0, 10),
                0.0001,
                points_per_pixel=1,
            )

        with self.assertRaisesRegex(ValueError, "sizes must be integers"):
            rasterize_points(
                points,
                (100.5, 120.5),
                0.0001,
                points_per_pixel=1,
            )


class TestRasterizeRectangleImagesMeshes(TestCaseMixin, unittest.TestCase):
    @staticmethod
    def _clone_mesh(verts0, faces0, device, batch_size):
        """
        Helper function to detach and clone the verts/faces.
        This is needed in order to set up the tensors for
        gradient computation in different tests.
        """
        verts = verts0.detach().clone()
        verts.requires_grad = True
        meshes = Meshes(verts=[verts], faces=[faces0])
        meshes = meshes.to(device).extend(batch_size)
        return verts, meshes

    def _rasterize(self, meshes, image_size, bin_size, blur):
        """
        Simple wrapper around the rasterize function to return
        the fragment data.
        """
        face_idxs, zbuf, bary_coords, pix_dists = rasterize_meshes(
            meshes,
            image_size,
            blur,
            faces_per_pixel=1,
            bin_size=bin_size,
        )
        return Fragments(
            pix_to_face=face_idxs,
            zbuf=zbuf,
            bary_coords=bary_coords,
            dists=pix_dists,
        )

    @staticmethod
    def _save_debug_image(fragments, image_size, bin_size, blur):
        """
        Save a mask image from the rasterization output for debugging.
        """
        H, W = image_size
        # Save out the last image for debugging
        rgb = (fragments.pix_to_face[-1, ..., :3].cpu() > -1).squeeze()
        suffix = "square" if H == W else "non_square"
        filename = "triangle_%s_bin_size_%s_blur_%.3f_%dx%d.png"
        filename = filename % (suffix, str(bin_size), blur, H, W)
        if DEBUG:
            filename = "DEBUG_%s" % filename
            Image.fromarray((rgb.numpy() * 255).astype(np.uint8)).save(
                DATA_DIR / filename
            )

    def _check_fragments(self, frag_1, frag_2):
        """
        Helper function to check that the tensors in
        the Fragments frag_1 and frag_2 are the same.
        """
        self.assertClose(frag_1.pix_to_face, frag_2.pix_to_face)
        self.assertClose(frag_1.dists, frag_2.dists)
        self.assertClose(frag_1.bary_coords, frag_2.bary_coords)
        self.assertClose(frag_1.zbuf, frag_2.zbuf)

    def _compare_square_with_nonsq(
        self,
        image_size,
        blur,
        device,
        verts0,
        faces0,
        nonsq_fragment_gradtensor_list,
        batch_size=1,
    ):
        """
        Calculate the output from rasterizing a square image with the minimum of (H, W).
        Then compare this with the same square region in the non square image.
        The input mesh faces given by faces0 and verts0 are contained within the
        [-1, 1] range of the image so all the relevant pixels will be within the square region.

        `nonsq_fragment_gradtensor_list` is a list of fragments and verts grad tensors
        from rasterizing non square images.
        """
        # Rasterize the square version of the image
        H, W = image_size
        S = min(H, W)
        verts_square, meshes_sq = self._clone_mesh(verts0, faces0, device, batch_size)
        square_fragments = self._rasterize(
            meshes_sq, image_size=(S, S), bin_size=0, blur=blur
        )
        # Save debug image
        _save_debug_image(square_fragments.pix_to_face, (S, S), 0, blur)

        # Extract the values in the square image which are non zero.
        square_mask = square_fragments.pix_to_face > -1
        square_dists = square_fragments.dists[square_mask]
        square_zbuf = square_fragments.zbuf[square_mask]
        square_bary = square_fragments.bary_coords[square_mask]

        # Retain gradients on the output of fragments to check
        # intermediate values with the non square outputs.
        square_fragments.dists.retain_grad()
        square_fragments.bary_coords.retain_grad()
        square_fragments.zbuf.retain_grad()

        # Calculate gradient for the square image
        torch.manual_seed(231)
        grad_zbuf = torch.randn_like(square_zbuf)
        grad_dist = torch.randn_like(square_dists)
        grad_bary = torch.randn_like(square_bary)
        loss0 = (
            (grad_dist * square_dists).sum()
            + (grad_zbuf * square_zbuf).sum()
            + (grad_bary * square_bary).sum()
        )
        loss0.backward()

        # Now compare against the non square outputs provided
        # in the nonsq_fragment_gradtensor_list list
        for fragments, grad_tensor, _name in nonsq_fragment_gradtensor_list:
            # Check that there are the same number of non zero pixels
            # in both the square and non square images.
            non_square_mask = fragments.pix_to_face > -1
            self.assertEqual(non_square_mask.sum().item(), square_mask.sum().item())

            # Check dists, zbuf and bary match the square image
            non_square_dists = fragments.dists[non_square_mask]
            non_square_zbuf = fragments.zbuf[non_square_mask]
            non_square_bary = fragments.bary_coords[non_square_mask]
            self.assertClose(square_dists, non_square_dists)
            self.assertClose(square_zbuf, non_square_zbuf)
            self.assertClose(
                square_bary,
                non_square_bary,
                atol=2e-7,
            )

            # Retain gradients to compare values with outputs from
            # square image
            fragments.dists.retain_grad()
            fragments.bary_coords.retain_grad()
            fragments.zbuf.retain_grad()
            loss1 = (
                (grad_dist * non_square_dists).sum()
                + (grad_zbuf * non_square_zbuf).sum()
                + (grad_bary * non_square_bary).sum()
            )
            loss1.sum().backward()

            # Get the non zero values in the intermediate gradients
            # and compare with the values from the square image
            non_square_grad_dists = fragments.dists.grad[non_square_mask]
            non_square_grad_bary = fragments.bary_coords.grad[non_square_mask]
            non_square_grad_zbuf = fragments.zbuf.grad[non_square_mask]

            self.assertClose(
                non_square_grad_dists,
                square_fragments.dists.grad[square_mask],
            )
            self.assertClose(
                non_square_grad_bary,
                square_fragments.bary_coords.grad[square_mask],
            )
            self.assertClose(
                non_square_grad_zbuf,
                square_fragments.zbuf.grad[square_mask],
            )

            # Finally check the gradients of the input vertices for
            # the square and non square case
            self.assertClose(verts_square.grad, grad_tensor.grad, rtol=3e-4, atol=5e-3)

    def test_gpu(self):
        """
        Test that the output of rendering non square images
        gives the same result as square images. i.e. the
        dists, zbuf, bary are all the same for the square
        region which is present in both images.
        """
        # Test both cases: (W > H), (H > W) as well as the case where
        # H and W are not integer multiples of each other (i.e. float aspect ratio)
        image_sizes = [(64, 128), (128, 64), (128, 256), (256, 128), (600, 1110)]

        devices = ["cuda:0"]
        blurs = [0.0, 0.001]
        batch_sizes = [1, 4]
        test_cases = product(image_sizes, blurs, devices, batch_sizes)

        for image_size, blur, device, batch_size in test_cases:
            # Initialize the verts grad tensor and the meshes objects
            verts_nonsq_naive, meshes_nonsq_naive = self._clone_mesh(
                verts0, faces0, device, batch_size
            )
            verts_nonsq_binned, meshes_nonsq_binned = self._clone_mesh(
                verts0, faces0, device, batch_size
            )

            # Get the outputs for both naive and coarse to fine rasterization
            fragments_naive = self._rasterize(
                meshes_nonsq_naive,
                image_size,
                blur=blur,
                bin_size=0,
            )
            fragments_binned = self._rasterize(
                meshes_nonsq_binned,
                image_size,
                blur=blur,
                bin_size=None,
            )

            # Save out debug images if needed
            _save_debug_image(fragments_naive.pix_to_face, image_size, 0, blur)
            _save_debug_image(fragments_binned.pix_to_face, image_size, None, blur)

            # Check naive and binned fragments give the same outputs
            self._check_fragments(fragments_naive, fragments_binned)

            # Here we want to compare the square image with the naive and the
            # coarse to fine methods outputs
            nonsq_fragment_gradtensor_list = [
                (fragments_naive, verts_nonsq_naive, "naive"),
                (fragments_binned, verts_nonsq_binned, "coarse-to-fine"),
            ]

            self._compare_square_with_nonsq(
                image_size,
                blur,
                device,
                verts0,
                faces0,
                nonsq_fragment_gradtensor_list,
                batch_size,
            )

    def test_cpu(self):
        """
        Test that the output of rendering non square images
        gives the same result as square images. i.e. the
        dists, zbuf, bary are all the same for the square
        region which is present in both images.

        In this test we compare between the naive C++ implementation
        and the naive python implementation as the Coarse/Fine
        method is not fully implemented in C++
        """
        # Test both when (W > H) and (H > W).
        # Using smaller image sizes here as the Python rasterizer is really slow.
        image_sizes = [(32, 64), (64, 32), (60, 110)]
        devices = ["cpu"]
        blurs = [0.0, 0.001]
        batch_sizes = [1]
        test_cases = product(image_sizes, blurs, devices, batch_sizes)

        for image_size, blur, device, batch_size in test_cases:
            # Initialize the verts grad tensor and the meshes objects
            verts_nonsq_naive, meshes_nonsq_naive = self._clone_mesh(
                verts0, faces0, device, batch_size
            )
            verts_nonsq_python, meshes_nonsq_python = self._clone_mesh(
                verts0, faces0, device, batch_size
            )

            # Compare Naive CPU with Python as Coarse/Fine rasteriztation
            # is not implemented for CPU
            fragments_naive = self._rasterize(
                meshes_nonsq_naive, image_size, bin_size=0, blur=blur
            )
            face_idxs, zbuf, bary_coords, pix_dists = rasterize_meshes_python(
                meshes_nonsq_python,
                image_size,
                blur,
                faces_per_pixel=1,
            )
            fragments_python = Fragments(
                pix_to_face=face_idxs,
                zbuf=zbuf,
                bary_coords=bary_coords,
                dists=pix_dists,
            )

            # Save debug images if DEBUG is set to true at the top of the file.
            _save_debug_image(fragments_naive.pix_to_face, image_size, 0, blur)
            _save_debug_image(fragments_python.pix_to_face, image_size, "python", blur)

            # List of non square outputs to compare with the square output
            nonsq_fragment_gradtensor_list = [
                (fragments_naive, verts_nonsq_naive, "naive"),
                (fragments_python, verts_nonsq_python, "python"),
            ]
            self._compare_square_with_nonsq(
                image_size,
                blur,
                device,
                verts0,
                faces0,
                nonsq_fragment_gradtensor_list,
                batch_size,
            )

    def test_render_cow(self):
        self._render_cow(MeshRasterizer)

    def test_render_cow_opengl(self):
        self._render_cow(MeshRasterizerOpenGL)

    def _render_cow(self, rasterizer_type):
        """
        Test a larger textured mesh is rendered correctly in a non square image.
        """
        device = torch.device("cuda:0")
        obj_dir = get_pytorch3d_dir() / "docs/tutorials/data"
        obj_filename = obj_dir / "cow_mesh/cow.obj"

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
        R, T = look_at_view_transform(1.2, 0, 90)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

        raster_settings = RasterizationSettings(
            image_size=(500, 800), blur_radius=0.0, faces_per_pixel=1
        )

        # Init shader settings
        materials = Materials(device=device)
        lights = PointLights(device=device)
        lights.location = torch.tensor([0.0, 0.0, -2.0], device=device)[None]

        # Init renderer
        rasterizer = rasterizer_type(cameras=cameras, raster_settings=raster_settings)
        if rasterizer_type == MeshRasterizer:
            blend_params = BlendParams(
                sigma=1e-1,
                gamma=1e-4,
                background_color=torch.tensor([1.0, 1.0, 1.0], device=device),
            )
            shader = SoftPhongShader(
                lights=lights,
                cameras=cameras,
                materials=materials,
                blend_params=blend_params,
            )
        else:
            blend_params = BlendParams(
                sigma=0.5,
                background_color=torch.tensor([1.0, 1.0, 1.0], device=device),
            )
            shader = SplatterPhongShader(
                lights=lights,
                cameras=cameras,
                materials=materials,
                blend_params=blend_params,
            )

        renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

        # Load reference image
        image_ref = load_rgb_image(
            f"test_cow_image_rectangle_{rasterizer_type.__name__}.png", DATA_DIR
        )

        for bin_size in [0, None]:
            if bin_size == 0 and rasterizer_type == MeshRasterizerOpenGL:
                continue

            # Check both naive and coarse to fine produce the same output.
            renderer.rasterizer.raster_settings.bin_size = bin_size
            images = renderer(mesh)
            rgb = images[0, ..., :3].squeeze().cpu()

            if DEBUG:
                Image.fromarray((rgb.numpy() * 255).astype(np.uint8)).save(
                    DATA_DIR
                    / f"DEBUG_cow_image_rectangle_{rasterizer_type.__name__}.png"
                )

            # NOTE some pixels can be flaky
            cond1 = torch.allclose(rgb, image_ref, atol=0.05)
            self.assertTrue(cond1)


class TestRasterizeRectangleImagesPointclouds(TestCaseMixin, unittest.TestCase):
    @staticmethod
    def _clone_pointcloud(verts0, device, batch_size):
        """
        Helper function to detach and clone the verts.
        This is needed in order to set up the tensors for
        gradient computation in different tests.
        """
        verts = verts0.detach().clone()
        verts.requires_grad = True
        pointclouds = Pointclouds(points=[verts])
        pointclouds = pointclouds.to(device).extend(batch_size)
        return verts, pointclouds

    def _rasterize(self, meshes, image_size, bin_size, blur):
        """
        Simple wrapper around the rasterize function to return
        the fragment data.
        """
        idxs, zbuf, dists = rasterize_points(
            meshes,
            image_size,
            blur,
            points_per_pixel=1,
            bin_size=bin_size,
        )
        return PointFragments(
            idx=idxs,
            zbuf=zbuf,
            dists=dists,
        )

    def _check_fragments(self, frag_1, frag_2):
        """
        Helper function to check that the tensors in
        the Fragments frag_1 and frag_2 are the same.
        """
        self.assertClose(frag_1.idx, frag_2.idx)
        self.assertClose(frag_1.dists, frag_2.dists)
        self.assertClose(frag_1.zbuf, frag_2.zbuf)

    def _compare_square_with_nonsq(
        self,
        image_size,
        blur,
        device,
        points,
        nonsq_fragment_gradtensor_list,
        batch_size=1,
    ):
        """
        Calculate the output from rasterizing a square image with the minimum of (H, W).
        Then compare this with the same square region in the non square image.
        The input points are contained within the [-1, 1] range of the image
        so all the relevant pixels will be within the square region.

        `nonsq_fragment_gradtensor_list` is a list of fragments and verts grad tensors
        from rasterizing non square images.
        """
        # Rasterize the square version of the image
        H, W = image_size
        S = min(H, W)
        points_square, pointclouds_sq = self._clone_pointcloud(
            points, device, batch_size
        )
        square_fragments = self._rasterize(
            pointclouds_sq, image_size=(S, S), bin_size=0, blur=blur
        )
        # Save debug image
        _save_debug_image(square_fragments.idx, (S, S), 0, blur)

        # Extract the values in the square image which are non zero.
        square_mask = square_fragments.idx > -1
        square_dists = square_fragments.dists[square_mask]
        square_zbuf = square_fragments.zbuf[square_mask]

        # Retain gradients on the output of fragments to check
        # intermediate values with the non square outputs.
        square_fragments.dists.retain_grad()
        square_fragments.zbuf.retain_grad()

        # Calculate gradient for the square image
        torch.manual_seed(231)
        grad_zbuf = torch.randn_like(square_zbuf)
        grad_dist = torch.randn_like(square_dists)
        loss0 = (grad_dist * square_dists).sum() + (grad_zbuf * square_zbuf).sum()
        loss0.backward()

        # Now compare against the non square outputs provided
        # in the nonsq_fragment_gradtensor_list list
        for fragments, grad_tensor, _name in nonsq_fragment_gradtensor_list:
            # Check that there are the same number of non zero pixels
            # in both the square and non square images.
            non_square_mask = fragments.idx > -1
            self.assertEqual(non_square_mask.sum().item(), square_mask.sum().item())

            # Check dists, zbuf and bary match the square image
            non_square_dists = fragments.dists[non_square_mask]
            non_square_zbuf = fragments.zbuf[non_square_mask]
            self.assertClose(square_dists, non_square_dists)
            self.assertClose(square_zbuf, non_square_zbuf)

            # Retain gradients to compare values with outputs from
            # square image
            fragments.dists.retain_grad()
            fragments.zbuf.retain_grad()
            loss1 = (grad_dist * non_square_dists).sum() + (
                grad_zbuf * non_square_zbuf
            ).sum()
            loss1.sum().backward()

            # Get the non zero values in the intermediate gradients
            # and compare with the values from the square image
            non_square_grad_dists = fragments.dists.grad[non_square_mask]
            non_square_grad_zbuf = fragments.zbuf.grad[non_square_mask]

            self.assertClose(
                non_square_grad_dists,
                square_fragments.dists.grad[square_mask],
            )
            self.assertClose(
                non_square_grad_zbuf,
                square_fragments.zbuf.grad[square_mask],
            )

            # Finally check the gradients of the input vertices for
            # the square and non square case
            self.assertClose(points_square.grad, grad_tensor.grad, rtol=2e-4)

    def test_gpu(self):
        """
        Test that the output of rendering non square images
        gives the same result as square images. i.e. the
        dists, zbuf, idx are all the same for the square
        region which is present in both images.
        """
        # Test both cases: (W > H), (H > W) as well as the case where
        # H and W are not integer multiples of each other (i.e. float aspect ratio)
        image_sizes = [(64, 128), (128, 64), (128, 256), (256, 128), (600, 1110)]

        devices = ["cuda:0"]
        blurs = [5e-2]
        batch_sizes = [1, 4]
        test_cases = product(image_sizes, blurs, devices, batch_sizes)

        for image_size, blur, device, batch_size in test_cases:
            # Initialize the verts grad tensor and the meshes objects
            verts_nonsq_naive, pointcloud_nonsq_naive = self._clone_pointcloud(
                torus_points, device, batch_size
            )
            verts_nonsq_binned, pointcloud_nonsq_binned = self._clone_pointcloud(
                torus_points, device, batch_size
            )

            # Get the outputs for both naive and coarse to fine rasterization
            fragments_naive = self._rasterize(
                pointcloud_nonsq_naive,
                image_size,
                blur=blur,
                bin_size=0,
            )
            fragments_binned = self._rasterize(
                pointcloud_nonsq_binned,
                image_size,
                blur=blur,
                bin_size=None,
            )

            # Save out debug images if needed
            _save_debug_image(fragments_naive.idx, image_size, 0, blur)
            _save_debug_image(fragments_binned.idx, image_size, None, blur)

            # Check naive and binned fragments give the same outputs
            self._check_fragments(fragments_naive, fragments_binned)

            # Here we want to compare the square image with the naive and the
            # coarse to fine methods outputs
            nonsq_fragment_gradtensor_list = [
                (fragments_naive, verts_nonsq_naive, "naive"),
                (fragments_binned, verts_nonsq_binned, "coarse-to-fine"),
            ]

            self._compare_square_with_nonsq(
                image_size,
                blur,
                device,
                torus_points,
                nonsq_fragment_gradtensor_list,
                batch_size,
            )

    def test_cpu(self):
        """
        Test that the output of rendering non square images
        gives the same result as square images. i.e. the
        dists, zbuf, idx are all the same for the square
        region which is present in both images.

        In this test we compare between the naive C++ implementation
        and the naive python implementation as the Coarse/Fine
        method is not fully implemented in C++
        """
        # Test both when (W > H) and (H > W).
        # Using smaller image sizes here as the Python rasterizer is really slow.
        image_sizes = [(32, 64), (64, 32), (60, 110)]
        devices = ["cpu"]
        blurs = [5e-2]
        batch_sizes = [1]
        test_cases = product(image_sizes, blurs, devices, batch_sizes)

        for image_size, blur, device, batch_size in test_cases:
            # Initialize the verts grad tensor and the meshes objects
            verts_nonsq_naive, pointcloud_nonsq_naive = self._clone_pointcloud(
                torus_points, device, batch_size
            )
            verts_nonsq_python, pointcloud_nonsq_python = self._clone_pointcloud(
                torus_points, device, batch_size
            )

            # Compare Naive CPU with Python as Coarse/Fine rasteriztation
            # is not implemented for CPU
            fragments_naive = self._rasterize(
                pointcloud_nonsq_naive, image_size, bin_size=0, blur=blur
            )
            idxs, zbuf, pix_dists = rasterize_points_python(
                pointcloud_nonsq_python,
                image_size,
                blur,
                points_per_pixel=1,
            )
            fragments_python = PointFragments(
                idx=idxs,
                zbuf=zbuf,
                dists=pix_dists,
            )

            # Save debug images if DEBUG is set to true at the top of the file.
            _save_debug_image(fragments_naive.idx, image_size, 0, blur)
            _save_debug_image(fragments_python.idx, image_size, "python", blur)

            # List of non square outputs to compare with the square output
            nonsq_fragment_gradtensor_list = [
                (fragments_naive, verts_nonsq_naive, "naive"),
                (fragments_python, verts_nonsq_python, "python"),
            ]
            self._compare_square_with_nonsq(
                image_size,
                blur,
                device,
                torus_points,
                nonsq_fragment_gradtensor_list,
                batch_size,
            )

    def test_render_pointcloud(self):
        """
        Test a textured point cloud is rendered correctly in a non square image.
        """
        device = torch.device("cuda:0")
        pointclouds = Pointclouds(
            points=[torus_points * 2.0],
            features=torch.ones_like(torus_points[None, ...]),
        ).to(device)
        R, T = look_at_view_transform(2.7, 0.0, 0.0)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        raster_settings = PointsRasterizationSettings(
            image_size=(512, 1024), radius=5e-2, points_per_pixel=1
        )
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        compositor = AlphaCompositor()
        renderer = PointsRenderer(rasterizer=rasterizer, compositor=compositor)

        # Load reference image
        image_ref = load_rgb_image("test_pointcloud_rectangle_image.png", DATA_DIR)

        for bin_size in [0, None]:
            # Check both naive and coarse to fine produce the same output.
            renderer.rasterizer.raster_settings.bin_size = bin_size
            images = renderer(pointclouds)
            rgb = images[0, ..., :3].squeeze().cpu()

            if DEBUG:
                Image.fromarray((rgb.numpy() * 255).astype(np.uint8)).save(
                    DATA_DIR / "DEBUG_pointcloud_rectangle_image.png"
                )

            # NOTE some pixels can be flaky
            cond1 = torch.allclose(rgb, image_ref, atol=0.05)
            self.assertTrue(cond1)
