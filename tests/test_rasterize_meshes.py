# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import functools
import unittest

import torch
from common_testing import TestCaseMixin, get_random_cuda_device
from pytorch3d import _C
from pytorch3d.renderer.mesh.rasterize_meshes import (
    rasterize_meshes,
    rasterize_meshes_python,
)
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere


class TestRasterizeMeshes(TestCaseMixin, unittest.TestCase):
    def test_simple_python(self):
        device = torch.device("cpu")
        self._simple_triangle_raster(rasterize_meshes_python, device, bin_size=-1)
        self._simple_blurry_raster(rasterize_meshes_python, device, bin_size=-1)
        self._test_behind_camera(rasterize_meshes_python, device, bin_size=-1)
        self._test_perspective_correct(rasterize_meshes_python, device, bin_size=-1)
        self._test_back_face_culling(rasterize_meshes_python, device, bin_size=-1)

    def test_simple_cpu_naive(self):
        device = torch.device("cpu")
        self._simple_triangle_raster(rasterize_meshes, device, bin_size=0)
        self._simple_blurry_raster(rasterize_meshes, device, bin_size=0)
        self._test_behind_camera(rasterize_meshes, device, bin_size=0)
        self._test_perspective_correct(rasterize_meshes, device, bin_size=0)
        self._test_back_face_culling(rasterize_meshes, device, bin_size=0)

    def test_simple_cuda_naive(self):
        device = get_random_cuda_device()
        self._simple_triangle_raster(rasterize_meshes, device, bin_size=0)
        self._simple_blurry_raster(rasterize_meshes, device, bin_size=0)
        self._test_behind_camera(rasterize_meshes, device, bin_size=0)
        self._test_perspective_correct(rasterize_meshes, device, bin_size=0)
        self._test_back_face_culling(rasterize_meshes, device, bin_size=0)

    def test_simple_cuda_binned(self):
        device = get_random_cuda_device()
        self._simple_triangle_raster(rasterize_meshes, device, bin_size=5)
        self._simple_blurry_raster(rasterize_meshes, device, bin_size=5)
        self._test_behind_camera(rasterize_meshes, device, bin_size=5)
        self._test_perspective_correct(rasterize_meshes, device, bin_size=5)
        self._test_back_face_culling(rasterize_meshes, device, bin_size=5)

    def test_python_vs_cpu_vs_cuda(self):
        torch.manual_seed(231)
        device = torch.device("cpu")
        image_size = 32
        blur_radius = 0.1 ** 2
        faces_per_pixel = 3

        for d in ["cpu", get_random_cuda_device()]:
            device = torch.device(d)
            compare_grads = True
            # Mesh with a single face.
            verts1 = torch.tensor(
                [[0.0, 0.6, 0.1], [-0.7, -0.4, 0.5], [0.7, -0.4, 0.7]],
                dtype=torch.float32,
                requires_grad=True,
                device=device,
            )
            faces1 = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)
            meshes1 = Meshes(verts=[verts1], faces=[faces1])
            args1 = (meshes1, image_size, blur_radius, faces_per_pixel)
            verts2 = verts1.detach().clone()
            verts2.requires_grad = True
            meshes2 = Meshes(verts=[verts2], faces=[faces1])
            args2 = (meshes2, image_size, blur_radius, faces_per_pixel)
            self._compare_impls(
                rasterize_meshes_python,
                rasterize_meshes,
                args1,
                args2,
                verts1,
                verts2,
                compare_grads=compare_grads,
            )

            # Mesh with multiple faces.
            # fmt: off
            verts1 = torch.tensor(
                [
                    [ -0.5, 0.0,  0.1],  # noqa: E241, E201
                    [  0.0, 0.6,  0.5],  # noqa: E241, E201
                    [  0.5, 0.0,  0.7],  # noqa: E241, E201
                    [-0.25, 0.0,  0.9],  # noqa: E241, E201
                    [ 0.26, 0.5,  0.8],  # noqa: E241, E201
                    [ 0.76, 0.0,  0.8],  # noqa: E241, E201
                    [-0.41, 0.0,  0.5],  # noqa: E241, E201
                    [ 0.61, 0.6,  0.6],  # noqa: E241, E201
                    [ 0.41, 0.0,  0.5],  # noqa: E241, E201
                    [ -0.2, 0.0, -0.5],  # noqa: E241, E201
                    [  0.3, 0.6, -0.5],  # noqa: E241, E201
                    [  0.4, 0.0, -0.5],  # noqa: E241, E201
                ],
                dtype=torch.float32,
                device=device,
                requires_grad=True
            )
            faces1 = torch.tensor(
                [
                    [ 1, 0,  2],  # noqa: E241, E201
                    [ 4, 3,  5],  # noqa: E241, E201
                    [ 7, 6,  8],  # noqa: E241, E201
                    [10, 9, 11]   # noqa: E241, E201
                ],
                dtype=torch.int64,
                device=device,
            )
            # fmt: on
            meshes = Meshes(verts=[verts1], faces=[faces1])
            args1 = (meshes, image_size, blur_radius, faces_per_pixel)
            verts2 = verts1.clone().detach()
            verts2.requires_grad = True
            meshes2 = Meshes(verts=[verts2], faces=[faces1])
            args2 = (meshes2, image_size, blur_radius, faces_per_pixel)
            self._compare_impls(
                rasterize_meshes_python,
                rasterize_meshes,
                args1,
                args2,
                verts1,
                verts2,
                compare_grads=compare_grads,
            )

            # Icosphere
            meshes = ico_sphere(device=device)
            verts1, faces1 = meshes.get_mesh_verts_faces(0)
            verts1.requires_grad = True
            meshes = Meshes(verts=[verts1], faces=[faces1])
            args1 = (meshes, image_size, blur_radius, faces_per_pixel)
            verts2 = verts1.detach().clone()
            verts2.requires_grad = True
            meshes2 = Meshes(verts=[verts2], faces=[faces1])
            args2 = (meshes2, image_size, blur_radius, faces_per_pixel)
            self._compare_impls(
                rasterize_meshes_python,
                rasterize_meshes,
                args1,
                args2,
                verts1,
                verts2,
                compare_grads=compare_grads,
            )

    def test_cpu_vs_cuda_naive(self):
        """
        Compare naive versions of cuda and cpp
        """

        torch.manual_seed(231)
        image_size = 64
        radius = 0.1 ** 2
        faces_per_pixel = 3
        device = torch.device("cpu")
        meshes_cpu = ico_sphere(0, device)
        verts1, faces1 = meshes_cpu.get_mesh_verts_faces(0)
        verts1.requires_grad = True
        meshes_cpu = Meshes(verts=[verts1], faces=[faces1])

        device = get_random_cuda_device()
        meshes_cuda = ico_sphere(0, device)
        verts2, faces2 = meshes_cuda.get_mesh_verts_faces(0)
        verts2.requires_grad = True
        meshes_cuda = Meshes(verts=[verts2], faces=[faces2])

        args_cpu = (meshes_cpu, image_size, radius, faces_per_pixel)
        args_cuda = (meshes_cuda, image_size, radius, faces_per_pixel, 0, 0)
        self._compare_impls(
            rasterize_meshes,
            rasterize_meshes,
            args_cpu,
            args_cuda,
            verts1,
            verts2,
            compare_grads=True,
        )

    def test_coarse_cpu(self):
        return self._test_coarse_rasterize(torch.device("cpu"))

    def test_coarse_cuda(self):
        return self._test_coarse_rasterize(get_random_cuda_device())

    def test_cpp_vs_cuda_naive_vs_cuda_binned(self):
        # Make sure that the backward pass runs for all pathways
        image_size = 64  # test is too slow for very large images.
        N = 1
        radius = 0.1 ** 2
        faces_per_pixel = 3

        grad_zbuf = torch.randn(N, image_size, image_size, faces_per_pixel)
        grad_dist = torch.randn(N, image_size, image_size, faces_per_pixel)
        grad_bary = torch.randn(N, image_size, image_size, faces_per_pixel, 3)

        device = torch.device("cpu")
        meshes = ico_sphere(0, device)
        verts, faces = meshes.get_mesh_verts_faces(0)
        verts.requires_grad = True
        meshes = Meshes(verts=[verts], faces=[faces])

        # Option I: CPU, naive
        args = (meshes, image_size, radius, faces_per_pixel)
        idx1, zbuf1, bary1, dist1 = rasterize_meshes(*args)

        loss = (
            (zbuf1 * grad_zbuf).sum()
            + (dist1 * grad_dist).sum()
            + (bary1 * grad_bary).sum()
        )
        loss.backward()
        idx1 = idx1.data.cpu().clone()
        zbuf1 = zbuf1.data.cpu().clone()
        dist1 = dist1.data.cpu().clone()
        grad1 = verts.grad.data.cpu().clone()

        # Option II: CUDA, naive
        device = get_random_cuda_device()
        meshes = ico_sphere(0, device)
        verts, faces = meshes.get_mesh_verts_faces(0)
        verts.requires_grad = True
        meshes = Meshes(verts=[verts], faces=[faces])

        args = (meshes, image_size, radius, faces_per_pixel, 0, 0)
        idx2, zbuf2, bary2, dist2 = rasterize_meshes(*args)
        grad_zbuf = grad_zbuf.to(device)
        grad_dist = grad_dist.to(device)
        grad_bary = grad_bary.to(device)
        loss = (
            (zbuf2 * grad_zbuf).sum()
            + (dist2 * grad_dist).sum()
            + (bary2 * grad_bary).sum()
        )
        loss.backward()
        idx2 = idx2.data.cpu().clone()
        zbuf2 = zbuf2.data.cpu().clone()
        dist2 = dist2.data.cpu().clone()
        grad2 = verts.grad.data.cpu().clone()

        # Option III: CUDA, binned
        meshes = ico_sphere(0, device)
        verts, faces = meshes.get_mesh_verts_faces(0)
        verts.requires_grad = True
        meshes = Meshes(verts=[verts], faces=[faces])

        args = (meshes, image_size, radius, faces_per_pixel, 32, 500)
        idx3, zbuf3, bary3, dist3 = rasterize_meshes(*args)

        loss = (
            (zbuf3 * grad_zbuf).sum()
            + (dist3 * grad_dist).sum()
            + (bary3 * grad_bary).sum()
        )
        loss.backward()
        idx3 = idx3.data.cpu().clone()
        zbuf3 = zbuf3.data.cpu().clone()
        dist3 = dist3.data.cpu().clone()
        grad3 = verts.grad.data.cpu().clone()

        # Make sure everything was the same
        self.assertTrue((idx1 == idx2).all().item())
        self.assertTrue((idx1 == idx3).all().item())
        self.assertClose(zbuf1, zbuf2, atol=1e-6)
        self.assertClose(zbuf1, zbuf3, atol=1e-6)
        self.assertClose(dist1, dist2, atol=1e-6)
        self.assertClose(dist1, dist3, atol=1e-6)

        self.assertClose(grad1, grad2, rtol=5e-3)  # flaky test
        self.assertClose(grad1, grad3, rtol=5e-3)
        self.assertClose(grad2, grad3, rtol=5e-3)

    def test_compare_coarse_cpu_vs_cuda(self):
        torch.manual_seed(231)
        N = 1
        image_size = 512
        blur_radius = 0.0
        bin_size = 32
        max_faces_per_bin = 20

        device = torch.device("cpu")

        meshes = ico_sphere(2, device)
        faces = meshes.faces_packed()
        verts = meshes.verts_packed()
        faces_verts = verts[faces]
        num_faces_per_mesh = meshes.num_faces_per_mesh()
        mesh_to_face_first_idx = meshes.mesh_to_faces_packed_first_idx()

        bin_faces_cpu = _C._rasterize_meshes_coarse(
            faces_verts,
            mesh_to_face_first_idx,
            num_faces_per_mesh,
            image_size,
            blur_radius,
            bin_size,
            max_faces_per_bin,
        )
        device = get_random_cuda_device()
        meshes = meshes.clone().to(device)

        faces = meshes.faces_packed()
        verts = meshes.verts_packed()
        faces_verts = verts[faces]
        num_faces_per_mesh = meshes.num_faces_per_mesh()
        mesh_to_face_first_idx = meshes.mesh_to_faces_packed_first_idx()

        bin_faces_cuda = _C._rasterize_meshes_coarse(
            faces_verts,
            mesh_to_face_first_idx,
            num_faces_per_mesh,
            image_size,
            blur_radius,
            bin_size,
            max_faces_per_bin,
        )

        # Bin faces might not be the same: CUDA version might write them in
        # any order. But if we sort the non-(-1) elements of the CUDA output
        # then they should be the same.

        for n in range(N):
            for by in range(bin_faces_cpu.shape[1]):
                for bx in range(bin_faces_cpu.shape[2]):
                    K = (bin_faces_cuda[n, by, bx] != -1).sum().item()
                    idxs_cpu = bin_faces_cpu[n, by, bx].tolist()
                    idxs_cuda = bin_faces_cuda[n, by, bx].tolist()
                    idxs_cuda[:K] = sorted(idxs_cuda[:K])
                    self.assertEqual(idxs_cpu, idxs_cuda)

    def test_python_vs_cpp_perspective_correct(self):
        torch.manual_seed(232)
        N = 2
        V = 10
        F = 5
        verts1 = torch.randn(N, V, 3, requires_grad=True)
        verts2 = verts1.detach().clone().requires_grad_(True)
        faces = torch.randint(V, size=(N, F, 3))
        meshes1 = Meshes(verts1, faces)
        meshes2 = Meshes(verts2, faces)

        kwargs = {"image_size": 24, "perspective_correct": True}
        fn1 = functools.partial(rasterize_meshes, meshes1, **kwargs)
        fn2 = functools.partial(rasterize_meshes_python, meshes2, **kwargs)
        args = ()
        self._compare_impls(fn1, fn2, args, args, verts1, verts2, compare_grads=True)

    def test_cpp_vs_cuda_perspective_correct(self):
        meshes = ico_sphere(2, device=torch.device("cpu"))
        verts1, faces1 = meshes.get_mesh_verts_faces(0)
        verts1.requires_grad = True
        meshes1 = Meshes(verts=[verts1], faces=[faces1])
        device = get_random_cuda_device()
        verts2 = verts1.detach().to(device).requires_grad_(True)
        faces2 = faces1.detach().clone().to(device)
        meshes2 = Meshes(verts=[verts2], faces=[faces2])

        kwargs = {"image_size": 64, "perspective_correct": True}
        fn1 = functools.partial(rasterize_meshes, meshes1, **kwargs)
        fn2 = functools.partial(rasterize_meshes, meshes2, bin_size=0, **kwargs)
        args = ()
        self._compare_impls(fn1, fn2, args, args, verts1, verts2, compare_grads=True)

    def test_cuda_naive_vs_binned_perspective_correct(self):
        device = get_random_cuda_device()
        meshes = ico_sphere(2, device=device)
        verts1, faces1 = meshes.get_mesh_verts_faces(0)
        verts1.requires_grad = True
        meshes1 = Meshes(verts=[verts1], faces=[faces1])
        verts2 = verts1.detach().clone().requires_grad_(True)
        faces2 = faces1.detach().clone()
        meshes2 = Meshes(verts=[verts2], faces=[faces2])

        kwargs = {"image_size": 64, "perspective_correct": True}
        fn1 = functools.partial(rasterize_meshes, meshes1, bin_size=0, **kwargs)
        fn2 = functools.partial(rasterize_meshes, meshes2, bin_size=8, **kwargs)
        args = ()
        self._compare_impls(fn1, fn2, args, args, verts1, verts2, compare_grads=True)

    def test_bin_size_error(self):
        meshes = ico_sphere(2)
        image_size = 1024
        bin_size = 16
        with self.assertRaisesRegex(ValueError, "bin_size too small"):
            rasterize_meshes(meshes, image_size, 0.0, 2, bin_size)

    def _test_back_face_culling(self, rasterize_meshes_fn, device, bin_size):
        # Square based pyramid mesh.
        # fmt: off
        verts = torch.tensor([
            [-0.5, 0.0,  0.5],  # noqa: E241 E201 Front right
            [ 0.5, 0.0,  0.5],  # noqa: E241 E201 Front left
            [ 0.5, 0.0,  1.5],  # noqa: E241 E201 Back left
            [-0.5, 0.0,  1.5],  # noqa: E241 E201 Back right
            [ 0.0, 1.0,  1.0]   # noqa: E241 E201 Top point of pyramid
        ], dtype=torch.float32, device=device)

        faces = torch.tensor([
            [2, 1, 0],  # noqa: E241 E201 Square base
            [3, 2, 0],  # noqa: E241 E201 Square base
            [1, 0, 4],  # noqa: E241 E201 Triangle on front
            [2, 4, 3],  # noqa: E241 E201 Triangle on back
            [3, 4, 0],  # noqa: E241 E201 Triangle on left side
            [1, 4, 2]   # noqa: E241 E201 Triangle on right side
        ], dtype=torch.int64, device=device)
        # fmt: on
        mesh = Meshes(verts=[verts], faces=[faces])
        kwargs = {
            "meshes": mesh,
            "image_size": 10,
            "faces_per_pixel": 2,
            "blur_radius": 0.0,
            "perspective_correct": False,
            "cull_backfaces": False,
        }
        if bin_size != -1:
            kwargs["bin_size"] = bin_size

        # fmt: off
        pix_to_face_frontface = torch.tensor([
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241 E201
            [-1, -1, -1, -1,  2,  2, -1, -1, -1, -1],  # noqa: E241 E201
            [-1, -1, -1, -1,  2,  2, -1, -1, -1, -1],  # noqa: E241 E201
            [-1, -1, -1,  2,  2,  2,  2, -1, -1, -1],  # noqa: E241 E201
            [-1, -1, -1,  2,  2,  2,  2, -1, -1, -1],  # noqa: E241 E201
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241 E201
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241 E201
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241 E201
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241 E201
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]   # noqa: E241 E201
        ], dtype=torch.int64, device=device)
        pix_to_face_backface = torch.tensor([
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241 E201
            [-1, -1, -1, -1,  3,  3, -1, -1, -1, -1],  # noqa: E241 E201
            [-1, -1, -1, -1,  3,  3, -1, -1, -1, -1],  # noqa: E241 E201
            [-1, -1, -1,  3,  3,  3,  3, -1, -1, -1],  # noqa: E241 E201
            [-1, -1, -1,  3,  3,  3,  3, -1, -1, -1],  # noqa: E241 E201
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241 E201
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241 E201
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241 E201
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241 E201
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]   # noqa: E241 E201
        ], dtype=torch.int64, device=device)
        # fmt: on

        pix_to_face_padded = -(torch.ones_like(pix_to_face_frontface))
        # Run with and without culling
        # Without culling, for k=0, the front face (i.e. face 2) is
        # rasterized and for k=1, the back face (i.e. face 3) is
        # rasterized.
        idx_f, zbuf_f, bary_f, dists_f = rasterize_meshes_fn(**kwargs)
        self.assertTrue(torch.all(idx_f[..., 0].squeeze() == pix_to_face_frontface))
        self.assertTrue(torch.all(idx_f[..., 1].squeeze() == pix_to_face_backface))

        # With culling, for k=0, the front face (i.e. face 2) is
        # rasterized and for k=1, there are no faces rasterized
        kwargs["cull_backfaces"] = True
        idx_t, zbuf_t, bary_t, dists_t = rasterize_meshes_fn(**kwargs)
        self.assertTrue(torch.all(idx_t[..., 0].squeeze() == pix_to_face_frontface))
        self.assertTrue(torch.all(idx_t[..., 1].squeeze() == pix_to_face_padded))

    def _compare_impls(
        self,
        fn1,
        fn2,
        args1,
        args2,
        grad_var1=None,
        grad_var2=None,
        compare_grads=False,
    ):
        idx1, zbuf1, bary1, dist1 = fn1(*args1)
        idx2, zbuf2, bary2, dist2 = fn2(*args2)
        self.assertTrue((idx1.cpu() == idx2.cpu()).all().item())
        self.assertClose(zbuf1.cpu(), zbuf2.cpu(), rtol=1e-4)
        self.assertClose(dist1.cpu(), dist2.cpu(), rtol=6e-3)
        self.assertClose(bary1.cpu(), bary2.cpu(), rtol=1e-3)
        if not compare_grads:
            return

        # Compare gradients.
        torch.manual_seed(231)
        grad_zbuf = torch.randn_like(zbuf1)
        grad_dist = torch.randn_like(dist1)
        grad_bary = torch.randn_like(bary1)
        loss1 = (
            (dist1 * grad_dist).sum()
            + (zbuf1 * grad_zbuf).sum()
            + (bary1 * grad_bary).sum()
        )
        loss1.backward()
        grad_verts1 = grad_var1.grad.data.clone().cpu()

        grad_zbuf = grad_zbuf.to(zbuf2)
        grad_dist = grad_dist.to(dist2)
        grad_bary = grad_bary.to(bary2)
        loss2 = (
            (dist2 * grad_dist).sum()
            + (zbuf2 * grad_zbuf).sum()
            + (bary2 * grad_bary).sum()
        )
        grad_var1.grad.data.zero_()
        loss2.backward()
        grad_verts2 = grad_var2.grad.data.clone().cpu()
        self.assertClose(grad_verts1, grad_verts2, rtol=2e-3)

    def _test_perspective_correct(self, rasterize_meshes_fn, device, bin_size=None):
        # fmt: off
        verts = torch.tensor([
                [-0.4, -0.4, 10],  # noqa: E241, E201
                [ 0.4, -0.4, 10],  # noqa: E241, E201
                [ 0.0,  0.4, 20],  # noqa: E241, E201
        ], dtype=torch.float32, device=device)
        # fmt: on
        faces = torch.tensor([[0, 1, 2]], device=device)
        meshes = Meshes(verts=[verts], faces=[faces])
        kwargs = {
            "meshes": meshes,
            "image_size": 11,
            "faces_per_pixel": 1,
            "blur_radius": 0.2,
            "perspective_correct": False,
        }
        if bin_size != -1:
            kwargs["bin_size"] = bin_size

        # Run with and without perspective correction
        idx_f, zbuf_f, bary_f, dists_f = rasterize_meshes_fn(**kwargs)

        kwargs["perspective_correct"] = True
        idx_t, zbuf_t, bary_t, dists_t = rasterize_meshes_fn(**kwargs)

        # Expected output tensors in the format with axes +X left, +Y up, +Z in
        # idx and dists should be the same with or without perspecitve correction
        # fmt: off
        idx_expected = torch.tensor([
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
            [-1, -1, -1, -1,  0,  0,  0, -1, -1, -1, -1],  # noqa: E241, E201
            [-1, -1, -1,  0,  0,  0,  0,  0, -1, -1, -1],  # noqa: E241, E201
            [-1, -1, -1,  0,  0,  0,  0,  0, -1, -1, -1],  # noqa: E241, E201
            [-1, -1,  0,  0,  0,  0,  0,  0,  0, -1, -1],  # noqa: E241, E201
            [-1, -1,  0,  0,  0,  0,  0,  0,  0, -1, -1],  # noqa: E241, E201
            [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],  # noqa: E241, E201
            [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],  # noqa: E241, E201
            [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],  # noqa: E241, E201
            [-1, -1,  0,  0,  0,  0,  0,  0,  0, -1, -1],  # noqa: E241, E201
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]   # noqa: E241, E201
        ], dtype=torch.int64, device=device).view(1, 11, 11, 1)

        dists_expected = torch.tensor([
            [-1.,     -1.,     -1.,     -1.,    -1.,     -1.,     -1.,     -1.,     -1.,   -1., -1.],  # noqa: E241, E201
            [-1.,     -1.,     -1.,     -1., 0.1402,  0.1071,  0.1402,     -1.,     -1.,   -1., -1.],  # noqa: E241, E201
            [-1.,     -1., -    1., 0.1523,  0.0542,  0.0212,  0.0542,  0.1523,     -1.,   -1., -1.],  # noqa: E241, E201
            [-1.,     -1.,     -1., 0.0955,  0.0214, -0.0003,  0.0214,  0.0955,     -1.,   -1., -1.],  # noqa: E241, E201
            [-1.,     -1., 0.1523,  0.0518,  0.0042, -0.0095,  0.0042,  0.0518, 0.1523,    -1., -1.],  # noqa: E241, E201
            [-1.,     -1., 0.0955,  0.0214, -0.0003,  -0.032, -0.0003,  0.0214, 0.0955,    -1., -1.],  # noqa: E241, E201
            [-1., 0.1523,  0.0518,  0.0042, -0.0095, -0.0476, -0.0095,  0.0042, 0.0518, 0.1523, -1.],  # noqa: E241, E201
            [-1., 0.1084,  0.0225, -0.0003, -0.0013, -0.0013, -0.0013, -0.0003, 0.0225, 0.1084, -1.],  # noqa: E241, E201
            [-1., 0.1283,  0.0423,  0.0212,  0.0212,  0.0212,  0.0212,  0.0212, 0.0423, 0.1283, -1.],  # noqa: E241, E201
            [-1.,     -1., 0.1283,  0.1071,  0.1071,  0.1071,  0.1071,  0.1071, 0.1283,    -1., -1.],  # noqa: E241, E201
            [-1.,     -1.,     -1.,     -1.,     -1.,     -1.,     -1.,     -1.,    -1.,   -1., -1.]   # noqa: E241, E201
        ], dtype=torch.float32, device=device).view(1, 11, 11, 1)

        # zbuf and barycentric will be different with perspective correction
        zbuf_f_expected = torch.tensor([
            [-1.,      -1.,     -1.,     -1.,     -1.,     -1.,      -1.,    -1.,     -1.,     -1., -1.],  # noqa: E241, E201
            [-1.,      -1.,     -1.,     -1., 24.0909, 24.0909, 24.0909,     -1.,     -1.,     -1., -1.],  # noqa: E241, E201
            [-1.,      -1.,     -1., 21.8182, 21.8182, 21.8182, 21.8182, 21.8182,     -1.,     -1., -1.],  # noqa: E241, E201
            [-1.,      -1.,     -1., 19.5455, 19.5455, 19.5455, 19.5455, 19.5455,     -1.,     -1., -1.],  # noqa: E241, E201
            [-1.,      -1., 17.2727, 17.2727, 17.2727, 17.2727, 17.2727, 17.2727, 17.2727,     -1., -1.],  # noqa: E241, E201
            [-1.,      -1.,      15.,     15.,     15.,     15.,     15.,    15.,     15.,     -1., -1.],  # noqa: E241, E201
            [-1., 12.7273,  12.7273, 12.7273, 12.7273, 12.7273, 12.7273, 12.7273, 12.7273, 12.7273, -1.],  # noqa: E241, E201
            [-1., 10.4545,  10.4545, 10.4545, 10.4545, 10.4545, 10.4545, 10.4545, 10.4545, 10.4545, -1.],  # noqa: E241, E201
            [-1.,  8.1818,   8.1818,  8.1818,  8.1818,  8.1818,  8.1818,  8.1818,  8.1818,  8.1818, -1.],  # noqa: E241, E201
            [-1.,      -1.,  5.9091,  5.9091,  5.9091,  5.9091,  5.9091,  5.9091,  5.9091,     -1., -1.],  # noqa: E241, E201
            [-1.,       -1.,     -1.,     -1.,     -1.,     -1.,     -1.,     -1.,     -1.,    -1., -1.],  # noqa: E241, E201
        ], dtype=torch.float32, device=device).view(1, 11, 11, 1)

        zbuf_t_expected = torch.tensor([
             [-1.,     -1.,     -1.,     -1.,     -1.,     -1.,     -1.,     -1.,     -1.,     -1., -1.],  # noqa: E241, E201
             [-1.,     -1.,     -1.,     -1., 33.8461, 33.8462, 33.8462,     -1.,     -1.,     -1., -1.],  # noqa: E241, E201
             [-1.,     -1.,     -1., 24.4444, 24.4444, 24.4444, 24.4444, 24.4444,     -1.,     -1., -1.],  # noqa: E241, E201
             [-1.,     -1.,     -1., 19.1304, 19.1304, 19.1304, 19.1304, 19.1304,     -1.,     -1., -1.],  # noqa: E241, E201
             [-1.,     -1., 15.7143, 15.7143, 15.7143, 15.7143, 15.7143, 15.7143, 15.7143,     -1., -1.],  # noqa: E241, E201
             [-1.,     -1., 13.3333, 13.3333, 13.3333, 13.3333, 13.3333, 13.3333, 13.3333,     -1., -1.],  # noqa: E241, E201
             [-1., 11.5789, 11.5789, 11.5789, 11.5789, 11.5789, 11.5789, 11.5789, 11.5789, 11.5789, -1.],  # noqa: E241, E201
             [-1., 10.2326, 10.2326, 10.2326, 10.2326, 10.2326, 10.2326, 10.2326, 10.2326, 10.2326, -1.],  # noqa: E241, E201
             [-1.,  9.1667,  9.1667,  9.1667,  9.1667,  9.1667,  9.1667,  9.1667,  9.1667,  9.1667, -1.],  # noqa: E241, E201
             [-1.,      -1., 8.3019,  8.3019,  8.3019,  8.3019,  8.3019,  8.3019,  8.3019,     -1., -1.],  # noqa: E241, E201
             [-1.,      -1.,     -1.,    -1.,     -1.,     -1.,     -1.,     -1.,     -1.,     -1., -1.]   # noqa: E241, E201
        ], dtype=torch.float32, device=device).view(1, 11, 11, 1)
        # fmt: on

        self.assertTrue(torch.all(idx_f == idx_expected).item())
        self.assertTrue(torch.all(idx_t == idx_expected).item())
        dists_t_max_diff = (dists_t - dists_expected).abs().max().item()
        dists_f_max_diff = (dists_f - dists_expected).abs().max().item()
        self.assertLess(dists_t_max_diff, 1e-4)
        self.assertLess(dists_f_max_diff, 1e-4)
        zbuf_f_max_diff = (zbuf_f - zbuf_f_expected).abs().max().item()
        zbuf_t_max_diff = (zbuf_t - zbuf_t_expected).abs().max().item()
        self.assertLess(zbuf_f_max_diff, 1e-4)
        self.assertLess(zbuf_t_max_diff, 1e-4)

        # Check barycentrics by using them to re-compute zbuf
        z0 = verts[0, 2]
        z1 = verts[1, 2]
        z2 = verts[2, 2]
        w0_f, w1_f, w2_f = bary_f.unbind(dim=4)
        w0_t, w1_t, w2_t = bary_t.unbind(dim=4)
        zbuf_f_bary = w0_f * z0 + w1_f * z1 + w2_f * z2
        zbuf_t_bary = w0_t * z0 + w1_t * z1 + w2_t * z2
        mask = idx_expected != -1
        zbuf_f_bary_diff = (zbuf_f_bary[mask] - zbuf_f_expected[mask]).abs().max()
        zbuf_t_bary_diff = (zbuf_t_bary[mask] - zbuf_t_expected[mask]).abs().max()
        self.assertLess(zbuf_f_bary_diff, 1e-4)
        self.assertLess(zbuf_t_bary_diff, 1e-4)

    def _test_behind_camera(self, rasterize_meshes_fn, device, bin_size=None):
        """
        All verts are behind the camera so nothing should get rasterized.
        """
        N = 1
        # fmt: off
        verts = torch.tensor(
            [
                [ -0.5, 0.0, -0.1],  # noqa: E241, E201
                [  0.0, 0.6, -0.1],  # noqa: E241, E201
                [  0.5, 0.0, -0.1],  # noqa: E241, E201
                [-0.25, 0.0, -0.9],  # noqa: E241, E201
                [ 0.25, 0.5, -0.9],  # noqa: E241, E201
                [ 0.75, 0.0, -0.9],  # noqa: E241, E201
                [ -0.4, 0.0, -0.5],  # noqa: E241, E201
                [  0.6, 0.6, -0.5],  # noqa: E241, E201
                [  0.8, 0.0, -0.5],  # noqa: E241, E201
                [ -0.2, 0.0, -0.5],  # noqa: E241, E201
                [  0.3, 0.6, -0.5],  # noqa: E241, E201
                [  0.4, 0.0, -0.5],  # noqa: E241, E201
            ],
            dtype=torch.float32,
            device=device,
        )
        # fmt: on
        faces = torch.tensor(
            [[1, 0, 2], [4, 3, 5], [7, 6, 8], [10, 9, 11]],
            dtype=torch.int64,
            device=device,
        )
        meshes = Meshes(verts=[verts], faces=[faces])
        image_size = 16
        faces_per_pixel = 1
        radius = 0.2
        idx_expected = torch.full(
            (N, image_size, image_size, faces_per_pixel),
            fill_value=-1,
            dtype=torch.int64,
            device=device,
        )
        bary_expected = torch.full(
            (N, image_size, image_size, faces_per_pixel, 3),
            fill_value=-1,
            dtype=torch.float32,
            device=device,
        )
        zbuf_expected = torch.full(
            (N, image_size, image_size, faces_per_pixel),
            fill_value=-1,
            dtype=torch.float32,
            device=device,
        )
        dists_expected = zbuf_expected.clone()
        if bin_size == -1:
            # naive python version with no binning
            idx, zbuf, bary, dists = rasterize_meshes_fn(
                meshes, image_size, radius, faces_per_pixel
            )
        else:
            idx, zbuf, bary, dists = rasterize_meshes_fn(
                meshes, image_size, radius, faces_per_pixel, bin_size
            )
        idx_same = (idx == idx_expected).all().item()
        zbuf_same = (zbuf == zbuf_expected).all().item()
        self.assertTrue(idx_same)
        self.assertTrue(zbuf_same)
        self.assertClose(bary, bary_expected)
        self.assertClose(dists, dists_expected)

    def _simple_triangle_raster(self, raster_fn, device, bin_size=None):
        image_size = 10

        # Mesh with a single non-symmetrical face - this will help
        # check that the XY directions are correctly oriented.
        verts0 = torch.tensor(
            [[-0.3, -0.4, 0.1], [0.0, 0.6, 0.1], [0.9, -0.4, 0.1]],
            dtype=torch.float32,
            device=device,
        )
        faces0 = torch.tensor([[1, 0, 2]], dtype=torch.int64, device=device)

        # Mesh with two overlapping faces.
        # fmt: off
        verts1 = torch.tensor(
            [
                [-0.9, -0.2, 0.1],  # noqa: E241, E201
                [ 0.0,  0.6, 0.1],  # noqa: E241, E201
                [ 0.7, -0.4, 0.1],  # noqa: E241, E201
                [-0.7,  0.4, 0.5],  # noqa: E241, E201
                [ 0.0, -0.6, 0.5],  # noqa: E241, E201
                [ 0.7,  0.4, 0.5],  # noqa: E241, E201
            ],
            dtype=torch.float32,
            device=device,
        )
        # fmt on
        faces1 = torch.tensor(
            [[1, 0, 2], [3, 4, 5]], dtype=torch.int64, device=device
        )

        # Expected output tensors in the format with axes +X left, +Y up, +Z in
        # k = 0, closest point.
        # fmt off
        expected_p2face_k0 = torch.tensor(
            [
                [
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
                    [-1, -1, -1, -1,  0, -1, -1, -1, -1, -1],  # noqa: E241, E201
                    [-1, -1, -1,  0,  0,  0, -1, -1, -1, -1],  # noqa: E241, E201
                    [-1, -1,  0,  0,  0,  0, -1, -1, -1, -1],  # noqa: E241, E201
                    [-1,  0,  0,  0,  0,  0, -1, -1, -1, -1],  # noqa: E241, E201
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
                ],
                [
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
                    [-1, -1, -1, -1, -1,  1, -1, -1, -1, -1],  # noqa: E241, E201
                    [-1, -1,  2,  2,  1,  1,  1,  2, -1, -1],  # noqa: E241, E201
                    [-1, -1, -1,  1,  1,  1,  1,  1, -1, -1],  # noqa: E241, E201
                    [-1, -1, -1,  1,  1,  1,  1,  1,  1, -1],  # noqa: E241, E201
                    [-1, -1,  1,  1,  1,  2, -1, -1, -1, -1],  # noqa: E241, E201
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
                ],
            ],
            dtype=torch.int64,
            device=device,
        )
        expected_zbuf_k0 = torch.tensor(
        [
            [
                [-1,  -1,  -1,  -1,  -1,  -1, -1, -1, -1, -1],  # noqa: E241, E201
                [-1,  -1,  -1,  -1,  -1,  -1, -1, -1, -1, -1],  # noqa: E241, E201
                [-1,  -1,  -1,  -1,  -1,  -1, -1, -1, -1, -1],  # noqa: E241, E201
                [-1,  -1,  -1,  -1, 0.1,  -1, -1, -1, -1, -1],  # noqa: E241, E201
                [-1,  -1,  -1, 0.1, 0.1, 0.1, -1, -1, -1, -1],  # noqa: E241, E201
                [-1,  -1, 0.1, 0.1, 0.1, 0.1, -1, -1, -1, -1],  # noqa: E241, E201
                [-1, 0.1, 0.1, 0.1, 0.1, 0.1, -1, -1, -1, -1],  # noqa: E241, E201
                [-1,  -1,  -1,  -1,  -1,  -1, -1, -1, -1, -1],  # noqa: E241, E201
                [-1,  -1,  -1,  -1,  -1,  -1, -1, -1, -1, -1],  # noqa: E241, E201
                [-1,  -1,  -1,  -1,  -1,  -1, -1, -1, -1, -1]   # noqa: E241, E201
            ],
            [
                [-1, -1,  -1,  -1,  -1, -1,   -1,  -1,  -1, -1],  # noqa: E241, E201
                [-1, -1,  -1,  -1,  -1, -1,   -1,  -1,  -1, -1],  # noqa: E241, E201
                [-1, -1,  -1,  -1,  -1, 0.1,  -1,  -1,  -1, -1],  # noqa: E241, E201
                [-1, -1, 0.5, 0.5, 0.1, 0.1, 0.1, 0.5,  -1, -1],  # noqa: E241, E201
                [-1, -1,  -1, 0.1, 0.1, 0.1, 0.1, 0.1,  -1, -1],  # noqa: E241, E201
                [-1, -1,  -1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, -1],  # noqa: E241, E201
                [-1, -1, 0.1, 0.1, 0.1, 0.5,  -1,  -1,  -1, -1],  # noqa: E241, E201
                [-1, -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, -1],  # noqa: E241, E201
                [-1, -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, -1],  # noqa: E241, E201
                [-1, -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, -1]   # noqa: E241, E201
            ]
        ],
            device=device,
        )
        # fmt: on

        meshes = Meshes(verts=[verts0, verts1], faces=[faces0, faces1])

        # k = 1, second closest point.
        expected_p2face_k1 = expected_p2face_k0.clone()
        expected_p2face_k1[0, :] = torch.ones_like(expected_p2face_k1[0, :]) * -1

        # fmt: off
        expected_p2face_k1[1, :] = torch.tensor(
        [
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
            [-1, -1, -1, -1,  2,  2,  2, -1, -1, -1],  # noqa: E241, E201
            [-1, -1, -1,  2,  2,  2,  2, -1, -1, -1],  # noqa: E241, E201
            [-1, -1, -1,  2,  2,  2,  2, -1, -1, -1],  # noqa: E241, E201
            [-1, -1, -1, -1,  2, -1, -1, -1, -1, -1],  # noqa: E241, E201
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]   # noqa: E241, E201
        ],
            dtype=torch.int64,
            device=device,
        )
        expected_zbuf_k1 = expected_zbuf_k0.clone()
        expected_zbuf_k1[0, :] = torch.ones_like(expected_zbuf_k1[0, :]) * -1
        expected_zbuf_k1[1, :] = torch.tensor(
            [
                [-1., -1., -1.,  -1.,  -1.,  -1., -1., -1., -1., -1.],  # noqa: E241, E201
                [-1., -1., -1.,  -1.,  -1.,  -1., -1., -1., -1., -1.],  # noqa: E241, E201
                [-1., -1., -1.,  -1.,  -1.,  -1., -1., -1., -1., -1.],  # noqa: E241, E201
                [-1., -1., -1.,  -1.,  0.5, 0.5,  0.5, -1., -1., -1.],  # noqa: E241, E201
                [-1., -1., -1.,  0.5,  0.5, 0.5,  0.5, -1., -1., -1.],  # noqa: E241, E201
                [-1., -1., -1.,  0.5,  0.5, 0.5,  0.5, -1., -1., -1.],  # noqa: E241, E201
                [-1., -1., -1.,  -1.,  0.5,  -1., -1., -1., -1., -1.],  # noqa: E241, E201
                [-1., -1., -1.,  -1.,  -1.,  -1., -1., -1., -1., -1.],  # noqa: E241, E201
                [-1., -1., -1.,  -1.,  -1.,  -1., -1., -1., -1., -1.],  # noqa: E241, E201
                [-1., -1., -1.,  -1.,  -1.,  -1., -1., -1., -1., -1.]   # noqa: E241, E201
            ],
            dtype=torch.float32,
            device=device,
        )
        # fmt: on

        #  Coordinate conventions +Y up, +Z in, +X left
        if bin_size == -1:
            # simple python, no bin_size
            p2face, zbuf, bary, pix_dists = raster_fn(meshes, image_size, 0.0, 2)
        else:
            p2face, zbuf, bary, pix_dists = raster_fn(
                meshes, image_size, 0.0, 2, bin_size
            )

        self.assertClose(p2face[..., 0], expected_p2face_k0)
        self.assertClose(zbuf[..., 0], expected_zbuf_k0)
        self.assertClose(p2face[..., 1], expected_p2face_k1)
        self.assertClose(zbuf[..., 1], expected_zbuf_k1)

    def _simple_blurry_raster(self, raster_fn, device, bin_size=None):
        """
        Check that pix_to_face, dist and zbuf values are invariant to the
        ordering of faces.
        """
        image_size = 10
        blur_radius = 0.12 ** 2
        faces_per_pixel = 1

        # fmt: off
        verts = torch.tensor(
            [
                [ -0.3, 0.0,  0.1],  # noqa: E241, E201
                [  0.0, 0.6,  0.1],  # noqa: E241, E201
                [  0.8, 0.0,  0.1],  # noqa: E241, E201
                [-0.25, 0.0,  0.9],  # noqa: E241, E201
                [0.25,  0.5,  0.9],  # noqa: E241, E201
                [0.75,  0.0,  0.9],  # noqa: E241, E201
                [-0.4,  0.0,  0.5],  # noqa: E241, E201
                [ 0.6,  0.6,  0.5],  # noqa: E241, E201
                [ 0.8,  0.0,  0.5],  # noqa: E241, E201
                [-0.2,  0.0, -0.5],  # noqa: E241, E201  face behind the camera
                [ 0.3,  0.6, -0.5],  # noqa: E241, E201
                [ 0.4,  0.0, -0.5],  # noqa: E241, E201
            ],
            dtype=torch.float32,
            device=device,
        )
        # Face with index 0 is non symmetric about the X and Y axis to
        # test that the positive Y and X directions are correct in the output.
        faces_packed = torch.tensor(
            [[1, 0, 2], [4, 3, 5], [7, 6, 8], [10, 9, 11]],
            dtype=torch.int64,
            device=device,
        )
        expected_p2f = torch.tensor(
            [
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
                [-1,  2,  2,  0,  0,  0, -1, -1, -1, -1],  # noqa: E241, E201
                [-1,  2,  0,  0,  0,  0, -1, -1, -1, -1],  # noqa: E241, E201
                [-1,  0,  0,  0,  0,  0,  0, -1, -1, -1],  # noqa: E241, E201
                [-1,  0,  0,  0,  0,  0,  0, -1, -1, -1],  # noqa: E241, E201
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
            ],
            dtype=torch.int64,
            device=device,
        )
        expected_zbuf = torch.tensor(
            [
                [-1.,   -1.,  -1.,  -1.,  -1.,  -1., -1., -1., -1., -1.],  # noqa: E241, E201
                [-1.,   -1.,  -1.,  -1.,  -1.,  -1., -1., -1., -1., -1.],  # noqa: E241, E201
                [-1.,  0.5,  0.5,  0.1,  0.1,  0.1,  -1., -1., -1., -1.],  # noqa: E241, E201
                [-1.,  0.5,  0.1,  0.1,  0.1,  0.1,  -1., -1., -1., -1.],  # noqa: E241, E201
                [-1.,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1, -1., -1., -1.],  # noqa: E241, E201
                [-1.,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1, -1., -1., -1.],  # noqa: E241, E201
                [-1.,   -1.,  -1.,  -1.,  -1.,  -1., -1., -1., -1., -1.],  # noqa: E241, E201
                [-1.,   -1.,  -1.,  -1.,  -1.,  -1., -1., -1., -1., -1.],  # noqa: E241, E201
                [-1.,   -1.,  -1.,  -1.,  -1.,  -1., -1., -1., -1., -1.],  # noqa: E241, E201
                [-1.,   -1.,  -1.,  -1.,  -1.,  -1., -1., -1., -1., -1.]   # noqa: E241, E201
            ],
            dtype=torch.float32,
            device=device,
        )
        # fmt: on

        for i, order in enumerate([[0, 1, 2], [1, 2, 0], [2, 0, 1]]):
            faces = faces_packed[order]  # rearrange order of faces.
            mesh = Meshes(verts=[verts], faces=[faces])
            if bin_size == -1:
                # simple python, no bin size arg
                pix_to_face, zbuf, bary_coords, dists = raster_fn(
                    mesh, image_size, blur_radius, faces_per_pixel
                )
            else:
                pix_to_face, zbuf, bary_coords, dists = raster_fn(
                    mesh, image_size, blur_radius, faces_per_pixel, bin_size
                )
            if i == 0:
                expected_dists = dists
            p2f = expected_p2f.clone()
            p2f[expected_p2f == 0] = order.index(0)
            p2f[expected_p2f == 1] = order.index(1)
            p2f[expected_p2f == 2] = order.index(2)
            self.assertClose(pix_to_face.squeeze(), p2f)
            self.assertClose(zbuf.squeeze(), expected_zbuf, rtol=1e-5)
            self.assertClose(dists, expected_dists)

    def _test_coarse_rasterize(self, device):
        image_size = 16
        # No blurring. This test checks that the XY directions are
        # correctly oriented.
        blur_radius = 0.0
        bin_size = 8
        max_faces_per_bin = 3

        # fmt: off
        verts = torch.tensor(
            [
                [-0.5,   0.1,  0.1],  # noqa: E241, E201
                [-0.3,   0.6,  0.1],  # noqa: E241, E201
                [-0.1,   0.1,  0.1],  # noqa: E241, E201
                [-0.3,  -0.1,  0.4],  # noqa: E241, E201
                [ 0.3,   0.5,  0.4],  # noqa: E241, E201
                [0.75,  -0.1,  0.4],  # noqa: E241, E201
                [ 0.2,  -0.3,  0.9],  # noqa: E241, E201
                [ 0.3,  -0.7,  0.9],  # noqa: E241, E201
                [ 0.6,  -0.3,  0.9],  # noqa: E241, E201
                [-0.4,   0.0, -1.5],  # noqa: E241, E201
                [ 0.6,   0.6, -1.5],  # noqa: E241, E201
                [ 0.8,   0.0, -1.5],  # noqa: E241, E201
            ],
            device=device,
        )
        # Expected faces using axes convention +Y down, + X right, +Z in
        # Non symmetrical triangles i.e face 0 and 3 are in one bin only
        faces = torch.tensor(
            [
                [ 1, 0,  2],  # noqa: E241, E201  bin 01 only
                [ 4, 3,  5],  # noqa: E241, E201  all bins
                [ 7, 6,  8],  # noqa: E241, E201  bin 10 only
                [10, 9, 11],  # noqa: E241, E201  negative z, should not appear.
            ],
            dtype=torch.int64,
            device=device,
        )
        # fmt: on

        meshes = Meshes(verts=[verts], faces=[faces])
        faces_verts = verts[faces]
        num_faces_per_mesh = meshes.num_faces_per_mesh()
        mesh_to_face_first_idx = meshes.mesh_to_faces_packed_first_idx()

        # Expected faces using axes convention +Y down, + X right, + Z in
        bin_faces_expected = (
            torch.ones((1, 2, 2, max_faces_per_bin), dtype=torch.int32, device=device)
            * -1
        )
        bin_faces_expected[0, 1, 1, 0] = torch.tensor([1])
        bin_faces_expected[0, 0, 1, 0:2] = torch.tensor([1, 2])
        bin_faces_expected[0, 1, 0, 0:2] = torch.tensor([0, 1])
        bin_faces_expected[0, 0, 0, 0] = torch.tensor([1])

        # +Y up, +X left, +Z in
        bin_faces = _C._rasterize_meshes_coarse(
            faces_verts,
            mesh_to_face_first_idx,
            num_faces_per_mesh,
            image_size,
            blur_radius,
            bin_size,
            max_faces_per_bin,
        )

        bin_faces_same = (bin_faces.squeeze() == bin_faces_expected).all()
        self.assertTrue(bin_faces_same.item() == 1)

    @staticmethod
    def rasterize_meshes_python_with_init(
        num_meshes: int, ico_level: int, image_size: int, blur_radius: float
    ):
        device = torch.device("cpu")
        meshes = ico_sphere(ico_level, device)
        meshes_batch = meshes.extend(num_meshes)

        def rasterize():
            rasterize_meshes_python(meshes_batch, image_size, blur_radius)

        return rasterize

    @staticmethod
    def rasterize_meshes_cpu_with_init(
        num_meshes: int, ico_level: int, image_size: int, blur_radius: float
    ):
        meshes = ico_sphere(ico_level, torch.device("cpu"))
        meshes_batch = meshes.extend(num_meshes)

        def rasterize():
            rasterize_meshes(meshes_batch, image_size, blur_radius, bin_size=0)

        return rasterize

    @staticmethod
    def rasterize_meshes_cuda_with_init(
        num_meshes: int,
        ico_level: int,
        image_size: int,
        blur_radius: float,
        bin_size: int,
        max_faces_per_bin: int,
    ):

        meshes = ico_sphere(ico_level, get_random_cuda_device())
        meshes_batch = meshes.extend(num_meshes)
        torch.cuda.synchronize()

        def rasterize():
            rasterize_meshes(
                meshes_batch, image_size, blur_radius, 8, bin_size, max_faces_per_bin
            )
            torch.cuda.synchronize()

        return rasterize
