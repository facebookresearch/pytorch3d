#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import torch

from pytorch3d import _C
from pytorch3d.renderer.mesh.rasterize_meshes import (
    rasterize_meshes,
    rasterize_meshes_python,
)
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere


class TestRasterizeMeshes(unittest.TestCase):
    def test_simple_python(self):
        device = torch.device("cpu")
        self._simple_triangle_raster(
            rasterize_meshes_python, device, bin_size=-1
        )  # don't set binsize
        self._simple_blurry_raster(rasterize_meshes_python, device, bin_size=-1)
        self._test_behind_camera(rasterize_meshes_python, device, bin_size=-1)

    def test_simple_cpu_naive(self):
        device = torch.device("cpu")
        self._simple_triangle_raster(rasterize_meshes, device)
        self._simple_blurry_raster(rasterize_meshes, device)
        self._test_behind_camera(rasterize_meshes, device)

    def test_simple_cuda_naive(self):
        device = torch.device("cuda:0")
        self._simple_triangle_raster(rasterize_meshes, device, bin_size=0)
        self._simple_blurry_raster(rasterize_meshes, device, bin_size=0)
        self._test_behind_camera(rasterize_meshes, device, bin_size=0)

    def test_simple_cuda_binned(self):
        device = torch.device("cuda:0")
        self._simple_triangle_raster(rasterize_meshes, device, bin_size=5)
        self._simple_blurry_raster(rasterize_meshes, device, bin_size=5)
        self._test_behind_camera(rasterize_meshes, device, bin_size=5)

    def test_python_vs_cpu_vs_cuda(self):
        torch.manual_seed(231)
        device = torch.device("cpu")
        image_size = 32
        blur_radius = 0.1 ** 2
        faces_per_pixel = 3

        for d in ["cpu", "cuda"]:
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

        device = torch.device("cuda:0")
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
        return self._test_coarse_rasterize(torch.device("cuda:0"))

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
        device = torch.device("cuda:0")
        meshes = ico_sphere(0, device)
        verts, faces = meshes.get_mesh_verts_faces(0)
        verts.requires_grad = True
        meshes = Meshes(verts=[verts], faces=[faces])

        args = (meshes, image_size, radius, faces_per_pixel, 0, 0)
        idx2, zbuf2, bary2, dist2 = rasterize_meshes(*args)
        grad_zbuf = grad_zbuf.cuda()
        grad_dist = grad_dist.cuda()
        grad_bary = grad_bary.cuda()
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
        device = torch.device("cuda:0")
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
        self.assertTrue(torch.allclose(zbuf1, zbuf2, atol=1e-6))
        self.assertTrue(torch.allclose(zbuf1, zbuf3, atol=1e-6))
        self.assertTrue(torch.allclose(dist1, dist2, atol=1e-6))
        self.assertTrue(torch.allclose(dist1, dist3, atol=1e-6))

        self.assertTrue(torch.allclose(grad1, grad2, rtol=5e-3))  # flaky test
        self.assertTrue(torch.allclose(grad1, grad3, rtol=5e-3))
        self.assertTrue(torch.allclose(grad2, grad3, rtol=5e-3))

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
        args = (
            faces_verts,
            mesh_to_face_first_idx,
            num_faces_per_mesh,
            image_size,
            blur_radius,
            bin_size,
            max_faces_per_bin,
        )
        bin_faces_cpu = _C._rasterize_meshes_coarse(*args)

        device = torch.device("cuda:0")
        meshes = ico_sphere(2, device)

        faces = meshes.faces_packed()
        verts = meshes.verts_packed()
        faces_verts = verts[faces]
        num_faces_per_mesh = meshes.num_faces_per_mesh()
        mesh_to_face_first_idx = meshes.mesh_to_faces_packed_first_idx()
        args = (
            faces_verts,
            mesh_to_face_first_idx,
            num_faces_per_mesh,
            image_size,
            blur_radius,
            bin_size,
            max_faces_per_bin,
        )
        bin_faces_cuda = _C._rasterize_meshes_coarse(*args)

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
        self.assertTrue(torch.allclose(zbuf1.cpu(), zbuf2.cpu(), rtol=1e-4))
        self.assertTrue(torch.allclose(dist1.cpu(), dist2.cpu(), rtol=6e-3))
        self.assertTrue(torch.allclose(bary1.cpu(), bary2.cpu(), rtol=1e-3))
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
        self.assertTrue(torch.allclose(grad_verts1, grad_verts2, rtol=1e-3))

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
        self.assertTrue(torch.allclose(bary, bary_expected))
        self.assertTrue(torch.allclose(dists, dists_expected))

    def _simple_triangle_raster(self, raster_fn, device, bin_size=None):
        image_size = 10

        # Mesh with a single face.
        verts0 = torch.tensor(
            [[-0.7, -0.4, 0.1], [0.0, 0.6, 0.1], [0.7, -0.4, 0.1]],
            dtype=torch.float32,
            device=device,
        )
        faces0 = torch.tensor([[1, 0, 2]], dtype=torch.int64, device=device)

        # Mesh with two overlapping faces.
        # fmt: off
        verts1 = torch.tensor(
            [
                [-0.7, -0.4, 0.1],  # noqa: E241, E201
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

        # fmt off
        expected_p2face_k0 = torch.tensor(
            [
                [
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
                    [-1, -1,  0,  0,  0,  0,  0,  0, -1, -1],  # noqa: E241, E201
                    [-1, -1, -1,  0,  0,  0,  0, -1, -1, -1],  # noqa: E241, E201
                    [-1, -1, -1,  0,  0,  0,  0, -1, -1, -1],  # noqa: E241, E201
                    [-1, -1, -1, -1,  0,  0, -1, -1, -1, -1],  # noqa: E241, E201
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
                ],
                [
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
                    [-1, -1,  1,  1,  1,  1,  1,  1, -1, -1],  # noqa: E241, E201
                    [-1, -1, -1,  1,  1,  1,  1, -1, -1, -1],  # noqa: E241, E201
                    [-1, -1, -1,  1,  1,  1,  1, -1, -1, -1],  # noqa: E241, E201
                    [-1, -1,  2,  2,  1,  1,  2,  2, -1, -1],  # noqa: E241, E201
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
                    [-1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1, -1],  # noqa: E241, E201
                    [-1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1, -1],  # noqa: E241, E201
                    [-1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1, -1],  # noqa: E241, E201
                    [-1, -1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, -1, -1],  # noqa: E241, E201
                    [-1, -1,  -1, 0.1, 0.1, 0.1, 0.1,  -1, -1, -1],  # noqa: E241, E201
                    [-1, -1,  -1, 0.1, 0.1, 0.1, 0.1,  -1, -1, -1],  # noqa: E241, E201
                    [-1, -1,  -1,  -1, 0.1, 0.1,  -1,  -1, -1, -1],  # noqa: E241, E201
                    [-1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1, -1],  # noqa: E241, E201
                    [-1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1, -1],  # noqa: E241, E201
                    [-1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1, -1],  # noqa: E241, E201
                ],
                [
                    [-1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1, -1],  # noqa: E241, E201
                    [-1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1, -1],  # noqa: E241, E201
                    [-1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1, -1],  # noqa: E241, E201
                    [-1, -1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, -1, -1],  # noqa: E241, E201
                    [-1, -1,  -1, 0.1, 0.1, 0.1, 0.1,  -1, -1, -1],  # noqa: E241, E201
                    [-1, -1,  -1, 0.1, 0.1, 0.1, 0.1,  -1, -1, -1],  # noqa: E241, E201
                    [-1, -1, 0.5, 0.5, 0.1, 0.1, 0.5, 0.5, -1, -1],  # noqa: E241, E201
                    [-1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1, -1],  # noqa: E241, E201
                    [-1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1, -1],  # noqa: E241, E201
                    [-1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1, -1],  # noqa: E241, E201
                ],
            ],
            device=device,
        )
        # fmt: on

        meshes = Meshes(verts=[verts0, verts1], faces=[faces0, faces1])
        if bin_size == -1:
            # simple python case with no binning
            p2face, zbuf, bary, pix_dists = raster_fn(
                meshes, image_size, 0.0, 2
            )
        else:
            p2face, zbuf, bary, pix_dists = raster_fn(
                meshes, image_size, 0.0, 2, bin_size
            )
        # k = 0, closest point.
        self.assertTrue(torch.allclose(p2face[..., 0], expected_p2face_k0))
        self.assertTrue(torch.allclose(zbuf[..., 0], expected_zbuf_k0))

        # k = 1, second closest point.
        expected_p2face_k1 = expected_p2face_k0.clone()
        expected_p2face_k1[0, :] = (
            torch.ones_like(expected_p2face_k1[0, :]) * -1
        )

        # fmt: off
        expected_p2face_k1[1, :] = torch.tensor(
            [
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
                [-1, -1, -1, -1,  2,  2, -1, -1, -1, -1],  # noqa: E241, E201
                [-1, -1, -1,  2,  2,  2,  2, -1, -1, -1],  # noqa: E241, E201
                [-1, -1, -1,  2,  2,  2,  2, -1, -1, -1],  # noqa: E241, E201
                [-1, -1, -1, -1,  2,  2, -1, -1, -1, -1],  # noqa: E241, E201
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
            ],
            dtype=torch.int64,
            device=device,
        )
        expected_zbuf_k1 = expected_zbuf_k0.clone()
        expected_zbuf_k1[0, :] = torch.ones_like(expected_zbuf_k1[0, :]) * -1
        expected_zbuf_k1[1, :] = torch.tensor(
            [
                [-1, -1, -1,  -1,  -1,  -1,  -1, -1, -1, -1],  # noqa: E241, E201
                [-1, -1, -1,  -1,  -1,  -1,  -1, -1, -1, -1],  # noqa: E241, E201
                [-1, -1, -1,  -1,  -1,  -1,  -1, -1, -1, -1],  # noqa: E241, E201
                [-1, -1, -1,  -1, 0.5, 0.5,  -1, -1, -1, -1],  # noqa: E241, E201
                [-1, -1, -1, 0.5, 0.5, 0.5, 0.5, -1, -1, -1],  # noqa: E241, E201
                [-1, -1, -1, 0.5, 0.5, 0.5, 0.5, -1, -1, -1],  # noqa: E241, E201
                [-1, -1, -1,  -1, 0.5, 0.5,  -1, -1, -1, -1],  # noqa: E241, E201
                [-1, -1, -1,  -1,  -1,  -1,  -1, -1, -1, -1],  # noqa: E241, E201
                [-1, -1, -1,  -1,  -1,  -1,  -1, -1, -1, -1],  # noqa: E241, E201
                [-1, -1, -1,  -1,  -1,  -1,  -1, -1, -1, -1],  # noqa: E241, E201
            ],
            dtype=torch.float32,
            device=device,
        )
        # fmt: on
        self.assertTrue(torch.allclose(p2face[..., 1], expected_p2face_k1))
        self.assertTrue(torch.allclose(zbuf[..., 1], expected_zbuf_k1))

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
                [ -0.5, 0.0,  0.1],  # noqa: E241, E201
                [  0.0, 0.6,  0.1],  # noqa: E241, E201
                [  0.5, 0.0,  0.1],  # noqa: E241, E201
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
        faces_packed = torch.tensor(
            [[1, 0, 2], [4, 3, 5], [7, 6, 8], [10, 9, 11]],
            dtype=torch.int64,
            device=device,
        )
        expected_p2f = torch.tensor(
            [
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
                [-1, -1,  0,  0,  0,  0,  0,  0,  2, -1],  # noqa: E241, E201
                [-1, -1,  0,  0,  0,  0,  0,  0,  2, -1],  # noqa: E241, E201
                [-1, -1, -1,  0,  0,  0,  0,  2,  2, -1],  # noqa: E241, E201
                [-1, -1, -1, -1,  0,  0,  2,  2,  2, -1],  # noqa: E241, E201
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # noqa: E241, E201
            ],
            dtype=torch.int64,
            device=device,
        )
        expected_zbuf = torch.tensor(
            [
                [-1, -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, -1],  # noqa: E241, E201
                [-1, -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, -1],  # noqa: E241, E201
                [-1, -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, -1],  # noqa: E241, E201
                [-1, -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, -1],  # noqa: E241, E201
                [-1, -1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, -1],  # noqa: E241, E201
                [-1, -1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, -1],  # noqa: E241, E201
                [-1, -1,  -1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5, -1],  # noqa: E241, E201
                [-1, -1,  -1,  -1, 0.1, 0.1, 0.5, 0.5, 0.5, -1],  # noqa: E241, E201
                [-1, -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, -1],  # noqa: E241, E201
                [-1, -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, -1],  # noqa: E241, E201
            ],
            dtype=torch.float32,
            device=device,
        )
        # fmt: on

        for i, order in enumerate([[0, 1, 2], [1, 2, 0], [2, 0, 1]]):
            faces = faces_packed[order]  # rearrange order of faces.
            mesh = Meshes(verts=[verts], faces=[faces])
            if bin_size == -1:
                # simple python case with no binning
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

            self.assertTrue(torch.allclose(pix_to_face.squeeze(), p2f))
            self.assertTrue(
                torch.allclose(zbuf.squeeze(), expected_zbuf, rtol=1e-5)
            )
            self.assertTrue(torch.allclose(dists, expected_dists))

    def _test_coarse_rasterize(self, device):
        image_size = 16
        blur_radius = 0.2 ** 2
        bin_size = 8
        max_faces_per_bin = 3

        # fmt: off
        verts = torch.tensor(
            [
                [-0.5,  0.0,  0.1],  # noqa: E241, E201
                [ 0.0,  0.6,  0.1],  # noqa: E241, E201
                [ 0.5,  0.0,  0.1],  # noqa: E241, E201
                [-0.3,  0.0,  0.4],  # noqa: E241, E201
                [ 0.3,  0.5,  0.4],  # noqa: E241, E201
                [0.75,  0.0,  0.4],  # noqa: E241, E201
                [-0.4, -0.3,  0.9],  # noqa: E241, E201
                [ 0.2, -0.7,  0.9],  # noqa: E241, E201
                [ 0.4, -0.3,  0.9],  # noqa: E241, E201
                [-0.4,  0.0, -1.5],  # noqa: E241, E201
                [ 0.6,  0.6, -1.5],  # noqa: E241, E201
                [ 0.8,  0.0, -1.5],  # noqa: E241, E201
            ],
            device=device,
        )
        faces = torch.tensor(
            [
                [ 1, 0,  2],  # noqa: E241, E201  bin 00 and bin 01
                [ 4, 3,  5],  # noqa: E241, E201  bin 00 and bin 01
                [ 7, 6,  8],  # noqa: E241, E201  bin 10 and bin 11
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

        bin_faces_expected = (
            torch.ones(
                (1, 2, 2, max_faces_per_bin), dtype=torch.int32, device=device
            )
            * -1
        )
        bin_faces_expected[0, 0, 0, 0:2] = torch.tensor([0, 1])
        bin_faces_expected[0, 0, 1, 0:2] = torch.tensor([0, 1])
        bin_faces_expected[0, 1, 0, 0:3] = torch.tensor([0, 1, 2])
        bin_faces_expected[0, 1, 1, 0:3] = torch.tensor([0, 1, 2])
        bin_faces = _C._rasterize_meshes_coarse(
            faces_verts,
            mesh_to_face_first_idx,
            num_faces_per_mesh,
            image_size,
            blur_radius,
            bin_size,
            max_faces_per_bin,
        )
        bin_faces_same = (
            bin_faces.squeeze().flip(dims=[0]) == bin_faces_expected
        ).all()
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

        meshes = ico_sphere(ico_level, torch.device("cuda:0"))
        meshes_batch = meshes.extend(num_meshes)
        torch.cuda.synchronize()

        def rasterize():
            rasterize_meshes(
                meshes_batch,
                image_size,
                blur_radius,
                8,
                bin_size,
                max_faces_per_bin,
            )
            torch.cuda.synchronize()

        return rasterize
