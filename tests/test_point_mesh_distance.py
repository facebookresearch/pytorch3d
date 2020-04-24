# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest

import numpy as np
import torch
from common_testing import TestCaseMixin, get_random_cuda_device
from pytorch3d import _C
from pytorch3d.loss import point_mesh_edge_distance, point_mesh_face_distance
from pytorch3d.structures import Meshes, Pointclouds, packed_to_list


class TestPointMeshDistance(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)
        torch.manual_seed(42)

    @staticmethod
    def eps():
        return 1e-8

    @staticmethod
    def init_meshes_clouds(
        batch_size: int = 10,
        num_verts: int = 1000,
        num_faces: int = 3000,
        num_points: int = 3000,
        device: str = "cuda:0",
    ):
        device = torch.device(device)
        nump = torch.randint(low=1, high=num_points, size=(batch_size,))
        numv = torch.randint(low=3, high=num_verts, size=(batch_size,))
        numf = torch.randint(low=1, high=num_faces, size=(batch_size,))
        verts_list = []
        faces_list = []
        points_list = []
        for i in range(batch_size):
            # Randomly choose vertices
            verts = torch.rand((numv[i], 3), dtype=torch.float32, device=device)
            verts.requires_grad_(True)

            # Randomly choose faces. Our tests below compare argmin indices
            # over faces and edges. Argmin is sensitive even to small numeral variations
            # thus we make sure that faces are valid
            # i.e. a face f = (i0, i1, i2) s.t. i0 != i1 != i2,
            # otherwise argmin due to numeral sensitivities cannot be resolved
            faces, allf = [], 0
            validf = numv[i].item() - numv[i].item() % 3
            while allf < numf[i]:
                ff = torch.randperm(numv[i], device=device)[:validf].view(-1, 3)
                faces.append(ff)
                allf += ff.shape[0]
            faces = torch.cat(faces, 0)
            if faces.shape[0] > numf[i]:
                faces = faces[: numf[i]]

            verts_list.append(verts)
            faces_list.append(faces)

            # Randomly choose points
            points = torch.rand((nump[i], 3), dtype=torch.float32, device=device)
            points.requires_grad_(True)

            points_list.append(points)

        meshes = Meshes(verts_list, faces_list)
        pcls = Pointclouds(points_list)

        return meshes, pcls

    @staticmethod
    def _point_to_bary(point: torch.Tensor, tri: torch.Tensor) -> torch.Tensor:
        """
        Computes the barycentric coordinates of point wrt triangle (tri)
        Note that point needs to live in the space spanned by tri = (a, b, c),
        i.e. by taking the projection of an arbitrary point on the space spanned by tri

        Args:
            point: FloatTensor of shape (3)
            tri: FloatTensor of shape (3, 3)
        Returns:
            bary: FloatTensor of shape (3)
        """
        assert point.dim() == 1 and point.shape[0] == 3
        assert tri.dim() == 2 and tri.shape[0] == 3 and tri.shape[1] == 3

        a, b, c = tri.unbind(0)

        v0 = b - a
        v1 = c - a
        v2 = point - a

        d00 = v0.dot(v0)
        d01 = v0.dot(v1)
        d11 = v1.dot(v1)
        d20 = v2.dot(v0)
        d21 = v2.dot(v1)

        denom = d00 * d11 - d01 * d01
        s2 = (d11 * d20 - d01 * d21) / denom
        s3 = (d00 * d21 - d01 * d20) / denom
        s1 = 1.0 - s2 - s3

        bary = torch.tensor([s1, s2, s3])
        return bary

    @staticmethod
    def _is_inside_triangle(point: torch.Tensor, tri: torch.Tensor) -> torch.Tensor:
        """
        Computes whether point is inside triangle tri
        Note that point needs to live in the space spanned by tri = (a, b, c)
        i.e. by taking the projection of an arbitrary point on the space spanned by tri

        Args:
            point: FloatTensor of shape (3)
            tri: FloatTensor of shape (3, 3)
        Returns:
            inside: BoolTensor of shape (1)
        """
        bary = TestPointMeshDistance._point_to_bary(point, tri)
        inside = ((bary >= 0.0) * (bary <= 1.0)).all()
        return inside

    @staticmethod
    def _point_to_edge_distance(
        point: torch.Tensor, edge: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the squared euclidean distance of points to edges
        Args:
            point: FloatTensor of shape (3)
            edge: FloatTensor of shape (2, 3)
        Returns:
            dist: FloatTensor of shape (1)

        If a, b are the start and end points of the segments, we
        parametrize a point p as
            x(t) = a + t * (b - a)
        To find t which describes p we minimize (x(t) - p) ^ 2
        Note that p does not need to live in the space spanned by (a, b)
        """
        s0, s1 = edge.unbind(0)

        s01 = s1 - s0
        norm_s01 = s01.dot(s01)

        same_edge = norm_s01 < TestPointMeshDistance.eps()
        if same_edge:
            dist = 0.5 * (point - s0).dot(point - s0) + 0.5 * (point - s1).dot(
                point - s1
            )
            return dist

        t = s01.dot(point - s0) / norm_s01
        t = torch.clamp(t, min=0.0, max=1.0)
        x = s0 + t * s01
        dist = (x - point).dot(x - point)
        return dist

    @staticmethod
    def _point_to_tri_distance(point: torch.Tensor, tri: torch.Tensor) -> torch.Tensor:
        """
        Computes the squared euclidean distance of points to edges
        Args:
            point: FloatTensor of shape (3)
            tri: FloatTensor of shape (3, 3)
        Returns:
            dist: FloatTensor of shape (1)
        """
        a, b, c = tri.unbind(0)
        cross = torch.cross(b - a, c - a)
        norm = cross.norm()
        normal = torch.nn.functional.normalize(cross, dim=0)

        # p0 is the projection of p onto the plane spanned by (a, b, c)
        # p0 = p + tt * normal, s.t. (p0 - a) is orthogonal to normal
        # => tt = dot(a - p, n)
        tt = normal.dot(a) - normal.dot(point)
        p0 = point + tt * normal
        dist_p = tt * tt

        # Compute the distance of p to all edge segments
        e01_dist = TestPointMeshDistance._point_to_edge_distance(point, tri[[0, 1]])
        e02_dist = TestPointMeshDistance._point_to_edge_distance(point, tri[[0, 2]])
        e12_dist = TestPointMeshDistance._point_to_edge_distance(point, tri[[1, 2]])

        with torch.no_grad():
            inside_tri = TestPointMeshDistance._is_inside_triangle(p0, tri)

        if inside_tri and (norm > TestPointMeshDistance.eps()):
            return dist_p
        else:
            if e01_dist.le(e02_dist) and e01_dist.le(e12_dist):
                return e01_dist
            elif e02_dist.le(e01_dist) and e02_dist.le(e12_dist):
                return e02_dist
            else:
                return e12_dist

    def test_point_edge_array_distance(self):
        """
        Test CUDA implementation for PointEdgeArrayDistanceForward
            &  PointEdgeArrayDistanceBackward
        """
        P, E = 16, 32
        device = get_random_cuda_device()
        points = torch.rand((P, 3), dtype=torch.float32, device=device)
        edges = torch.rand((E, 2, 3), dtype=torch.float32, device=device)

        # randomly make some edge points equal
        same = torch.rand((E,), dtype=torch.float32, device=device) > 0.5
        edges[same, 1] = edges[same, 0].clone().detach()

        points.requires_grad = True
        edges.requires_grad = True
        grad_dists = torch.rand((P, E), dtype=torch.float32, device=device)

        # Naive python implementation
        dists_naive = torch.zeros((P, E), dtype=torch.float32, device=device)
        for p in range(P):
            for e in range(E):
                dist = self._point_to_edge_distance(points[p], edges[e])
                dists_naive[p, e] = dist

        # Cuda Forward Implementation
        dists_cuda = _C.point_edge_array_dist_forward(points, edges)

        # Compare
        self.assertClose(dists_naive.cpu(), dists_cuda.cpu())

        # CUDA Bacwkard Implementation
        grad_points_cuda, grad_edges_cuda = _C.point_edge_array_dist_backward(
            points, edges, grad_dists
        )

        dists_naive.backward(grad_dists)
        grad_points_naive = points.grad
        grad_edges_naive = edges.grad

        # Compare
        self.assertClose(grad_points_naive.cpu(), grad_points_cuda.cpu())
        self.assertClose(grad_edges_naive.cpu(), grad_edges_cuda.cpu())

    def test_point_edge_distance(self):
        """
        Test CUDA implementation for PointEdgeDistanceForward
            &  PointEdgeDistanceBackward
        """
        device = get_random_cuda_device()
        N, V, F, P = 4, 32, 16, 24
        meshes, pcls = self.init_meshes_clouds(N, V, F, P, device=device)

        # make points packed a leaf node
        points_packed = pcls.points_packed().detach().clone()  # (P, 3)

        points_first_idx = pcls.cloud_to_packed_first_idx()
        max_p = pcls.num_points_per_cloud().max().item()

        # make edges packed a leaf node
        verts_packed = meshes.verts_packed()
        edges_packed = verts_packed[meshes.edges_packed()]  # (E, 2, 3)
        edges_packed = edges_packed.clone().detach()

        edges_first_idx = meshes.mesh_to_edges_packed_first_idx()

        # leaf nodes
        points_packed.requires_grad = True
        edges_packed.requires_grad = True
        grad_dists = torch.rand(
            (points_packed.shape[0],), dtype=torch.float32, device=device
        )

        # Cuda Implementation: forrward
        dists_cuda, idx_cuda = _C.point_edge_dist_forward(
            points_packed, points_first_idx, edges_packed, edges_first_idx, max_p
        )
        # Cuda Implementation: backward
        grad_points_cuda, grad_edges_cuda = _C.point_edge_dist_backward(
            points_packed, edges_packed, idx_cuda, grad_dists
        )

        # Naive Implementation: forward
        edges_list = packed_to_list(edges_packed, meshes.num_edges_per_mesh().tolist())
        dists_naive = []
        for i in range(N):
            points = pcls.points_list()[i]
            edges = edges_list[i]
            dists_temp = torch.zeros(
                (points.shape[0], edges.shape[0]), dtype=torch.float32, device=device
            )
            for p in range(points.shape[0]):
                for e in range(edges.shape[0]):
                    dist = self._point_to_edge_distance(points[p], edges[e])
                    dists_temp[p, e] = dist
            # torch.min() doesn't necessarily return the first index of the
            # smallest value, our warp_reduce does. So it's not straightforward
            # to directly compare indices, nor the gradients of grad_edges which
            # also depend on the indices of the minimum value.
            # To be able to compare, we will compare dists_temp.min(1) and
            # then feed the cuda indices to the naive output

            start = points_first_idx[i]
            end = points_first_idx[i + 1] if i < N - 1 else points_packed.shape[0]

            min_idx = idx_cuda[start:end] - edges_first_idx[i]
            iidx = torch.arange(points.shape[0], device=device)
            min_dist = dists_temp[iidx, min_idx]

            dists_naive.append(min_dist)

        dists_naive = torch.cat(dists_naive)

        # Compare
        self.assertClose(dists_naive.cpu(), dists_cuda.cpu())

        # Naive Implementation: backward
        dists_naive.backward(grad_dists)
        grad_points_naive = torch.cat([cloud.grad for cloud in pcls.points_list()])
        grad_edges_naive = edges_packed.grad

        # Compare
        self.assertClose(grad_points_naive.cpu(), grad_points_cuda.cpu(), atol=1e-7)
        self.assertClose(grad_edges_naive.cpu(), grad_edges_cuda.cpu(), atol=5e-7)

    def test_edge_point_distance(self):
        """
        Test CUDA implementation for EdgePointDistanceForward
            &  EdgePointDistanceBackward
        """
        device = get_random_cuda_device()
        N, V, F, P = 4, 32, 16, 24
        meshes, pcls = self.init_meshes_clouds(N, V, F, P, device=device)

        # make points packed a leaf node
        points_packed = pcls.points_packed().detach().clone()  # (P, 3)

        points_first_idx = pcls.cloud_to_packed_first_idx()

        # make edges packed a leaf node
        verts_packed = meshes.verts_packed()
        edges_packed = verts_packed[meshes.edges_packed()]  # (E, 2, 3)
        edges_packed = edges_packed.clone().detach()

        edges_first_idx = meshes.mesh_to_edges_packed_first_idx()
        max_e = meshes.num_edges_per_mesh().max().item()

        # leaf nodes
        points_packed.requires_grad = True
        edges_packed.requires_grad = True
        grad_dists = torch.rand(
            (edges_packed.shape[0],), dtype=torch.float32, device=device
        )

        # Cuda Implementation: forward
        dists_cuda, idx_cuda = _C.edge_point_dist_forward(
            points_packed, points_first_idx, edges_packed, edges_first_idx, max_e
        )

        # Cuda Implementation: backward
        grad_points_cuda, grad_edges_cuda = _C.edge_point_dist_backward(
            points_packed, edges_packed, idx_cuda, grad_dists
        )

        # Naive Implementation: forward
        edges_list = packed_to_list(edges_packed, meshes.num_edges_per_mesh().tolist())
        dists_naive = []
        for i in range(N):
            points = pcls.points_list()[i]
            edges = edges_list[i]
            dists_temp = torch.zeros(
                (edges.shape[0], points.shape[0]), dtype=torch.float32, device=device
            )
            for e in range(edges.shape[0]):
                for p in range(points.shape[0]):
                    dist = self._point_to_edge_distance(points[p], edges[e])
                    dists_temp[e, p] = dist

            # torch.min() doesn't necessarily return the first index of the
            # smallest value, our warp_reduce does. So it's not straightforward
            # to directly compare indices, nor the gradients of grad_edges which
            # also depend on the indices of the minimum value.
            # To be able to compare, we will compare dists_temp.min(1) and
            # then feed the cuda indices to the naive output

            start = edges_first_idx[i]
            end = edges_first_idx[i + 1] if i < N - 1 else edges_packed.shape[0]

            min_idx = idx_cuda.cpu()[start:end] - points_first_idx[i]
            iidx = torch.arange(edges.shape[0], device=device)
            min_dist = dists_temp[iidx, min_idx]

            dists_naive.append(min_dist)

        dists_naive = torch.cat(dists_naive)

        # Compare
        self.assertClose(dists_naive.cpu(), dists_cuda.cpu())

        # Naive Implementation: backward
        dists_naive.backward(grad_dists)
        grad_points_naive = torch.cat([cloud.grad for cloud in pcls.points_list()])
        grad_edges_naive = edges_packed.grad

        # Compare
        self.assertClose(grad_points_naive.cpu(), grad_points_cuda.cpu(), atol=1e-7)
        self.assertClose(grad_edges_naive.cpu(), grad_edges_cuda.cpu(), atol=5e-7)

    def test_point_mesh_edge_distance(self):
        """
        Test point_mesh_edge_distance from pytorch3d.loss
        """
        device = get_random_cuda_device()
        N, V, F, P = 4, 32, 16, 24
        meshes, pcls = self.init_meshes_clouds(N, V, F, P, device=device)

        # clone and detach for another backward pass through the op
        verts_op = [verts.clone().detach() for verts in meshes.verts_list()]
        for i in range(N):
            verts_op[i].requires_grad = True

        faces_op = [faces.clone().detach() for faces in meshes.faces_list()]
        meshes_op = Meshes(verts=verts_op, faces=faces_op)
        points_op = [points.clone().detach() for points in pcls.points_list()]
        for i in range(N):
            points_op[i].requires_grad = True
        pcls_op = Pointclouds(points_op)

        # Cuda implementation: forward & backward
        loss_op = point_mesh_edge_distance(meshes_op, pcls_op)

        # Naive implementation: forward & backward
        edges_packed = meshes.edges_packed()
        edges_list = packed_to_list(edges_packed, meshes.num_edges_per_mesh().tolist())
        loss_naive = torch.zeros((N), dtype=torch.float32, device=device)
        for i in range(N):
            points = pcls.points_list()[i]
            verts = meshes.verts_list()[i]
            v_first_idx = meshes.mesh_to_verts_packed_first_idx()[i]
            edges = verts[edges_list[i] - v_first_idx]

            num_p = points.shape[0]
            num_e = edges.shape[0]
            dists = torch.zeros((num_p, num_e), dtype=torch.float32, device=device)
            for p in range(num_p):
                for e in range(num_e):
                    dist = self._point_to_edge_distance(points[p], edges[e])
                    dists[p, e] = dist

            min_dist_p, min_idx_p = dists.min(1)
            min_dist_e, min_idx_e = dists.min(0)

            loss_naive[i] = min_dist_p.mean() + min_dist_e.mean()
        loss_naive = loss_naive.mean()

        # NOTE that hear the comparison holds despite the discrepancy
        # due to the argmin indices returned by min(). This is because
        # we don't will compare gradients on the verts and not on the
        # edges or faces.

        # Compare forward pass
        self.assertClose(loss_op, loss_naive)

        # Compare backward pass
        rand_val = torch.rand((1)).item()
        grad_dist = torch.tensor(rand_val, dtype=torch.float32, device=device)

        loss_naive.backward(grad_dist)
        loss_op.backward(grad_dist)

        # check verts grad
        for i in range(N):
            self.assertClose(
                meshes.verts_list()[i].grad, meshes_op.verts_list()[i].grad
            )
            self.assertClose(pcls.points_list()[i].grad, pcls_op.points_list()[i].grad)

    def test_point_face_array_distance(self):
        """
        Test CUDA implementation for PointFaceArrayDistanceForward
            &  PointFaceArrayDistanceBackward
        """
        P, T = 16, 32
        device = get_random_cuda_device()
        points = torch.rand((P, 3), dtype=torch.float32, device=device)
        tris = torch.rand((T, 3, 3), dtype=torch.float32, device=device)

        points.requires_grad = True
        tris.requires_grad = True
        grad_dists = torch.rand((P, T), dtype=torch.float32, device=device)

        points_temp = points.clone().detach()
        points_temp.requires_grad = True
        tris_temp = tris.clone().detach()
        tris_temp.requires_grad = True

        # Naive python implementation
        dists_naive = torch.zeros((P, T), dtype=torch.float32, device=device)
        for p in range(P):
            for t in range(T):
                dist = self._point_to_tri_distance(points[p], tris[t])
                dists_naive[p, t] = dist

        # Naive Backward
        dists_naive.backward(grad_dists)
        grad_points_naive = points.grad
        grad_tris_naive = tris.grad

        # Cuda Forward Implementation
        dists_cuda = _C.point_face_array_dist_forward(points, tris)

        # Compare
        self.assertClose(dists_naive.cpu(), dists_cuda.cpu())

        # CUDA Backward Implementation
        grad_points_cuda, grad_tris_cuda = _C.point_face_array_dist_backward(
            points, tris, grad_dists
        )

        # Compare
        self.assertClose(grad_points_naive.cpu(), grad_points_cuda.cpu())
        self.assertClose(grad_tris_naive.cpu(), grad_tris_cuda.cpu(), atol=5e-6)

    def test_point_face_distance(self):
        """
        Test CUDA implementation for PointFaceDistanceForward
            &  PointFaceDistanceBackward
        """
        device = get_random_cuda_device()
        N, V, F, P = 4, 32, 16, 24
        meshes, pcls = self.init_meshes_clouds(N, V, F, P, device=device)

        # make points packed a leaf node
        points_packed = pcls.points_packed().detach().clone()  # (P, 3)

        points_first_idx = pcls.cloud_to_packed_first_idx()
        max_p = pcls.num_points_per_cloud().max().item()

        # make edges packed a leaf node
        verts_packed = meshes.verts_packed()
        faces_packed = verts_packed[meshes.faces_packed()]  # (T, 3, 3)
        faces_packed = faces_packed.clone().detach()

        faces_first_idx = meshes.mesh_to_faces_packed_first_idx()

        # leaf nodes
        points_packed.requires_grad = True
        faces_packed.requires_grad = True
        grad_dists = torch.rand(
            (points_packed.shape[0],), dtype=torch.float32, device=device
        )

        # Cuda Implementation: forward
        dists_cuda, idx_cuda = _C.point_face_dist_forward(
            points_packed, points_first_idx, faces_packed, faces_first_idx, max_p
        )

        # Cuda Implementation: backward
        grad_points_cuda, grad_faces_cuda = _C.point_face_dist_backward(
            points_packed, faces_packed, idx_cuda, grad_dists
        )

        # Naive Implementation: forward
        faces_list = packed_to_list(faces_packed, meshes.num_faces_per_mesh().tolist())
        dists_naive = []
        for i in range(N):
            points = pcls.points_list()[i]
            tris = faces_list[i]
            dists_temp = torch.zeros(
                (points.shape[0], tris.shape[0]), dtype=torch.float32, device=device
            )
            for p in range(points.shape[0]):
                for t in range(tris.shape[0]):
                    dist = self._point_to_tri_distance(points[p], tris[t])
                    dists_temp[p, t] = dist

            # torch.min() doesn't necessarily return the first index of the
            # smallest value, our warp_reduce does. So it's not straightforward
            # to directly compare indices, nor the gradients of grad_tris which
            # also depend on the indices of the minimum value.
            # To be able to compare, we will compare dists_temp.min(1) and
            # then feed the cuda indices to the naive output

            start = points_first_idx[i]
            end = points_first_idx[i + 1] if i < N - 1 else points_packed.shape[0]

            min_idx = idx_cuda.cpu()[start:end] - faces_first_idx[i]
            iidx = torch.arange(points.shape[0], device=device)
            min_dist = dists_temp[iidx, min_idx]

            dists_naive.append(min_dist)

        dists_naive = torch.cat(dists_naive)

        # Compare
        self.assertClose(dists_naive.cpu(), dists_cuda.cpu())

        #  Naive Implementation: backward
        dists_naive.backward(grad_dists)
        grad_points_naive = torch.cat([cloud.grad for cloud in pcls.points_list()])
        grad_faces_naive = faces_packed.grad

        # Compare
        self.assertClose(grad_points_naive.cpu(), grad_points_cuda.cpu(), atol=1e-7)
        self.assertClose(grad_faces_naive.cpu(), grad_faces_cuda.cpu(), atol=5e-7)

    def test_face_point_distance(self):
        """
        Test CUDA implementation for FacePointDistanceForward
            &  FacePointDistanceBackward
        """
        device = get_random_cuda_device()
        N, V, F, P = 4, 32, 16, 24
        meshes, pcls = self.init_meshes_clouds(N, V, F, P, device=device)

        # make points packed a leaf node
        points_packed = pcls.points_packed().detach().clone()  # (P, 3)

        points_first_idx = pcls.cloud_to_packed_first_idx()

        # make edges packed a leaf node
        verts_packed = meshes.verts_packed()
        faces_packed = verts_packed[meshes.faces_packed()]  # (T, 3, 3)
        faces_packed = faces_packed.clone().detach()

        faces_first_idx = meshes.mesh_to_faces_packed_first_idx()
        max_f = meshes.num_faces_per_mesh().max().item()

        # leaf nodes
        points_packed.requires_grad = True
        faces_packed.requires_grad = True
        grad_dists = torch.rand(
            (faces_packed.shape[0],), dtype=torch.float32, device=device
        )

        # Cuda Implementation: forward
        dists_cuda, idx_cuda = _C.face_point_dist_forward(
            points_packed, points_first_idx, faces_packed, faces_first_idx, max_f
        )

        # Cuda Implementation: backward
        grad_points_cuda, grad_faces_cuda = _C.face_point_dist_backward(
            points_packed, faces_packed, idx_cuda, grad_dists
        )

        # Naive Implementation: forward
        faces_list = packed_to_list(faces_packed, meshes.num_faces_per_mesh().tolist())
        dists_naive = []
        for i in range(N):
            points = pcls.points_list()[i]
            tris = faces_list[i]
            dists_temp = torch.zeros(
                (tris.shape[0], points.shape[0]), dtype=torch.float32, device=device
            )
            for t in range(tris.shape[0]):
                for p in range(points.shape[0]):
                    dist = self._point_to_tri_distance(points[p], tris[t])
                    dists_temp[t, p] = dist

            # torch.min() doesn't necessarily return the first index of the
            # smallest value, our warp_reduce does. So it's not straightforward
            # to directly compare indices, nor the gradients of grad_tris which
            # also depend on the indices of the minimum value.
            # To be able to compare, we will compare dists_temp.min(1) and
            # then feed the cuda indices to the naive output

            start = faces_first_idx[i]
            end = faces_first_idx[i + 1] if i < N - 1 else faces_packed.shape[0]

            min_idx = idx_cuda.cpu()[start:end] - points_first_idx[i]
            iidx = torch.arange(tris.shape[0], device=device)
            min_dist = dists_temp[iidx, min_idx]

            dists_naive.append(min_dist)

        dists_naive = torch.cat(dists_naive)

        # Compare
        self.assertClose(dists_naive.cpu(), dists_cuda.cpu())

        # Naive Implementation: backward
        dists_naive.backward(grad_dists)
        grad_points_naive = torch.cat([cloud.grad for cloud in pcls.points_list()])
        grad_faces_naive = faces_packed.grad

        # Compare
        self.assertClose(grad_points_naive.cpu(), grad_points_cuda.cpu(), atol=1e-7)
        self.assertClose(grad_faces_naive.cpu(), grad_faces_cuda.cpu(), atol=5e-7)

    def test_point_mesh_face_distance(self):
        """
        Test point_mesh_face_distance from pytorch3d.loss
        """
        device = get_random_cuda_device()
        N, V, F, P = 4, 32, 16, 24
        meshes, pcls = self.init_meshes_clouds(N, V, F, P, device=device)

        # clone and detach for another backward pass through the op
        verts_op = [verts.clone().detach() for verts in meshes.verts_list()]
        for i in range(N):
            verts_op[i].requires_grad = True

        faces_op = [faces.clone().detach() for faces in meshes.faces_list()]
        meshes_op = Meshes(verts=verts_op, faces=faces_op)
        points_op = [points.clone().detach() for points in pcls.points_list()]
        for i in range(N):
            points_op[i].requires_grad = True
        pcls_op = Pointclouds(points_op)

        # naive implementation
        loss_naive = torch.zeros((N), dtype=torch.float32, device=device)
        for i in range(N):
            points = pcls.points_list()[i]
            verts = meshes.verts_list()[i]
            faces = meshes.faces_list()[i]
            tris = verts[faces]

            num_p = points.shape[0]
            num_t = tris.shape[0]
            dists = torch.zeros((num_p, num_t), dtype=torch.float32, device=device)
            for p in range(num_p):
                for t in range(num_t):
                    dist = self._point_to_tri_distance(points[p], tris[t])
                    dists[p, t] = dist

            min_dist_p, min_idx_p = dists.min(1)
            min_dist_t, min_idx_t = dists.min(0)

            loss_naive[i] = min_dist_p.mean() + min_dist_t.mean()
        loss_naive = loss_naive.mean()

        # Op
        loss_op = point_mesh_face_distance(meshes_op, pcls_op)

        # Compare forward pass
        self.assertClose(loss_op, loss_naive)

        # Compare backward pass
        rand_val = torch.rand((1)).item()
        grad_dist = torch.tensor(rand_val, dtype=torch.float32, device=device)

        loss_naive.backward(grad_dist)
        loss_op.backward(grad_dist)

        # check verts grad
        for i in range(N):
            self.assertClose(
                meshes.verts_list()[i].grad, meshes_op.verts_list()[i].grad
            )
            self.assertClose(pcls.points_list()[i].grad, pcls_op.points_list()[i].grad)

    @staticmethod
    def point_mesh_edge(N: int, V: int, F: int, P: int, device: str):
        device = torch.device(device)
        meshes, pcls = TestPointMeshDistance.init_meshes_clouds(
            N, V, F, P, device=device
        )
        torch.cuda.synchronize()

        def loss():
            point_mesh_edge_distance(meshes, pcls)
            torch.cuda.synchronize()

        return loss

    @staticmethod
    def point_mesh_face(N: int, V: int, F: int, P: int, device: str):
        device = torch.device(device)
        meshes, pcls = TestPointMeshDistance.init_meshes_clouds(
            N, V, F, P, device=device
        )
        torch.cuda.synchronize()

        def loss():
            point_mesh_face_distance(meshes, pcls)
            torch.cuda.synchronize()

        return loss
