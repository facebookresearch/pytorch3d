# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest

import numpy as np
import torch
from common_testing import TestCaseMixin
from pytorch3d.structures.meshes import Meshes


class TestMeshes(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)
        torch.manual_seed(42)

    @staticmethod
    def init_mesh(
        num_meshes: int = 10,
        max_v: int = 100,
        max_f: int = 300,
        lists_to_tensors: bool = False,
        device: str = "cpu",
    ):
        """
        Function to generate a Meshes object of N meshes with
        random numbers of vertices and faces.

        Args:
            num_meshes: Number of meshes to generate.
            max_v: Max number of vertices per mesh.
            max_f: Max number of faces per mesh.
            lists_to_tensors: Determines whether the generated meshes should be
                              constructed from lists (=False) or
                              a tensor (=True) of faces/verts.

        Returns:
            Meshes object.
        """
        device = torch.device(device)

        verts_list = []
        faces_list = []

        # Randomly generate numbers of faces and vertices in each mesh.
        if lists_to_tensors:
            # If we define faces/verts with tensors, f/v has to be the
            # same for each mesh in the batch.
            f = torch.randint(max_f, size=(1,), dtype=torch.int32)
            v = torch.randint(3, high=max_v, size=(1,), dtype=torch.int32)
            f = f.repeat(num_meshes)
            v = v.repeat(num_meshes)
        else:
            # For lists of faces and vertices, we can sample different v/f
            # per mesh.
            f = torch.randint(max_f, size=(num_meshes,), dtype=torch.int32)
            v = torch.randint(3, high=max_v, size=(num_meshes,), dtype=torch.int32)

        # Generate the actual vertices and faces.
        for i in range(num_meshes):
            verts = torch.rand((v[i], 3), dtype=torch.float32, device=device)
            faces = torch.randint(
                v[i], size=(f[i], 3), dtype=torch.int64, device=device
            )
            verts_list.append(verts)
            faces_list.append(faces)

        if lists_to_tensors:
            verts_list = torch.stack(verts_list)
            faces_list = torch.stack(faces_list)

        return Meshes(verts=verts_list, faces=faces_list)

    @staticmethod
    def init_simple_mesh(device: str = "cpu"):
        """
        Returns a Meshes data structure of simple mesh examples.

        Returns:
            Meshes object.
        """
        device = torch.device(device)

        verts = [
            torch.tensor(
                [[0.1, 0.3, 0.5], [0.5, 0.2, 0.1], [0.6, 0.8, 0.7]],
                dtype=torch.float32,
                device=device,
            ),
            torch.tensor(
                [[0.1, 0.3, 0.3], [0.6, 0.7, 0.8], [0.2, 0.3, 0.4], [0.1, 0.5, 0.3]],
                dtype=torch.float32,
                device=device,
            ),
            torch.tensor(
                [
                    [0.7, 0.3, 0.6],
                    [0.2, 0.4, 0.8],
                    [0.9, 0.5, 0.2],
                    [0.2, 0.3, 0.4],
                    [0.9, 0.3, 0.8],
                ],
                dtype=torch.float32,
                device=device,
            ),
        ]
        faces = [
            torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device),
            torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int64, device=device),
            torch.tensor(
                [
                    [1, 2, 0],
                    [0, 1, 3],
                    [2, 3, 1],
                    [4, 3, 2],
                    [4, 0, 1],
                    [4, 3, 1],
                    [4, 2, 1],
                ],
                dtype=torch.int64,
                device=device,
            ),
        ]
        return Meshes(verts=verts, faces=faces)

    def test_simple(self):
        mesh = TestMeshes.init_simple_mesh("cuda:0")

        # Check that faces/verts per mesh are set in init:
        self.assertClose(mesh._num_faces_per_mesh.cpu(), torch.tensor([1, 2, 7]))
        self.assertClose(mesh._num_verts_per_mesh.cpu(), torch.tensor([3, 4, 5]))

        # Check computed tensors
        self.assertClose(
            mesh.verts_packed_to_mesh_idx().cpu(),
            torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2]),
        )
        self.assertClose(
            mesh.mesh_to_verts_packed_first_idx().cpu(), torch.tensor([0, 3, 7])
        )
        self.assertClose(
            mesh.verts_padded_to_packed_idx().cpu(),
            torch.tensor([0, 1, 2, 5, 6, 7, 8, 10, 11, 12, 13, 14]),
        )
        self.assertClose(
            mesh.faces_packed_to_mesh_idx().cpu(),
            torch.tensor([0, 1, 1, 2, 2, 2, 2, 2, 2, 2]),
        )
        self.assertClose(
            mesh.mesh_to_faces_packed_first_idx().cpu(), torch.tensor([0, 1, 3])
        )
        self.assertClose(
            mesh.num_edges_per_mesh().cpu(), torch.tensor([3, 5, 10], dtype=torch.int32)
        )
        self.assertClose(
            mesh.mesh_to_edges_packed_first_idx().cpu(),
            torch.tensor([0, 3, 8], dtype=torch.int64),
        )

    def test_simple_random_meshes(self):

        # Define the test mesh object either as a list or tensor of faces/verts.
        for lists_to_tensors in (False, True):
            N = 10
            mesh = TestMeshes.init_mesh(N, 100, 300, lists_to_tensors=lists_to_tensors)
            verts_list = mesh.verts_list()
            faces_list = mesh.faces_list()

            # Check batch calculations.
            verts_padded = mesh.verts_padded()
            faces_padded = mesh.faces_padded()
            verts_per_mesh = mesh.num_verts_per_mesh()
            faces_per_mesh = mesh.num_faces_per_mesh()
            for n in range(N):
                v = verts_list[n].shape[0]
                f = faces_list[n].shape[0]
                self.assertClose(verts_padded[n, :v, :], verts_list[n])
                if verts_padded.shape[1] > v:
                    self.assertTrue(verts_padded[n, v:, :].eq(0).all())
                self.assertClose(faces_padded[n, :f, :], faces_list[n])
                if faces_padded.shape[1] > f:
                    self.assertTrue(faces_padded[n, f:, :].eq(-1).all())
                self.assertEqual(verts_per_mesh[n], v)
                self.assertEqual(faces_per_mesh[n], f)

            # Check compute packed.
            verts_packed = mesh.verts_packed()
            vert_to_mesh = mesh.verts_packed_to_mesh_idx()
            mesh_to_vert = mesh.mesh_to_verts_packed_first_idx()
            faces_packed = mesh.faces_packed()
            face_to_mesh = mesh.faces_packed_to_mesh_idx()
            mesh_to_face = mesh.mesh_to_faces_packed_first_idx()

            curv, curf = 0, 0
            for n in range(N):
                v = verts_list[n].shape[0]
                f = faces_list[n].shape[0]
                self.assertClose(verts_packed[curv : curv + v, :], verts_list[n])
                self.assertClose(faces_packed[curf : curf + f, :] - curv, faces_list[n])
                self.assertTrue(vert_to_mesh[curv : curv + v].eq(n).all())
                self.assertTrue(face_to_mesh[curf : curf + f].eq(n).all())
                self.assertTrue(mesh_to_vert[n] == curv)
                self.assertTrue(mesh_to_face[n] == curf)
                curv += v
                curf += f

            # Check compute edges and compare with numpy unique.
            edges = mesh.edges_packed().cpu().numpy()
            edge_to_mesh_idx = mesh.edges_packed_to_mesh_idx().cpu().numpy()
            num_edges_per_mesh = mesh.num_edges_per_mesh().cpu().numpy()

            npfaces_packed = mesh.faces_packed().cpu().numpy()
            e01 = npfaces_packed[:, [0, 1]]
            e12 = npfaces_packed[:, [1, 2]]
            e20 = npfaces_packed[:, [2, 0]]
            npedges = np.concatenate((e12, e20, e01), axis=0)
            npedges = np.sort(npedges, axis=1)

            unique_edges, unique_idx = np.unique(npedges, return_index=True, axis=0)
            self.assertTrue(np.allclose(edges, unique_edges))
            temp = face_to_mesh.cpu().numpy()
            temp = np.concatenate((temp, temp, temp), axis=0)
            edge_to_mesh = temp[unique_idx]
            self.assertTrue(np.allclose(edge_to_mesh_idx, edge_to_mesh))
            num_edges = np.bincount(edge_to_mesh, minlength=N)
            self.assertTrue(np.allclose(num_edges_per_mesh, num_edges))
            mesh_to_edges_packed_first_idx = (
                mesh.mesh_to_edges_packed_first_idx().cpu().numpy()
            )
            self.assertTrue(
                np.allclose(mesh_to_edges_packed_first_idx[1:], num_edges.cumsum()[:-1])
            )
            self.assertTrue(mesh_to_edges_packed_first_idx[0] == 0)

    def test_allempty(self):
        verts_list = []
        faces_list = []
        mesh = Meshes(verts=verts_list, faces=faces_list)
        self.assertEqual(len(mesh), 0)
        self.assertEqual(mesh.verts_padded().shape[0], 0)
        self.assertEqual(mesh.faces_padded().shape[0], 0)
        self.assertEqual(mesh.verts_packed().shape[0], 0)
        self.assertEqual(mesh.faces_packed().shape[0], 0)
        self.assertEqual(mesh.num_faces_per_mesh().shape[0], 0)
        self.assertEqual(mesh.num_verts_per_mesh().shape[0], 0)

    def test_empty(self):
        N, V, F = 10, 100, 300
        device = torch.device("cuda:0")
        verts_list = []
        faces_list = []
        valid = torch.randint(2, size=(N,), dtype=torch.uint8, device=device)
        for n in range(N):
            if valid[n]:
                v = torch.randint(
                    3, high=V, size=(1,), dtype=torch.int32, device=device
                )[0]
                f = torch.randint(F, size=(1,), dtype=torch.int32, device=device)[0]
                verts = torch.rand((v, 3), dtype=torch.float32, device=device)
                faces = torch.randint(v, size=(f, 3), dtype=torch.int64, device=device)
            else:
                verts = torch.tensor([], dtype=torch.float32, device=device)
                faces = torch.tensor([], dtype=torch.int64, device=device)
            verts_list.append(verts)
            faces_list.append(faces)

        mesh = Meshes(verts=verts_list, faces=faces_list)
        verts_padded = mesh.verts_padded()
        faces_padded = mesh.faces_padded()
        verts_per_mesh = mesh.num_verts_per_mesh()
        faces_per_mesh = mesh.num_faces_per_mesh()
        for n in range(N):
            v = len(verts_list[n])
            f = len(faces_list[n])
            if v > 0:
                self.assertClose(verts_padded[n, :v, :], verts_list[n])
                if verts_padded.shape[1] > v:
                    self.assertTrue(verts_padded[n, v:, :].eq(0).all())
            if f > 0:
                self.assertClose(faces_padded[n, :f, :], faces_list[n])
                if faces_padded.shape[1] > f:
                    self.assertTrue(faces_padded[n, f:, :].eq(-1).all())
            self.assertTrue(verts_per_mesh[n] == v)
            self.assertTrue(faces_per_mesh[n] == f)

    def test_padding(self):
        N, V, F = 10, 100, 300
        device = torch.device("cuda:0")
        verts, faces = [], []
        valid = torch.randint(2, size=(N,), dtype=torch.uint8, device=device)
        num_verts, num_faces = (
            torch.zeros(N, dtype=torch.int32),
            torch.zeros(N, dtype=torch.int32),
        )
        for n in range(N):
            verts.append(torch.rand((V, 3), dtype=torch.float32, device=device))
            this_faces = torch.full((F, 3), -1, dtype=torch.int64, device=device)
            if valid[n]:
                v = torch.randint(
                    3, high=V, size=(1,), dtype=torch.int32, device=device
                )[0]
                f = torch.randint(F, size=(1,), dtype=torch.int32, device=device)[0]
                this_faces[:f, :] = torch.randint(
                    v, size=(f, 3), dtype=torch.int64, device=device
                )
                num_verts[n] = v
                num_faces[n] = f
            faces.append(this_faces)

        mesh = Meshes(verts=torch.stack(verts), faces=torch.stack(faces))

        # Check verts/faces per mesh are set correctly in init.
        self.assertListEqual(mesh._num_faces_per_mesh.tolist(), num_faces.tolist())
        self.assertListEqual(mesh._num_verts_per_mesh.tolist(), [V] * N)

        for n, (vv, ff) in enumerate(zip(mesh.verts_list(), mesh.faces_list())):
            self.assertClose(ff, faces[n][: num_faces[n]])
            self.assertClose(vv, verts[n])

        new_faces = [ff.clone() for ff in faces]
        v = torch.randint(3, high=V, size=(1,), dtype=torch.int32, device=device)[0]
        f = torch.randint(F - 10, size=(1,), dtype=torch.int32, device=device)[0]
        this_faces = torch.full((F, 3), -1, dtype=torch.int64, device=device)
        this_faces[10 : f + 10, :] = torch.randint(
            v, size=(f, 3), dtype=torch.int64, device=device
        )
        new_faces[3] = this_faces

        with self.assertRaisesRegex(ValueError, "Padding of faces"):
            Meshes(verts=torch.stack(verts), faces=torch.stack(new_faces))

    def test_clone(self):
        N = 5
        mesh = TestMeshes.init_mesh(N, 10, 100)
        for force in [0, 1]:
            if force:
                # force mesh to have computed attributes
                mesh.verts_packed()
                mesh.edges_packed()
                mesh.verts_padded()

            new_mesh = mesh.clone()

            # Modify tensors in both meshes.
            new_mesh._verts_list[0] = new_mesh._verts_list[0] * 5

            # Check cloned and original Meshes objects do not share tensors.
            self.assertFalse(
                torch.allclose(new_mesh._verts_list[0], mesh._verts_list[0])
            )
            self.assertSeparate(new_mesh.verts_packed(), mesh.verts_packed())
            self.assertSeparate(new_mesh.verts_padded(), mesh.verts_padded())
            self.assertSeparate(new_mesh.faces_packed(), mesh.faces_packed())
            self.assertSeparate(new_mesh.faces_padded(), mesh.faces_padded())
            self.assertSeparate(new_mesh.edges_packed(), mesh.edges_packed())

    def test_laplacian_packed(self):
        def naive_laplacian_packed(meshes):
            verts_packed = meshes.verts_packed()
            edges_packed = meshes.edges_packed()
            V = verts_packed.shape[0]

            L = torch.zeros((V, V), dtype=torch.float32, device=meshes.device)
            for e in edges_packed:
                L[e[0], e[1]] = 1
                # symetric
                L[e[1], e[0]] = 1

            deg = L.sum(1).view(-1, 1)
            deg[deg > 0] = 1.0 / deg[deg > 0]
            L = L * deg
            diag = torch.eye(V, dtype=torch.float32, device=meshes.device)
            L.masked_fill_(diag > 0, -1)
            return L

        # Note that we don't test with random meshes for this case, as the
        # definition of Laplacian is defined for simple graphs (aka valid meshes)
        meshes = TestMeshes.init_simple_mesh("cuda:0")

        lapl_naive = naive_laplacian_packed(meshes)
        lapl = meshes.laplacian_packed().to_dense()
        # check with naive
        self.assertClose(lapl, lapl_naive)

    def test_offset_verts(self):
        def naive_offset_verts(mesh, vert_offsets_packed):
            # new Meshes class
            new_verts_packed = mesh.verts_packed() + vert_offsets_packed
            new_verts_list = list(
                new_verts_packed.split(mesh.num_verts_per_mesh().tolist(), 0)
            )
            new_faces_list = [f.clone() for f in mesh.faces_list()]
            return Meshes(verts=new_verts_list, faces=new_faces_list)

        N = 5
        mesh = TestMeshes.init_mesh(N, 10, 100)
        all_v = mesh.verts_packed().size(0)
        verts_per_mesh = mesh.num_verts_per_mesh()
        for force in [0, 1]:
            if force:
                # force mesh to have computed attributes
                mesh._compute_packed(refresh=True)
                mesh._compute_padded()
                mesh._compute_edges_packed()
                mesh.verts_padded_to_packed_idx()
                mesh._compute_face_areas_normals(refresh=True)
                mesh._compute_vertex_normals(refresh=True)

            deform = torch.rand((all_v, 3), dtype=torch.float32, device=mesh.device)
            # new meshes class to hold the deformed mesh
            new_mesh_naive = naive_offset_verts(mesh, deform)

            new_mesh = mesh.offset_verts(deform)

            # check verts_list & faces_list
            verts_cumsum = torch.cumsum(verts_per_mesh, 0).tolist()
            verts_cumsum.insert(0, 0)
            for i in range(N):
                self.assertClose(
                    new_mesh.verts_list()[i],
                    mesh.verts_list()[i]
                    + deform[verts_cumsum[i] : verts_cumsum[i + 1]],
                )
                self.assertClose(
                    new_mesh.verts_list()[i], new_mesh_naive.verts_list()[i]
                )
                self.assertClose(mesh.faces_list()[i], new_mesh_naive.faces_list()[i])
                self.assertClose(
                    new_mesh.faces_list()[i], new_mesh_naive.faces_list()[i]
                )
                # check faces and vertex normals
                self.assertClose(
                    new_mesh.verts_normals_list()[i],
                    new_mesh_naive.verts_normals_list()[i],
                )
                self.assertClose(
                    new_mesh.faces_normals_list()[i],
                    new_mesh_naive.faces_normals_list()[i],
                )

            # check padded & packed
            self.assertClose(new_mesh.faces_padded(), new_mesh_naive.faces_padded())
            self.assertClose(new_mesh.verts_padded(), new_mesh_naive.verts_padded())
            self.assertClose(new_mesh.faces_packed(), new_mesh_naive.faces_packed())
            self.assertClose(new_mesh.verts_packed(), new_mesh_naive.verts_packed())
            self.assertClose(new_mesh.edges_packed(), new_mesh_naive.edges_packed())
            self.assertClose(
                new_mesh.verts_packed_to_mesh_idx(),
                new_mesh_naive.verts_packed_to_mesh_idx(),
            )
            self.assertClose(
                new_mesh.mesh_to_verts_packed_first_idx(),
                new_mesh_naive.mesh_to_verts_packed_first_idx(),
            )
            self.assertClose(
                new_mesh.num_verts_per_mesh(), new_mesh_naive.num_verts_per_mesh()
            )
            self.assertClose(
                new_mesh.faces_packed_to_mesh_idx(),
                new_mesh_naive.faces_packed_to_mesh_idx(),
            )
            self.assertClose(
                new_mesh.mesh_to_faces_packed_first_idx(),
                new_mesh_naive.mesh_to_faces_packed_first_idx(),
            )
            self.assertClose(
                new_mesh.num_faces_per_mesh(), new_mesh_naive.num_faces_per_mesh()
            )
            self.assertClose(
                new_mesh.edges_packed_to_mesh_idx(),
                new_mesh_naive.edges_packed_to_mesh_idx(),
            )
            self.assertClose(
                new_mesh.verts_padded_to_packed_idx(),
                new_mesh_naive.verts_padded_to_packed_idx(),
            )
            self.assertTrue(all(new_mesh.valid == new_mesh_naive.valid))
            self.assertTrue(new_mesh.equisized == new_mesh_naive.equisized)

            # check face areas, normals and vertex normals
            self.assertClose(
                new_mesh.verts_normals_packed(), new_mesh_naive.verts_normals_packed()
            )
            self.assertClose(
                new_mesh.verts_normals_padded(), new_mesh_naive.verts_normals_padded()
            )
            self.assertClose(
                new_mesh.faces_normals_packed(), new_mesh_naive.faces_normals_packed()
            )
            self.assertClose(
                new_mesh.faces_normals_padded(), new_mesh_naive.faces_normals_padded()
            )
            self.assertClose(
                new_mesh.faces_areas_packed(), new_mesh_naive.faces_areas_packed()
            )
            self.assertClose(
                new_mesh.mesh_to_edges_packed_first_idx(),
                new_mesh_naive.mesh_to_edges_packed_first_idx(),
            )

    def test_scale_verts(self):
        def naive_scale_verts(mesh, scale):
            if not torch.is_tensor(scale):
                scale = torch.ones(len(mesh)).mul_(scale)
            # new Meshes class
            new_verts_list = [
                scale[i] * v.clone() for (i, v) in enumerate(mesh.verts_list())
            ]
            new_faces_list = [f.clone() for f in mesh.faces_list()]
            return Meshes(verts=new_verts_list, faces=new_faces_list)

        N = 5
        for test in ["tensor", "scalar"]:
            mesh = TestMeshes.init_mesh(N, 10, 100)
            for force in [0, 1]:
                if force:
                    # force mesh to have computed attributes
                    mesh.verts_packed()
                    mesh.edges_packed()
                    mesh.verts_padded()
                    mesh._compute_face_areas_normals(refresh=True)
                    mesh._compute_vertex_normals(refresh=True)

                if test == "tensor":
                    scales = torch.rand(N)
                elif test == "scalar":
                    scales = torch.rand(1)[0].item()
                new_mesh_naive = naive_scale_verts(mesh, scales)
                new_mesh = mesh.scale_verts(scales)
                for i in range(N):
                    if test == "tensor":
                        self.assertClose(
                            scales[i] * mesh.verts_list()[i], new_mesh.verts_list()[i]
                        )
                    else:
                        self.assertClose(
                            scales * mesh.verts_list()[i], new_mesh.verts_list()[i]
                        )
                    self.assertClose(
                        new_mesh.verts_list()[i], new_mesh_naive.verts_list()[i]
                    )
                    self.assertClose(
                        mesh.faces_list()[i], new_mesh_naive.faces_list()[i]
                    )
                    self.assertClose(
                        new_mesh.faces_list()[i], new_mesh_naive.faces_list()[i]
                    )
                    # check face and vertex normals
                    self.assertClose(
                        new_mesh.verts_normals_list()[i],
                        new_mesh_naive.verts_normals_list()[i],
                    )
                    self.assertClose(
                        new_mesh.faces_normals_list()[i],
                        new_mesh_naive.faces_normals_list()[i],
                    )

                # check padded & packed
                self.assertClose(new_mesh.faces_padded(), new_mesh_naive.faces_padded())
                self.assertClose(new_mesh.verts_padded(), new_mesh_naive.verts_padded())
                self.assertClose(new_mesh.faces_packed(), new_mesh_naive.faces_packed())
                self.assertClose(new_mesh.verts_packed(), new_mesh_naive.verts_packed())
                self.assertClose(new_mesh.edges_packed(), new_mesh_naive.edges_packed())
                self.assertClose(
                    new_mesh.verts_packed_to_mesh_idx(),
                    new_mesh_naive.verts_packed_to_mesh_idx(),
                )
                self.assertClose(
                    new_mesh.mesh_to_verts_packed_first_idx(),
                    new_mesh_naive.mesh_to_verts_packed_first_idx(),
                )
                self.assertClose(
                    new_mesh.num_verts_per_mesh(), new_mesh_naive.num_verts_per_mesh()
                )
                self.assertClose(
                    new_mesh.faces_packed_to_mesh_idx(),
                    new_mesh_naive.faces_packed_to_mesh_idx(),
                )
                self.assertClose(
                    new_mesh.mesh_to_faces_packed_first_idx(),
                    new_mesh_naive.mesh_to_faces_packed_first_idx(),
                )
                self.assertClose(
                    new_mesh.num_faces_per_mesh(), new_mesh_naive.num_faces_per_mesh()
                )
                self.assertClose(
                    new_mesh.edges_packed_to_mesh_idx(),
                    new_mesh_naive.edges_packed_to_mesh_idx(),
                )
                self.assertClose(
                    new_mesh.verts_padded_to_packed_idx(),
                    new_mesh_naive.verts_padded_to_packed_idx(),
                )
                self.assertTrue(all(new_mesh.valid == new_mesh_naive.valid))
                self.assertTrue(new_mesh.equisized == new_mesh_naive.equisized)

                # check face areas, normals and vertex normals
                self.assertClose(
                    new_mesh.verts_normals_packed(),
                    new_mesh_naive.verts_normals_packed(),
                )
                self.assertClose(
                    new_mesh.verts_normals_padded(),
                    new_mesh_naive.verts_normals_padded(),
                )
                self.assertClose(
                    new_mesh.faces_normals_packed(),
                    new_mesh_naive.faces_normals_packed(),
                )
                self.assertClose(
                    new_mesh.faces_normals_padded(),
                    new_mesh_naive.faces_normals_padded(),
                )
                self.assertClose(
                    new_mesh.faces_areas_packed(), new_mesh_naive.faces_areas_packed()
                )
                self.assertClose(
                    new_mesh.mesh_to_edges_packed_first_idx(),
                    new_mesh_naive.mesh_to_edges_packed_first_idx(),
                )

    def test_extend_list(self):
        N = 10
        mesh = TestMeshes.init_mesh(5, 10, 100)
        for force in [0, 1]:
            if force:
                # force some computes to happen
                mesh._compute_packed(refresh=True)
                mesh._compute_padded()
                mesh._compute_edges_packed()
                mesh.verts_padded_to_packed_idx()
            new_mesh = mesh.extend(N)
            self.assertEqual(len(mesh) * 10, len(new_mesh))
            for i in range(len(mesh)):
                for n in range(N):
                    self.assertClose(
                        mesh.verts_list()[i], new_mesh.verts_list()[i * N + n]
                    )
                    self.assertClose(
                        mesh.faces_list()[i], new_mesh.faces_list()[i * N + n]
                    )
                    self.assertTrue(mesh.valid[i] == new_mesh.valid[i * N + n])
            self.assertAllSeparate(
                mesh.verts_list()
                + new_mesh.verts_list()
                + mesh.faces_list()
                + new_mesh.faces_list()
            )
            self.assertTrue(new_mesh._verts_packed is None)
            self.assertTrue(new_mesh._faces_packed is None)
            self.assertTrue(new_mesh._verts_padded is None)
            self.assertTrue(new_mesh._faces_padded is None)
            self.assertTrue(new_mesh._edges_packed is None)

        with self.assertRaises(ValueError):
            mesh.extend(N=-1)

    def test_to(self):
        mesh = TestMeshes.init_mesh(5, 10, 100, device=torch.device("cuda:0"))
        device = torch.device("cuda:1")

        new_mesh = mesh.to(device)
        self.assertTrue(new_mesh.device == device)
        self.assertTrue(mesh.device == torch.device("cuda:0"))

    def test_split_mesh(self):
        mesh = TestMeshes.init_mesh(5, 10, 100)
        split_sizes = [2, 3]
        split_meshes = mesh.split(split_sizes)
        self.assertTrue(len(split_meshes[0]) == 2)
        self.assertTrue(
            split_meshes[0].verts_list()
            == [mesh.get_mesh_verts_faces(0)[0], mesh.get_mesh_verts_faces(1)[0]]
        )
        self.assertTrue(len(split_meshes[1]) == 3)
        self.assertTrue(
            split_meshes[1].verts_list()
            == [
                mesh.get_mesh_verts_faces(2)[0],
                mesh.get_mesh_verts_faces(3)[0],
                mesh.get_mesh_verts_faces(4)[0],
            ]
        )

        split_sizes = [2, 0.3]
        with self.assertRaises(ValueError):
            mesh.split(split_sizes)

    def test_update_padded(self):
        # Define the test mesh object either as a list or tensor of faces/verts.
        N = 10
        for lists_to_tensors in (False, True):
            for force in (True, False):
                mesh = TestMeshes.init_mesh(
                    N, 100, 300, lists_to_tensors=lists_to_tensors
                )
                num_verts_per_mesh = mesh.num_verts_per_mesh()
                if force:
                    # force mesh to have computed attributes
                    mesh.verts_packed()
                    mesh.edges_packed()
                    mesh.laplacian_packed()
                    mesh.faces_areas_packed()

                new_verts = torch.rand((mesh._N, mesh._V, 3), device=mesh.device)
                new_verts_list = [
                    new_verts[i, : num_verts_per_mesh[i]] for i in range(N)
                ]
                new_mesh = mesh.update_padded(new_verts)

                # check the attributes assigned at construction time
                self.assertEqual(new_mesh._N, mesh._N)
                self.assertEqual(new_mesh._F, mesh._F)
                self.assertEqual(new_mesh._V, mesh._V)
                self.assertEqual(new_mesh.equisized, mesh.equisized)
                self.assertTrue(all(new_mesh.valid == mesh.valid))
                self.assertNotSeparate(
                    new_mesh.num_verts_per_mesh(), mesh.num_verts_per_mesh()
                )
                self.assertClose(
                    new_mesh.num_verts_per_mesh(), mesh.num_verts_per_mesh()
                )
                self.assertNotSeparate(
                    new_mesh.num_faces_per_mesh(), mesh.num_faces_per_mesh()
                )
                self.assertClose(
                    new_mesh.num_faces_per_mesh(), mesh.num_faces_per_mesh()
                )

                # check that the following attributes are not assigned
                self.assertIsNone(new_mesh._verts_list)
                self.assertIsNone(new_mesh._faces_areas_packed)
                self.assertIsNone(new_mesh._faces_normals_packed)
                self.assertIsNone(new_mesh._verts_normals_packed)

                check_tensors = [
                    "_faces_packed",
                    "_verts_packed_to_mesh_idx",
                    "_faces_packed_to_mesh_idx",
                    "_mesh_to_verts_packed_first_idx",
                    "_mesh_to_faces_packed_first_idx",
                    "_edges_packed",
                    "_edges_packed_to_mesh_idx",
                    "_mesh_to_edges_packed_first_idx",
                    "_faces_packed_to_edges_packed",
                    "_num_edges_per_mesh",
                ]
                for k in check_tensors:
                    v = getattr(new_mesh, k)
                    if not force:
                        self.assertIsNone(v)
                    else:
                        v_old = getattr(mesh, k)
                        self.assertNotSeparate(v, v_old)
                        self.assertClose(v, v_old)

                # check verts/faces padded
                self.assertClose(new_mesh.verts_padded(), new_verts)
                self.assertNotSeparate(new_mesh.verts_padded(), new_verts)
                self.assertClose(new_mesh.faces_padded(), mesh.faces_padded())
                self.assertNotSeparate(new_mesh.faces_padded(), mesh.faces_padded())
                # check verts/faces list
                for i in range(N):
                    self.assertNotSeparate(
                        new_mesh.faces_list()[i], mesh.faces_list()[i]
                    )
                    self.assertClose(new_mesh.faces_list()[i], mesh.faces_list()[i])
                    self.assertSeparate(new_mesh.verts_list()[i], mesh.verts_list()[i])
                    self.assertClose(new_mesh.verts_list()[i], new_verts_list[i])
                # check verts/faces packed
                self.assertClose(new_mesh.verts_packed(), torch.cat(new_verts_list))
                self.assertSeparate(new_mesh.verts_packed(), mesh.verts_packed())
                self.assertClose(new_mesh.faces_packed(), mesh.faces_packed())
                # check pad_to_packed
                self.assertClose(
                    new_mesh.verts_padded_to_packed_idx(),
                    mesh.verts_padded_to_packed_idx(),
                )
                # check edges
                self.assertClose(new_mesh.edges_packed(), mesh.edges_packed())

    def test_get_mesh_verts_faces(self):
        device = torch.device("cuda:0")
        verts_list = []
        faces_list = []
        verts_faces = [(10, 100), (20, 200)]
        for (V, F) in verts_faces:
            verts = torch.rand((V, 3), dtype=torch.float32, device=device)
            faces = torch.randint(V, size=(F, 3), dtype=torch.int64, device=device)
            verts_list.append(verts)
            faces_list.append(faces)

        mesh = Meshes(verts=verts_list, faces=faces_list)

        for i, (V, F) in enumerate(verts_faces):
            verts, faces = mesh.get_mesh_verts_faces(i)
            self.assertTrue(len(verts) == V)
            self.assertClose(verts, verts_list[i])
            self.assertTrue(len(faces) == F)
            self.assertClose(faces, faces_list[i])

        with self.assertRaises(ValueError):
            mesh.get_mesh_verts_faces(5)
        with self.assertRaises(ValueError):
            mesh.get_mesh_verts_faces(0.2)

    def test_get_bounding_boxes(self):
        device = torch.device("cuda:0")
        verts_list = []
        faces_list = []
        for (V, F) in [(10, 100)]:
            verts = torch.rand((V, 3), dtype=torch.float32, device=device)
            faces = torch.randint(V, size=(F, 3), dtype=torch.int64, device=device)
            verts_list.append(verts)
            faces_list.append(faces)

        mins = torch.min(verts, dim=0)[0]
        maxs = torch.max(verts, dim=0)[0]
        bboxes_gt = torch.stack([mins, maxs], dim=1).unsqueeze(0)
        mesh = Meshes(verts=verts_list, faces=faces_list)
        bboxes = mesh.get_bounding_boxes()
        self.assertClose(bboxes_gt, bboxes)

    def test_padded_to_packed_idx(self):
        device = torch.device("cuda:0")
        verts_list = []
        faces_list = []
        verts_faces = [(10, 100), (20, 200), (30, 300)]
        for (V, F) in verts_faces:
            verts = torch.rand((V, 3), dtype=torch.float32, device=device)
            faces = torch.randint(V, size=(F, 3), dtype=torch.int64, device=device)
            verts_list.append(verts)
            faces_list.append(faces)

        mesh = Meshes(verts=verts_list, faces=faces_list)
        verts_padded_to_packed_idx = mesh.verts_padded_to_packed_idx()
        verts_packed = mesh.verts_packed()
        verts_padded = mesh.verts_padded()
        verts_padded_flat = verts_padded.view(-1, 3)

        self.assertClose(verts_padded_flat[verts_padded_to_packed_idx], verts_packed)

        idx = verts_padded_to_packed_idx.view(-1, 1).expand(-1, 3)
        self.assertClose(verts_padded_flat.gather(0, idx), verts_packed)

    def test_getitem(self):
        device = torch.device("cuda:0")
        verts_list = []
        faces_list = []
        verts_faces = [(10, 100), (20, 200), (30, 300)]
        for (V, F) in verts_faces:
            verts = torch.rand((V, 3), dtype=torch.float32, device=device)
            faces = torch.randint(V, size=(F, 3), dtype=torch.int64, device=device)
            verts_list.append(verts)
            faces_list.append(faces)

        mesh = Meshes(verts=verts_list, faces=faces_list)

        def check_equal(selected, indices):
            for selectedIdx, index in enumerate(indices):
                self.assertClose(
                    selected.verts_list()[selectedIdx], mesh.verts_list()[index]
                )
                self.assertClose(
                    selected.faces_list()[selectedIdx], mesh.faces_list()[index]
                )

        # int index
        index = 1
        mesh_selected = mesh[index]
        self.assertTrue(len(mesh_selected) == 1)
        check_equal(mesh_selected, [index])

        # list index
        index = [1, 2]
        mesh_selected = mesh[index]
        self.assertTrue(len(mesh_selected) == len(index))
        check_equal(mesh_selected, index)

        # slice index
        index = slice(0, 2, 1)
        mesh_selected = mesh[index]
        check_equal(mesh_selected, [0, 1])

        # bool tensor
        index = torch.tensor([1, 0, 1], dtype=torch.bool, device=device)
        mesh_selected = mesh[index]
        self.assertTrue(len(mesh_selected) == index.sum())
        check_equal(mesh_selected, [0, 2])

        # int tensor
        index = torch.tensor([1, 2], dtype=torch.int64, device=device)
        mesh_selected = mesh[index]
        self.assertTrue(len(mesh_selected) == index.numel())
        check_equal(mesh_selected, index.tolist())

        # invalid index
        index = torch.tensor([1, 0, 1], dtype=torch.float32, device=device)
        with self.assertRaises(IndexError):
            mesh_selected = mesh[index]
        index = 1.2
        with self.assertRaises(IndexError):
            mesh_selected = mesh[index]

    def test_compute_faces_areas(self):
        verts = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.5, 0.0, 0.0],
                [0.25, 0.8, 0.0],
            ],
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2], [0, 3, 4]], dtype=torch.int64)
        mesh = Meshes(verts=[verts], faces=[faces])

        face_areas = mesh.faces_areas_packed()
        expected_areas = torch.tensor([0.125, 0.2])
        self.assertClose(face_areas, expected_areas)

    def test_compute_normals(self):

        # Simple case with one mesh where normals point in either +/- ijk
        verts = torch.tensor(
            [
                [0.1, 0.3, 0.0],
                [0.5, 0.2, 0.0],
                [0.6, 0.8, 0.0],
                [0.0, 0.3, 0.2],
                [0.0, 0.2, 0.5],
                [0.0, 0.8, 0.7],
                [0.5, 0.0, 0.2],
                [0.6, 0.0, 0.5],
                [0.8, 0.0, 0.7],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        faces = torch.tensor(
            [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=torch.int64
        )
        mesh = Meshes(verts=[verts], faces=[faces])

        verts_normals_expected = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        faces_normals_expected = verts_normals_expected[[0, 3, 6, 9], :]

        self.assertTrue(
            torch.allclose(mesh.verts_normals_list()[0], verts_normals_expected)
        )
        self.assertTrue(
            torch.allclose(mesh.faces_normals_list()[0], faces_normals_expected)
        )
        self.assertTrue(
            torch.allclose(mesh.verts_normals_packed(), verts_normals_expected)
        )
        self.assertTrue(
            torch.allclose(mesh.faces_normals_packed(), faces_normals_expected)
        )

        # Multiple meshes in the batch with equal sized meshes
        meshes_extended = mesh.extend(3)
        for m in meshes_extended.verts_normals_list():
            self.assertClose(m, verts_normals_expected)
        for f in meshes_extended.faces_normals_list():
            self.assertClose(f, faces_normals_expected)

        # Multiple meshes in the batch with different sized meshes
        # Check padded and packed normals are the correct sizes.
        verts2 = torch.tensor(
            [
                [0.1, 0.3, 0.0],
                [0.5, 0.2, 0.0],
                [0.6, 0.8, 0.0],
                [0.0, 0.3, 0.2],
                [0.0, 0.2, 0.5],
                [0.0, 0.8, 0.7],
            ],
            dtype=torch.float32,
        )
        faces2 = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int64)
        verts_list = [verts, verts2]
        faces_list = [faces, faces2]
        meshes = Meshes(verts=verts_list, faces=faces_list)
        verts_normals_padded = meshes.verts_normals_padded()
        faces_normals_padded = meshes.faces_normals_padded()

        for n in range(len(meshes)):
            v = verts_list[n].shape[0]
            f = faces_list[n].shape[0]
            if verts_normals_padded.shape[1] > v:
                self.assertTrue(verts_normals_padded[n, v:, :].eq(0).all())
                self.assertTrue(
                    torch.allclose(
                        verts_normals_padded[n, :v, :].view(-1, 3),
                        verts_normals_expected[:v, :],
                    )
                )
            if faces_normals_padded.shape[1] > f:
                self.assertTrue(faces_normals_padded[n, f:, :].eq(0).all())
                self.assertTrue(
                    torch.allclose(
                        faces_normals_padded[n, :f, :].view(-1, 3),
                        faces_normals_expected[:f, :],
                    )
                )

        verts_normals_packed = meshes.verts_normals_packed()
        faces_normals_packed = meshes.faces_normals_packed()
        self.assertTrue(
            list(verts_normals_packed.shape) == [verts.shape[0] + verts2.shape[0], 3]
        )
        self.assertTrue(
            list(faces_normals_packed.shape) == [faces.shape[0] + faces2.shape[0], 3]
        )

        # Single mesh where two faces share one vertex so the normal is
        # the weighted sum of the two face normals.
        verts = torch.tensor(
            [
                [0.1, 0.3, 0.0],
                [0.5, 0.2, 0.0],
                [0.0, 0.3, 0.2],  # vertex is shared between two faces
                [0.0, 0.2, 0.5],
                [0.0, 0.8, 0.7],
            ],
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2], [2, 3, 4]], dtype=torch.int64)
        mesh = Meshes(verts=[verts], faces=[faces])

        verts_normals_expected = torch.tensor(
            [
                [-0.2408, -0.9631, -0.1204],
                [-0.2408, -0.9631, -0.1204],
                [-0.9389, -0.3414, -0.0427],
                [-1.0000, 0.0000, 0.0000],
                [-1.0000, 0.0000, 0.0000],
            ]
        )
        faces_normals_expected = torch.tensor(
            [[-0.2408, -0.9631, -0.1204], [-1.0000, 0.0000, 0.0000]]
        )
        self.assertTrue(
            torch.allclose(
                mesh.verts_normals_list()[0], verts_normals_expected, atol=4e-5
            )
        )
        self.assertTrue(
            torch.allclose(
                mesh.faces_normals_list()[0], faces_normals_expected, atol=4e-5
            )
        )

        # Check empty mesh has empty normals
        meshes = Meshes(verts=[], faces=[])
        self.assertEqual(meshes.verts_normals_packed().shape[0], 0)
        self.assertEqual(meshes.verts_normals_padded().shape[0], 0)
        self.assertEqual(meshes.verts_normals_list(), [])
        self.assertEqual(meshes.faces_normals_packed().shape[0], 0)
        self.assertEqual(meshes.faces_normals_padded().shape[0], 0)
        self.assertEqual(meshes.faces_normals_list(), [])

    def test_compute_faces_areas_cpu_cuda(self):
        num_meshes = 10
        max_v = 100
        max_f = 300
        mesh_cpu = TestMeshes.init_mesh(num_meshes, max_v, max_f, device="cpu")
        device = torch.device("cuda:0")
        mesh_cuda = mesh_cpu.to(device)

        face_areas_cpu = mesh_cpu.faces_areas_packed()
        face_normals_cpu = mesh_cpu.faces_normals_packed()
        face_areas_cuda = mesh_cuda.faces_areas_packed()
        face_normals_cuda = mesh_cuda.faces_normals_packed()
        self.assertClose(face_areas_cpu, face_areas_cuda.cpu(), atol=1e-6)
        # because of the normalization of the normals with arbitrarily small values,
        # normals can become unstable. Thus only compare normals, for faces
        # with areas > eps=1e-6
        nonzero = face_areas_cpu > 1e-6
        self.assertClose(
            face_normals_cpu[nonzero], face_normals_cuda.cpu()[nonzero], atol=1e-6
        )

    @staticmethod
    def compute_packed_with_init(
        num_meshes: int = 10, max_v: int = 100, max_f: int = 300, device: str = "cpu"
    ):
        mesh = TestMeshes.init_mesh(num_meshes, max_v, max_f, device=device)
        torch.cuda.synchronize()

        def compute_packed():
            mesh._compute_packed(refresh=True)
            torch.cuda.synchronize()

        return compute_packed

    @staticmethod
    def compute_padded_with_init(
        num_meshes: int = 10, max_v: int = 100, max_f: int = 300, device: str = "cpu"
    ):
        mesh = TestMeshes.init_mesh(num_meshes, max_v, max_f, device=device)
        torch.cuda.synchronize()

        def compute_padded():
            mesh._compute_padded(refresh=True)
            torch.cuda.synchronize()

        return compute_padded
