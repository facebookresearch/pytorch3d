# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
from pytorch3d.structures import Meshes


class SubdivideMeshes(nn.Module):
    """
    Subdivide a triangle mesh by adding a new vertex at the center of each edge
    and dividing each face into four new faces. Vectors of vertex
    attributes can also be subdivided by averaging the values of the attributes
    at the two vertices which form each edge. This implementation
    preserves face orientation - if the vertices of a face are all ordered
    counter-clockwise, then the faces in the subdivided meshes will also have
    their vertices ordered counter-clockwise.

    If meshes is provided as an input, the initializer performs the relatively
    expensive computation of determining the new face indices. This one-time
    computation can be reused for all meshes with the same face topology
    but different vertex positions.
    """

    def __init__(self, meshes=None) -> None:
        """
        Args:
            meshes: Meshes object or None. If a meshes object is provided,
                the first mesh is used to compute the new faces of the
                subdivided topology which can be reused for meshes with
                the same input topology.
        """
        super(SubdivideMeshes, self).__init__()

        self.precomputed = False
        self._N = -1
        if meshes is not None:
            # This computation is on indices, so gradients do not need to be
            # tracked.
            mesh = meshes[0]
            with torch.no_grad():
                subdivided_faces = self.subdivide_faces(mesh)
                if subdivided_faces.shape[1] != 3:
                    raise ValueError("faces can only have three vertices")
                self.register_buffer("_subdivided_faces", subdivided_faces)
                self.precomputed = True

    def subdivide_faces(self, meshes):
        r"""
        Args:
            meshes: a Meshes object.

        Returns:
            subdivided_faces_packed: (4*sum(F_n), 3) shape LongTensor of
            original and new faces.

        Refer to pytorch3d.structures.meshes.py for more details on packed
        representations of faces.

        Each face is split into 4 faces e.g. Input face
        ::
                   v0
                   /\
                  /  \
                 /    \
             e1 /      \ e0
               /        \
              /          \
             /            \
            /______________\
          v2       e2       v1

          faces_packed = [[0, 1, 2]]
          faces_packed_to_edges_packed = [[2, 1, 0]]

        `faces_packed_to_edges_packed` is used to represent all the new
        vertex indices corresponding to the mid-points of edges in the mesh.
        The actual vertex coordinates will be computed in the forward function.
        To get the indices of the new vertices, offset
        `faces_packed_to_edges_packed` by the total number of vertices.
        ::
            faces_packed_to_edges_packed = [[2, 1, 0]] + 3 = [[5, 4, 3]]

        e.g. subdivided face
        ::
                   v0
                   /\
                  /  \
                 / f0 \
             v4 /______\ v3
               /\      /\
              /  \ f3 /  \
             / f2 \  / f1 \
            /______\/______\
           v2       v5       v1

           f0 = [0, 3, 4]
           f1 = [1, 5, 3]
           f2 = [2, 4, 5]
           f3 = [5, 4, 3]

        """
        verts_packed = meshes.verts_packed()
        with torch.no_grad():
            faces_packed = meshes.faces_packed()
            faces_packed_to_edges_packed = (
                meshes.faces_packed_to_edges_packed() + verts_packed.shape[0]
            )

            f0 = torch.stack(
                [
                    faces_packed[:, 0],
                    faces_packed_to_edges_packed[:, 2],
                    faces_packed_to_edges_packed[:, 1],
                ],
                dim=1,
            )
            f1 = torch.stack(
                [
                    faces_packed[:, 1],
                    faces_packed_to_edges_packed[:, 0],
                    faces_packed_to_edges_packed[:, 2],
                ],
                dim=1,
            )
            f2 = torch.stack(
                [
                    faces_packed[:, 2],
                    faces_packed_to_edges_packed[:, 1],
                    faces_packed_to_edges_packed[:, 0],
                ],
                dim=1,
            )
            f3 = faces_packed_to_edges_packed
            subdivided_faces_packed = torch.cat(
                [f0, f1, f2, f3], dim=0
            )  # (4*sum(F_n), 3)

            return subdivided_faces_packed

    def forward(self, meshes, feats=None):
        """
        Subdivide a batch of meshes by adding a new vertex on each edge, and
        dividing each face into four new faces. New meshes contains two types
        of vertices:
        1) Vertices that appear in the input meshes.
           Data for these vertices are copied from the input meshes.
        2) New vertices at the midpoint of each edge.
           Data for these vertices is the average of the data for the two
           vertices that make up the edge.

        Args:
            meshes: Meshes object representing a batch of meshes.
            feats: Per-vertex features to be subdivided along with the verts.
                Should be parallel to the packed vert representation of the
                input meshes; so it should have shape (V, D) where V is the
                total number of verts in the input meshes. Default: None.

        Returns:
            2-element tuple containing

            - **new_meshes**: Meshes object of a batch of subdivided meshes.
            - **new_feats**: (optional) Tensor of subdivided feats, parallel to the
              (packed) vertices of the subdivided meshes. Only returned
              if feats is not None.

        """
        self._N = len(meshes)
        if self.precomputed:
            return self.subdivide_homogeneous(meshes, feats)
        else:
            return self.subdivide_heterogenerous(meshes, feats)

    def subdivide_homogeneous(self, meshes, feats=None):
        """
        Subdivide verts (and optionally features) of a batch of meshes
        where each mesh has the same topology of faces. The subdivided faces
        are precomputed in the initializer.

        Args:
            meshes: Meshes object representing a batch of meshes.
            feats: Per-vertex features to be subdivided along with the verts.

        Returns:
            2-element tuple containing

            - **new_meshes**: Meshes object of a batch of subdivided meshes.
            - **new_feats**: (optional) Tensor of subdivided feats, parallel to the
              (packed) vertices of the subdivided meshes. Only returned
              if feats is not None.
        """
        verts = meshes.verts_padded()  # (N, V, D)
        edges = meshes[0].edges_packed()

        # The set of faces is the same across the different meshes.
        new_faces = self._subdivided_faces.view(1, -1, 3).expand(self._N, -1, -1)

        # Add one new vertex at the midpoint of each edge by taking the average
        # of the vertices that form each edge.
        new_verts = verts[:, edges].mean(dim=2)
        new_verts = torch.cat([verts, new_verts], dim=1)  # (sum(V_n)+sum(E_n), 3)
        new_feats = None

        # Calculate features for new vertices.
        if feats is not None:
            if feats.dim() == 2:
                # feats is in packed format, transform it from packed to
                # padded, i.e. (N*V, D) to (N, V, D).
                feats = feats.view(verts.size(0), verts.size(1), feats.size(1))
            if feats.dim() != 3:
                raise ValueError("features need to be of shape (N, V, D) or (N*V, D)")

            # Take average of the features at the vertices that form each edge.
            new_feats = feats[:, edges].mean(dim=2)
            new_feats = torch.cat([feats, new_feats], dim=1)  # (sum(V_n)+sum(E_n), 3)

        new_meshes = Meshes(verts=new_verts, faces=new_faces)

        if feats is None:
            return new_meshes
        else:
            return new_meshes, new_feats

    def subdivide_heterogenerous(self, meshes, feats=None):
        """
        Subdivide faces, verts (and optionally features) of a batch of meshes
        where each mesh can have different face topologies.

        Args:
            meshes: Meshes object representing a batch of meshes.
            feats: Per-vertex features to be subdivided along with the verts.

        Returns:
            2-element tuple containing

            - **new_meshes**: Meshes object of a batch of subdivided meshes.
            - **new_feats**: (optional) Tensor of subdivided feats, parallel to the
              (packed) vertices of the subdivided meshes. Only returned
              if feats is not None.
        """

        # The computation of new faces is on face indices, so gradients do not
        # need to be tracked.
        verts = meshes.verts_packed()
        with torch.no_grad():
            new_faces = self.subdivide_faces(meshes)
            edges = meshes.edges_packed()
            face_to_mesh_idx = meshes.faces_packed_to_mesh_idx()
            edge_to_mesh_idx = meshes.edges_packed_to_mesh_idx()
            num_edges_per_mesh = edge_to_mesh_idx.bincount(minlength=self._N)
            num_verts_per_mesh = meshes.num_verts_per_mesh()
            num_faces_per_mesh = meshes.num_faces_per_mesh()

            # Add one new vertex at the midpoint of each edge.
            new_verts_per_mesh = num_verts_per_mesh + num_edges_per_mesh  # (N,)
            new_face_to_mesh_idx = torch.cat([face_to_mesh_idx] * 4, dim=0)

            # Calculate the indices needed to group the new and existing verts
            # for each mesh.
            verts_sort_idx = _create_verts_index(
                num_verts_per_mesh, num_edges_per_mesh, meshes.device
            )  # (sum(V_n)+sum(E_n),)

            verts_ordered_idx_init = torch.zeros(
                new_verts_per_mesh.sum(), dtype=torch.int64, device=meshes.device
            )  # (sum(V_n)+sum(E_n),)

            # Reassign vertex indices so that existing and new vertices for each
            # mesh are sequential.
            verts_ordered_idx = verts_ordered_idx_init.scatter_add(
                0,
                verts_sort_idx,
                torch.arange(new_verts_per_mesh.sum(), device=meshes.device),
            )

            # Retrieve vertex indices for each face.
            new_faces = verts_ordered_idx[new_faces]

            # Calculate the indices needed to group the existing and new faces
            # for each mesh.
            face_sort_idx = _create_faces_index(
                num_faces_per_mesh, device=meshes.device
            )

            # Reorder the faces to sequentially group existing and new faces
            # for each mesh.
            new_faces = new_faces[face_sort_idx]
            new_face_to_mesh_idx = new_face_to_mesh_idx[face_sort_idx]
            new_faces_per_mesh = new_face_to_mesh_idx.bincount(
                minlength=self._N
            )  # (sum(F_n)*4)

        # Add one new vertex at the midpoint of each edge by taking the average
        # of the verts that form each edge.
        new_verts = verts[edges].mean(dim=1)
        new_verts = torch.cat([verts, new_verts], dim=0)

        # Reorder the verts to sequentially group existing and new verts for
        # each mesh.
        new_verts = new_verts[verts_sort_idx]

        if feats is not None:
            new_feats = feats[edges].mean(dim=1)
            new_feats = torch.cat([feats, new_feats], dim=0)
            new_feats = new_feats[verts_sort_idx]

        verts_list = list(new_verts.split(new_verts_per_mesh.tolist(), 0))
        faces_list = list(new_faces.split(new_faces_per_mesh.tolist(), 0))
        new_verts_per_mesh_cumsum = torch.cat(
            [
                new_verts_per_mesh.new_full(size=(1,), fill_value=0.0),
                new_verts_per_mesh.cumsum(0)[:-1],
            ],
            dim=0,
        )
        faces_list = [
            faces_list[n] - new_verts_per_mesh_cumsum[n] for n in range(self._N)
        ]
        if feats is not None:
            feats_list = new_feats.split(new_verts_per_mesh.tolist(), 0)
        new_meshes = Meshes(verts=verts_list, faces=faces_list)

        if feats is None:
            return new_meshes
        else:
            new_feats = torch.cat(feats_list, dim=0)
            return new_meshes, new_feats


def _create_verts_index(verts_per_mesh, edges_per_mesh, device=None):
    """
    Helper function to group the vertex indices for each mesh. New vertices are
    stacked at the end of the original verts tensor, so in order to have
    sequential packing, the verts tensor needs to be reordered so that the
    vertices corresponding to each mesh are grouped together.

    Args:
        verts_per_mesh: Tensor of shape (N,) giving the number of vertices
            in each mesh in the batch where N is the batch size.
        edges_per_mesh: Tensor of shape (N,) giving the number of edges
            in each mesh in the batch

    Returns:
        verts_idx: A tensor with vert indices for each mesh ordered sequentially
            by mesh index.
    """
    # e.g. verts_per_mesh = (4, 5, 6)
    # e.g. edges_per_mesh = (5, 7, 9)

    V = verts_per_mesh.sum()  # e.g. 15
    E = edges_per_mesh.sum()  # e.g. 21

    verts_per_mesh_cumsum = verts_per_mesh.cumsum(dim=0)  # (N,) e.g. (4, 9, 15)
    edges_per_mesh_cumsum = edges_per_mesh.cumsum(dim=0)  # (N,) e.g. (5, 12, 21)

    v_to_e_idx = verts_per_mesh_cumsum.clone()

    # vertex to edge index.
    v_to_e_idx[1:] += edges_per_mesh_cumsum[
        :-1
    ]  # e.g. (4, 9, 15) + (0, 5, 12) = (4, 14, 27)

    # vertex to edge offset.
    v_to_e_offset = V - verts_per_mesh_cumsum  # e.g. 15 - (4, 9, 15) = (11, 6, 0)
    v_to_e_offset[1:] += edges_per_mesh_cumsum[
        :-1
    ]  # e.g. (11, 6, 0) + (0, 5, 12) = (11, 11, 12)
    e_to_v_idx = (
        verts_per_mesh_cumsum[:-1] + edges_per_mesh_cumsum[:-1]
    )  # (4, 9) + (5, 12) = (9, 21)
    e_to_v_offset = (
        verts_per_mesh_cumsum[:-1] - edges_per_mesh_cumsum[:-1] - V
    )  # (4, 9) - (5, 12) - 15 = (-16, -18)

    # Add one new vertex per edge.
    idx_diffs = torch.ones(V + E, device=device, dtype=torch.int64)  # (36,)
    idx_diffs[v_to_e_idx] += v_to_e_offset
    idx_diffs[e_to_v_idx] += e_to_v_offset

    # e.g.
    # [
    #  1, 1, 1, 1, 12, 1, 1, 1, 1,
    #  -15, 1, 1, 1, 1, 12, 1, 1, 1, 1, 1, 1,
    #  -17, 1, 1, 1, 1, 1, 13, 1, 1, 1, 1, 1, 1, 1
    # ]

    verts_idx = idx_diffs.cumsum(dim=0) - 1

    # e.g.
    # [
    #  0, 1, 2, 3, 15, 16, 17, 18, 19,                            --> mesh 0
    #  4, 5, 6, 7, 8, 20, 21, 22, 23, 24, 25, 26,                 --> mesh 1
    #  9, 10, 11, 12, 13, 14, 27, 28, 29, 30, 31, 32, 33, 34, 35  --> mesh 2
    # ]
    # where for mesh 0, [0, 1, 2, 3] are the indices of the existing verts, and
    # [15, 16, 17, 18, 19] are the indices of the new verts after subdivision.

    return verts_idx


def _create_faces_index(faces_per_mesh: torch.Tensor, device=None):
    """
    Helper function to group the faces indices for each mesh. New faces are
    stacked at the end of the original faces tensor, so in order to have
    sequential packing, the faces tensor needs to be reordered to that faces
    corresponding to each mesh are grouped together.

    Args:
        faces_per_mesh: Tensor of shape (N,) giving the number of faces
            in each mesh in the batch where N is the batch size.

    Returns:
        faces_idx: A tensor with face indices for each mesh ordered sequentially
            by mesh index.
    """
    # e.g. faces_per_mesh = [2, 5, 3]

    F = faces_per_mesh.sum()  # e.g. 10
    faces_per_mesh_cumsum = faces_per_mesh.cumsum(dim=0)  # (N,) e.g. (2, 7, 10)

    switch1_idx = faces_per_mesh_cumsum.clone()
    switch1_idx[1:] += (
        3 * faces_per_mesh_cumsum[:-1]
    )  # e.g. (2, 7, 10) + (0, 6, 21) = (2, 13, 31)

    switch2_idx = 2 * faces_per_mesh_cumsum  # e.g. (4, 14, 20)
    switch2_idx[1:] += (
        2 * faces_per_mesh_cumsum[:-1]
    )  # e.g. (4, 14, 20) + (0, 4, 14) = (4, 18, 34)

    switch3_idx = 3 * faces_per_mesh_cumsum  # e.g. (6, 21, 30)
    switch3_idx[1:] += faces_per_mesh_cumsum[
        :-1
    ]  # e.g. (6, 21, 30) + (0, 2, 7) = (6, 23, 37)

    switch4_idx = 4 * faces_per_mesh_cumsum[:-1]  # e.g. (8, 28)

    switch123_offset = F - faces_per_mesh  # e.g. (8, 5, 7)

    # pyre-fixme[6]: For 1st param expected `Union[List[int], Size,
    #  typing.Tuple[int, ...]]` but got `Tensor`.
    idx_diffs = torch.ones(4 * F, device=device, dtype=torch.int64)
    idx_diffs[switch1_idx] += switch123_offset
    idx_diffs[switch2_idx] += switch123_offset
    idx_diffs[switch3_idx] += switch123_offset
    idx_diffs[switch4_idx] -= 3 * F

    # e.g
    # [
    #  1, 1, 9, 1, 9, 1, 9, 1,                                       -> mesh 0
    #  -29, 1, 1, 1, 1, 6, 1, 1, 1, 1, 6, 1, 1, 1, 1, 6, 1, 1, 1, 1, -> mesh 1
    #  -29, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1                          -> mesh 2
    # ]

    faces_idx = idx_diffs.cumsum(dim=0) - 1

    # e.g.
    # [
    #  0, 1, 10, 11, 20, 21, 30, 31,
    #  2, 3, 4, 5, 6, 12, 13, 14, 15, 16, 22, 23, 24, 25, 26, 32, 33, 34, 35, 36,
    #  7, 8, 9, 17, 18, 19, 27, 28, 29, 37, 38, 39
    # ]
    # where for mesh 0, [0, 1] are the indices of the existing faces, and
    # [10, 11, 20, 21, 30, 31] are the indices of the new faces after subdivision.

    return faces_idx
