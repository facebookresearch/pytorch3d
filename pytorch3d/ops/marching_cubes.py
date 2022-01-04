# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

import torch
from pytorch3d.ops.marching_cubes_data import EDGE_TABLE, EDGE_TO_VERTICES, FACE_TABLE
from pytorch3d.transforms import Translate


EPS = 0.00001


class Cube:
    def __init__(self, bfl_vertex: Tuple[int, int, int], spacing: int = 1) -> None:
        """
        Initializes a cube given the bottom front left vertex coordinate
        and the cube spacing

        Edge and vertex convention:

                    v4_______e4____________v5
                    /|                    /|
                   / |                   / |
                e7/  |                e5/  |
                 /___|______e6_________/   |
              v7|    |                 |v6 |e9
                |    |                 |   |
                |    |e8               |e10|
             e11|    |                 |   |
                |    |_________________|___|
                |   / v0      e0       |   /v1
                |  /                   |  /
                | /e3                  | /e1
                |/_____________________|/
                v3         e2          v2

        Args:
            bfl_vertex: a tuple of size 3 corresponding to the bottom front left vertex
                of the cube in (x, y, z) format
            spacing: the length of each edge of the cube
        """
        # match corner orders to algorithm convention
        if len(bfl_vertex) != 3:
            msg = "The vertex {} is size {} instead of size 3".format(
                bfl_vertex, len(bfl_vertex)
            )
            raise ValueError(msg)

        x, y, z = bfl_vertex
        self.vertices = torch.tensor(
            [
                [x, y, z + spacing],
                [x + spacing, y, z + spacing],
                [x + spacing, y, z],
                [x, y, z],
                [x, y + spacing, z + spacing],
                [x + spacing, y + spacing, z + spacing],
                [x + spacing, y + spacing, z],
                [x, y + spacing, z],
            ]
        )

    def get_index(self, volume_data: torch.Tensor, isolevel: float) -> int:
        """
        Calculates the cube_index in the range 0-255 to index
        into EDGE_TABLE and FACE_TABLE
        Args:
            volume_data: the 3D scalar data
            isolevel: the isosurface value used as a threshold
                for determining whether a point is inside/outside
                the volume
        """
        cube_index = 0
        bit = 1
        for index in range(len(self.vertices)):
            vertex = self.vertices[index]
            value = _get_value(vertex, volume_data)
            if value < isolevel:
                cube_index |= bit
            bit *= 2
        return cube_index


def marching_cubes_naive(
    volume_data_batch: torch.Tensor,
    isolevel: Optional[float] = None,
    spacing: int = 1,
    return_local_coords: bool = True,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Runs the classic marching cubes algorithm, iterating over
    the coordinates of the volume_data and using a given isolevel
    for determining intersected edges of cubes of size `spacing`.
    Returns vertices and faces of the obtained mesh.
    This operation is non-differentiable.

    This is a naive implementation, and is not optimized for efficiency.

    Args:
        volume_data_batch: a Tensor of size (N, D, H, W) corresponding to
            a batch of 3D scalar fields
        isolevel: the isosurface value to use as the threshold to determine
            whether points are within a volume. If None, then the average of the
            maximum and minimum value of the scalar field will be used.
        spacing: an integer specifying the cube size to use
        return_local_coords: bool. If True the output vertices will be in local coordinates in
        the range [-1, 1] x [-1, 1] x [-1, 1]. If False they will be in the range
        [0, W-1] x [0, H-1] x [0, D-1]
    Returns:
        verts: [(V_0, 3), (V_1, 3), ...] List of N FloatTensors of vertices.
        faces: [(F_0, 3), (F_1, 3), ...] List of N LongTensors of faces.
    """
    volume_data_batch = volume_data_batch.detach().cpu()
    batched_verts, batched_faces = [], []
    D, H, W = volume_data_batch.shape[1:]
    volume_size_xyz = volume_data_batch.new_tensor([W, H, D])[None]

    if return_local_coords:
        # Convert from local coordinates in the range [-1, 1] range to
        # world coordinates in the range [0, D-1], [0, H-1], [0, W-1]
        local_to_world_transform = Translate(
            x=+1.0, y=+1.0, z=+1.0, device=volume_data_batch.device
        ).scale((volume_size_xyz - 1) * spacing * 0.5)
        # Perform the inverse to go from world to local
        world_to_local_transform = local_to_world_transform.inverse()

    for i in range(len(volume_data_batch)):
        volume_data = volume_data_batch[i]
        curr_isolevel = (
            ((volume_data.max() + volume_data.min()) / 2).item()
            if isolevel is None
            else isolevel
        )
        edge_vertices_to_index = {}
        vertex_coords_to_index = {}
        verts, faces = [], []
        # Use length - spacing for the bounds since we are using
        # cubes of size spacing, with the lowest x,y,z values
        # (bottom front left)
        for x in range(0, W - spacing, spacing):
            for y in range(0, H - spacing, spacing):
                for z in range(0, D - spacing, spacing):
                    cube = Cube((x, y, z), spacing)
                    new_verts, new_faces = polygonise(
                        cube,
                        curr_isolevel,
                        volume_data,
                        edge_vertices_to_index,
                        vertex_coords_to_index,
                    )
                    verts.extend(new_verts)
                    faces.extend(new_faces)
        if len(faces) > 0 and len(verts) > 0:
            verts = torch.tensor(verts, dtype=torch.float32)
            # Convert vertices from world to local coords
            if return_local_coords:
                verts = world_to_local_transform.transform_points(verts[None, ...])
                verts = verts.squeeze()
            batched_verts.append(verts)
            batched_faces.append(torch.tensor(faces, dtype=torch.int64))
    return batched_verts, batched_faces


def polygonise(
    cube: Cube,
    isolevel: float,
    volume_data: torch.Tensor,
    edge_vertices_to_index: Dict[Tuple[Tuple, Tuple], int],
    vertex_coords_to_index: Dict[Tuple[float, float, float], int],
) -> Tuple[list, list]:
    """
    Runs the classic marching cubes algorithm for one Cube in the volume.
    Returns the vertices and faces for the given cube.

    Args:
        cube: a Cube indicating the cube being examined for edges that intersect
            the volume data.
        isolevel: the isosurface value to use as the threshold to determine
            whether points are within a volume.
        volume_data: a Tensor of shape (D, H, W) corresponding to
            a 3D scalar field
        edge_vertices_to_index: A dictionary which maps an edge's two coordinates
            to the index of its interpolated point, if that interpolated point
            has already been used by a previous point
        vertex_coords_to_index: A dictionary mapping a point (x, y, z) to the corresponding
            index of that vertex, if that point has already been marked as a vertex.
    Returns:
        verts: List of triangle vertices for the given cube in the volume
        faces: List of triangle faces for the given cube in the volume
    """
    num_existing_verts = max(edge_vertices_to_index.values(), default=-1) + 1
    verts, faces = [], []
    cube_index = cube.get_index(volume_data, isolevel)
    edges = EDGE_TABLE[cube_index]
    edge_indices = _get_edge_indices(edges)
    if len(edge_indices) == 0:
        return [], []

    new_verts, edge_index_to_point_index = _calculate_interp_vertices(
        edge_indices,
        volume_data,
        cube,
        isolevel,
        edge_vertices_to_index,
        vertex_coords_to_index,
        num_existing_verts,
    )

    # Create faces
    face_triangles = FACE_TABLE[cube_index]
    for i in range(0, len(face_triangles), 3):
        tri1 = edge_index_to_point_index[face_triangles[i]]
        tri2 = edge_index_to_point_index[face_triangles[i + 1]]
        tri3 = edge_index_to_point_index[face_triangles[i + 2]]
        if tri1 != tri2 and tri2 != tri3 and tri1 != tri3:
            faces.append([tri1, tri2, tri3])

    verts += new_verts
    return verts, faces


def _get_edge_indices(edges: int) -> List[int]:
    """
    Finds which edge numbers are intersected given the bit representation
    detailed in marching_cubes_data.EDGE_TABLE.

    Args:
        edges: an integer corresponding to the value at cube_index
            from the EDGE_TABLE in marching_cubes_data.py

    Returns:
        edge_indices: A list of edge indices
    """
    if edges == 0:
        return []

    edge_indices = []
    for i in range(12):
        if edges & (2 ** i):
            edge_indices.append(i)
    return edge_indices


def _calculate_interp_vertices(
    edge_indices: List[int],
    volume_data: torch.Tensor,
    cube: Cube,
    isolevel: float,
    edge_vertices_to_index: Dict[Tuple[Tuple, Tuple], int],
    vertex_coords_to_index: Dict[Tuple[float, float, float], int],
    num_existing_verts: int,
) -> Tuple[List, Dict[int, int]]:
    """
    Finds the interpolated vertices for the intersected edges, either referencing
    previous calculations or newly calculating and storing the new interpolated
    points.

    Args:
        edge_indices: the numbers of the edges which are intersected. See the
            Cube class for more detail on the edge numbering convention.
        volume_data: a Tensor of size (D, H, W) corresponding to
            a 3D scalar field
        cube: a Cube indicating the cube being examined for edges that intersect
            the volume
        isolevel: the isosurface value to use as the threshold to determine
            whether points are within a volume.
        edge_vertices_to_index: A dictionary which maps an edge's two coordinates
            to the index of its interpolated point, if that interpolated point
            has already been used by a previous point
        vertex_coords_to_index: A dictionary mapping a point (x, y, z) to the corresponding
            index of that vertex, if that point has already been marked as a vertex.
        num_existing_verts: the number of vertices that have been found in previous
            calls to polygonise for the given volume_data in the above function, marching_cubes.
            This is equal to the 1 + the maximum value in edge_vertices_to_index.
    Returns:
        interp_points: a list of new interpolated points
        edge_index_to_point_index: a dictionary mapping an edge number to the index in the
            marching cubes' vertices list of the interpolated point on that edge. To be precise,
            it refers to the index within the vertices list after interp_points
            has been appended to the verts list constructed in the marching_cubes_naive
            function.
    """
    interp_points = []
    edge_index_to_point_index = {}
    for edge_index in edge_indices:
        v1, v2 = EDGE_TO_VERTICES[edge_index]
        point1, point2 = cube.vertices[v1], cube.vertices[v2]
        p_tuple1, p_tuple2 = tuple(point1.tolist()), tuple(point2.tolist())
        if (p_tuple1, p_tuple2) in edge_vertices_to_index:
            edge_index_to_point_index[edge_index] = edge_vertices_to_index[
                (p_tuple1, p_tuple2)
            ]
        else:
            val1, val2 = _get_value(point1, volume_data), _get_value(
                point2, volume_data
            )

            point = None
            if abs(isolevel - val1) < EPS:
                point = point1

            if abs(isolevel - val2) < EPS:
                point = point2

            if abs(val1 - val2) < EPS:
                point = point1

            if point is None:
                mu = (isolevel - val1) / (val2 - val1)
                x1, y1, z1 = point1
                x2, y2, z2 = point2
                x = x1 + mu * (x2 - x1)
                y = y1 + mu * (y2 - y1)
                z = z1 + mu * (z2 - z1)
            else:
                x, y, z = point

            x, y, z = x.item(), y.item(), z.item()  # for dictionary keys

            vert_index = None
            if (x, y, z) in vertex_coords_to_index:
                vert_index = vertex_coords_to_index[(x, y, z)]
            else:
                vert_index = num_existing_verts + len(interp_points)
                interp_points.append([x, y, z])
                vertex_coords_to_index[(x, y, z)] = vert_index

            edge_vertices_to_index[(p_tuple1, p_tuple2)] = vert_index
            edge_index_to_point_index[edge_index] = vert_index

    return interp_points, edge_index_to_point_index


def _get_value(point: Tuple[int, int, int], volume_data: torch.Tensor) -> float:
    """
    Gets the value at a given coordinate point in the scalar field.

    Args:
        point: data of shape (3) corresponding to an xyz coordinate.
        volume_data: a Tensor of size (D, H, W) corresponding to
            a 3D scalar field
    Returns:
        data: scalar value in the volume at the given point
    """
    x, y, z = point
    return volume_data[z][y][x]
