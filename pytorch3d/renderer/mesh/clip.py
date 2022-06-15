# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional, Tuple

import torch


"""
Mesh clipping is done before rasterization and is implemented using 4 cases
(these will be referred to throughout the functions below)

Case 1: the triangle is completely in front of the clipping plane (it is left
        unchanged)
Case 2: the triangle is completely behind the clipping plane (it is culled)
Case 3: the triangle has exactly two vertices behind the clipping plane (it is
        clipped into a smaller triangle)
Case 4: the triangle has exactly one vertex behind the clipping plane (it is clipped
        into a smaller quadrilateral and divided into two triangular faces)

After rasterization, the Fragments from the clipped/modified triangles
are mapped back to the triangles in the original mesh. The indices,
barycentric coordinates and distances are all relative to original mesh triangles.

NOTE: It is assumed that all z-coordinates are in world coordinates (not NDC
coordinates), while x and y coordinates may be in NDC/screen coordinates
(i.e after applying a projective transform e.g. cameras.transform_points(points)).
"""


class ClippedFaces:
    """
    Helper class to store the data for the clipped version of a Meshes object
    (face_verts, mesh_to_face_first_idx, num_faces_per_mesh) along with
    conversion information (faces_clipped_to_unclipped_idx, barycentric_conversion,
    faces_clipped_to_conversion_idx, clipped_faces_neighbor_idx) required to convert
    barycentric coordinates from rasterization of the clipped Meshes to barycentric
    coordinates in terms of the unclipped Meshes.

    Args:
        face_verts: FloatTensor of shape (F_clipped, 3, 3) giving the verts of
            each of the clipped faces
        mesh_to_face_first_idx: an tensor of shape (N,), where N is the number of meshes
            in the batch.  The ith element stores the index into face_verts
            of the first face of the ith mesh.
        num_faces_per_mesh: a tensor of shape (N,) storing the number of faces in each mesh.
        faces_clipped_to_unclipped_idx: (F_clipped,) shaped LongTensor mapping each clipped
            face back to the face in faces_unclipped (i.e. the faces in the original meshes
            obtained using meshes.faces_packed())
        barycentric_conversion: (T, 3, 3) FloatTensor, where barycentric_conversion[i, :, k]
            stores the barycentric weights in terms of the world coordinates of the original
            (big) unclipped triangle for the kth vertex in the clipped (small) triangle.
            If the rasterizer then expresses some NDC coordinate in terms of barycentric
            world coordinates for the clipped (small) triangle as alpha_clipped[i,:],
            alpha_unclipped[i, :] = barycentric_conversion[i, :, :]*alpha_clipped[i, :]
        faces_clipped_to_conversion_idx: (F_clipped,) shaped LongTensor mapping each clipped
            face to the applicable row of barycentric_conversion (or set to -1 if conversion is
            not needed).
        clipped_faces_neighbor_idx: LongTensor of shape (F_clipped,) giving the index of the
            neighboring face for each case 4 triangle. e.g. for a case 4 face with f split
            into two triangles (t1, t2): clipped_faces_neighbor_idx[t1_idx] = t2_idx.
            Faces which are not clipped and subdivided are set to -1 (i.e cases 1/2/3).
    """

    __slots__ = [
        "face_verts",
        "mesh_to_face_first_idx",
        "num_faces_per_mesh",
        "faces_clipped_to_unclipped_idx",
        "barycentric_conversion",
        "faces_clipped_to_conversion_idx",
        "clipped_faces_neighbor_idx",
    ]

    def __init__(
        self,
        face_verts: torch.Tensor,
        mesh_to_face_first_idx: torch.Tensor,
        num_faces_per_mesh: torch.Tensor,
        faces_clipped_to_unclipped_idx: Optional[torch.Tensor] = None,
        barycentric_conversion: Optional[torch.Tensor] = None,
        faces_clipped_to_conversion_idx: Optional[torch.Tensor] = None,
        clipped_faces_neighbor_idx: Optional[torch.Tensor] = None,
    ) -> None:
        self.face_verts = face_verts
        self.mesh_to_face_first_idx = mesh_to_face_first_idx
        self.num_faces_per_mesh = num_faces_per_mesh
        self.faces_clipped_to_unclipped_idx = faces_clipped_to_unclipped_idx
        self.barycentric_conversion = barycentric_conversion
        self.faces_clipped_to_conversion_idx = faces_clipped_to_conversion_idx
        self.clipped_faces_neighbor_idx = clipped_faces_neighbor_idx


class ClipFrustum:
    """
    Helper class to store the information needed to represent a view frustum
    (left, right, top, bottom, znear, zfar), which is used to clip or cull triangles.
    Values left as None mean that culling should not be performed for that axis.
    The parameters perspective_correct, cull, and z_clip_value are used to define
    behavior for clipping triangles to the frustum.

    Args:
        left: NDC coordinate of the left clipping plane (along x axis)
        right: NDC coordinate of the right clipping plane (along x axis)
        top: NDC coordinate of the top clipping plane (along y axis)
        bottom: NDC coordinate of the bottom clipping plane (along y axis)
        znear: world space z coordinate of the near clipping plane
        zfar: world space z coordinate of the far clipping plane
        perspective_correct: should be set to True for a perspective camera
        cull: if True, triangles outside the frustum should be culled
        z_clip_value: if not None, then triangles should be clipped (possibly into
            smaller triangles) such that z >= z_clip_value.  This avoids projections
            that go to infinity as z->0
    """

    __slots__ = [
        "left",
        "right",
        "top",
        "bottom",
        "znear",
        "zfar",
        "perspective_correct",
        "cull",
        "z_clip_value",
    ]

    def __init__(
        self,
        left: Optional[float] = None,
        right: Optional[float] = None,
        top: Optional[float] = None,
        bottom: Optional[float] = None,
        znear: Optional[float] = None,
        zfar: Optional[float] = None,
        perspective_correct: bool = False,
        cull: bool = True,
        z_clip_value: Optional[float] = None,
    ) -> None:
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.znear = znear
        self.zfar = zfar
        self.perspective_correct = perspective_correct
        self.cull = cull
        self.z_clip_value = z_clip_value


def _get_culled_faces(face_verts: torch.Tensor, frustum: ClipFrustum) -> torch.Tensor:
    """
    Helper function used to find all the faces in Meshes which are
    fully outside the view frustum. A face is culled if all 3 vertices are outside
    the same axis of the view frustum.

    Args:
        face_verts: An (F,3,3) tensor, where F is the number of faces in
            the packed representation of Meshes. The 2nd dimension represents the 3 vertices
            of a triangle, and the 3rd dimension stores the xyz locations of each
            vertex.
        frustum: An instance of the ClipFrustum class with the information on the
            position of the clipping planes.

    Returns:
        faces_culled: An boolean tensor of size F specifying whether or not each face should be
            culled.
    """
    clipping_planes = (
        (frustum.left, 0, "<"),
        (frustum.right, 0, ">"),
        (frustum.top, 1, "<"),
        (frustum.bottom, 1, ">"),
        (frustum.znear, 2, "<"),
        (frustum.zfar, 2, ">"),
    )
    faces_culled = torch.zeros(
        [face_verts.shape[0]], dtype=torch.bool, device=face_verts.device
    )
    for plane in clipping_planes:
        clip_value, axis, op = plane
        # If clip_value is None then don't clip along that plane
        if frustum.cull and clip_value is not None:
            if op == "<":
                verts_clipped = face_verts[:, axis] < clip_value
            else:
                verts_clipped = face_verts[:, axis] > clip_value

            # If all verts are clipped then face is outside the frustum
            faces_culled |= verts_clipped.sum(1) == 3

    return faces_culled


def _find_verts_intersecting_clipping_plane(
    face_verts: torch.Tensor,
    p1_face_ind: torch.Tensor,
    clip_value: float,
    perspective_correct: bool,
) -> Tuple[Tuple[Any, Any, Any, Any, Any], List[Any]]:
    r"""
    Helper function to find the vertices used to form a new triangle for case 3/case 4 faces.

    Given a list of triangles that are already known to intersect the clipping plane,
    solve for the two vertices p4 and p5 where the edges of the triangle intersects the
    clipping plane.

                       p1
                       /\
                      /  \
                     /  t \
     _____________p4/______\p5__________ clip_value
                   /        \
                  /____      \
                p2     ---____\p3

    Args:
        face_verts: An (F,3,3) tensor, where F is the number of faces in
            the packed representation of the Meshes, the 2nd dimension represents
            the 3 vertices of the face, and the 3rd dimension stores the xyz locations of each
            vertex.  The z-coordinates must be represented in world coordinates, while
            the xy-coordinates may be in NDC/screen coordinates (i.e. after projection).
        p1_face_ind: A tensor of shape (N,) with values in the range of 0 to 2.  In each
            case 3/case 4 triangle, two vertices are on the same side of the
            clipping plane and the 3rd is on the other side.  p1_face_ind stores the index of
            the vertex that is not on the same side as any other vertex in the triangle.
        clip_value: Float, the z-value defining where to clip the triangle.
        perspective_correct: Bool, Should be set to true if a perspective camera was
            used and xy-coordinates of face_verts_unclipped are in NDC/screen coordinates.

    Returns:
        A 2-tuple
            p: (p1, p2, p3, p4, p5))
            p_barycentric (p1_bary, p2_bary, p3_bary, p4_bary, p5_bary)

        Each of p1...p5 is an (F,3) tensor of the xyz locations of the 5 points in the
        diagram above for case 3/case 4 faces. Each p1_bary...p5_bary is an (F, 3) tensor
        storing the barycentric weights used to encode p1...p5 in terms of the the original
        unclipped triangle.
    """

    # Let T be number of triangles in face_verts (note that these correspond to the subset
    # of case 1 or case 2 triangles). p1_face_ind, p2_face_ind, and p3_face_ind are (T)
    # tensors with values in the range of 0 to 2.  p1_face_ind stores the index of the
    # vertex that is not on the same side as any other vertex in the triangle, and
    # p2_face_ind and p3_face_ind are the indices of the other two vertices preserving
    # the same counterclockwise or clockwise ordering
    T = face_verts.shape[0]
    p2_face_ind = torch.remainder(p1_face_ind + 1, 3)
    p3_face_ind = torch.remainder(p1_face_ind + 2, 3)

    # p1, p2, p3 are (T, 3) tensors storing the corresponding (x, y, z) coordinates
    # of p1_face_ind, p2_face_ind, p3_face_ind
    p1 = face_verts.gather(1, p1_face_ind[:, None, None].expand(-1, -1, 3)).squeeze(1)
    p2 = face_verts.gather(1, p2_face_ind[:, None, None].expand(-1, -1, 3)).squeeze(1)
    p3 = face_verts.gather(1, p3_face_ind[:, None, None].expand(-1, -1, 3)).squeeze(1)

    ##################################
    # Solve for intersection point p4
    ##################################

    # p4 is a (T, 3) tensor is the point on the segment between p1 and p2 that
    # intersects the clipping plane.
    # Solve for the weight w2 such that p1.z*(1-w2) + p2.z*w2 = clip_value.
    # Then interpolate p4 = p1*(1-w2) + p2*w2 where it is assumed that z-coordinates
    # are expressed in world coordinates (since we want to clip z in world coordinates).
    w2 = (p1[:, 2] - clip_value) / (p1[:, 2] - p2[:, 2])
    p4 = p1 * (1 - w2[:, None]) + p2 * w2[:, None]
    if perspective_correct:
        # It is assumed that all z-coordinates are in world coordinates (not NDC
        # coordinates), while x and y coordinates may be in NDC/screen coordinates.
        # If x and y are in NDC/screen coordinates and a projective transform was used
        # in a perspective camera, then we effectively want to:
        # 1. Convert back to world coordinates (by multiplying by z)
        # 2. Interpolate using w2
        # 3. Convert back to NDC/screen coordinates (by dividing by the new z=clip_value)
        p1_world = p1[:, :2] * p1[:, 2:3]
        p2_world = p2[:, :2] * p2[:, 2:3]
        p4[:, :2] = (p1_world * (1 - w2[:, None]) + p2_world * w2[:, None]) / clip_value

    ##################################
    # Solve for intersection point p5
    ##################################

    # p5 is a (T, 3) tensor representing the point on the segment between p1 and p3 that
    # intersects the clipping plane.
    # Solve for the weight w3 such that p1.z * (1-w3) + p2.z * w3 = clip_value,
    # and then interpolate p5 = p1 * (1-w3) + p3 * w3
    w3 = (p1[:, 2] - clip_value) / (p1[:, 2] - p3[:, 2])
    w3 = w3.detach()
    p5 = p1 * (1 - w3[:, None]) + p3 * w3[:, None]
    if perspective_correct:
        # Again if using a perspective camera, convert back to world coordinates
        # interpolate and convert back
        p1_world = p1[:, :2] * p1[:, 2:3]
        p3_world = p3[:, :2] * p3[:, 2:3]
        p5[:, :2] = (p1_world * (1 - w3[:, None]) + p3_world * w3[:, None]) / clip_value

    # Set the barycentric coordinates of p1,p2,p3,p4,p5 in terms of the original
    # unclipped triangle in face_verts.
    T_idx = torch.arange(T, device=face_verts.device)
    p_barycentric = [torch.zeros((T, 3), device=face_verts.device) for i in range(5)]
    p_barycentric[0][(T_idx, p1_face_ind)] = 1
    p_barycentric[1][(T_idx, p2_face_ind)] = 1
    p_barycentric[2][(T_idx, p3_face_ind)] = 1
    p_barycentric[3][(T_idx, p1_face_ind)] = 1 - w2
    p_barycentric[3][(T_idx, p2_face_ind)] = w2
    p_barycentric[4][(T_idx, p1_face_ind)] = 1 - w3
    p_barycentric[4][(T_idx, p3_face_ind)] = w3

    p = (p1, p2, p3, p4, p5)

    return p, p_barycentric


###################
# Main Entry point
###################
def clip_faces(
    face_verts_unclipped: torch.Tensor,
    mesh_to_face_first_idx: torch.Tensor,
    num_faces_per_mesh: torch.Tensor,
    frustum: ClipFrustum,
) -> ClippedFaces:
    """
    Clip a mesh to the portion contained within a view frustum and with z > z_clip_value.

    There are two types of clipping:
      1) Cull triangles that are completely outside the view frustum.  This is purely
         to save computation by reducing the number of triangles that need to be
         rasterized.
      2) Clip triangles into the portion of the triangle where z > z_clip_value. The
         clipped region may be a quadrilateral, which results in splitting a triangle
         into two triangles. This does not save computation, but is necessary to
         correctly rasterize using perspective cameras for triangles that pass through
         z <= 0, because NDC/screen coordinates go to infinity at z=0.

    Args:
        face_verts_unclipped: An (F, 3, 3) tensor, where F is the number of faces in
            the packed representation of Meshes, the 2nd dimension represents the 3 vertices
            of the triangle, and the 3rd dimension stores the xyz locations of each
            vertex.  The z-coordinates must be represented in world coordinates, while
            the xy-coordinates may be in NDC/screen coordinates
        mesh_to_face_first_idx: an tensor of shape (N,), where N is the number of meshes
            in the batch.  The ith element stores the index into face_verts_unclipped
            of the first face of the ith mesh.
        num_faces_per_mesh: a tensor of shape (N,) storing the number of faces in each mesh.
        frustum: a ClipFrustum object defining the frustum used to cull faces.

    Returns:
        clipped_faces: ClippedFaces object storing a clipped version of the Meshes
            along with tensors that can be used to convert barycentric coordinates
            returned by rasterization of the clipped meshes into a barycentric
            coordinates for the unclipped meshes.
    """
    F = face_verts_unclipped.shape[0]
    device = face_verts_unclipped.device

    # Triangles completely outside the view frustum will be culled
    # faces_culled is of shape (F, )
    faces_culled = _get_culled_faces(face_verts_unclipped, frustum)

    # Triangles that are partially behind the z clipping plane will be clipped to
    # smaller triangles
    z_clip_value = frustum.z_clip_value
    perspective_correct = frustum.perspective_correct
    if z_clip_value is not None:
        # (F, 3) tensor (where F is the number of triangles) marking whether each vertex
        # in a triangle is behind the clipping plane
        faces_clipped_verts = face_verts_unclipped[:, :, 2] < z_clip_value

        # (F) dim tensor containing the number of clipped vertices in each triangle
        faces_num_clipped_verts = faces_clipped_verts.sum(1)
    else:
        faces_num_clipped_verts = torch.zeros([F], device=device)

    # If no triangles need to be clipped or culled, avoid unnecessary computation
    # and return early
    if faces_num_clipped_verts.sum().item() == 0 and faces_culled.sum().item() == 0:
        return ClippedFaces(
            face_verts=face_verts_unclipped,
            mesh_to_face_first_idx=mesh_to_face_first_idx,
            num_faces_per_mesh=num_faces_per_mesh,
        )

    #####################################################################################
    # Classify faces into the 4 relevant cases:
    #   1) The triangle is completely in front of the clipping plane (it is left
    #      unchanged)
    #   2) The triangle is completely behind the clipping plane (it is culled)
    #   3) The triangle has exactly two vertices behind the clipping plane (it is
    #      clipped into a smaller triangle)
    #   4) The triangle has exactly one vertex behind the clipping plane (it is clipped
    #      into a smaller quadrilateral and split into two triangles)
    #####################################################################################

    faces_unculled = ~faces_culled
    # Case 1:  no clipped verts or culled faces
    cases1_unclipped = (faces_num_clipped_verts == 0) & faces_unculled
    case1_unclipped_idx = cases1_unclipped.nonzero(as_tuple=True)[0]
    # Case 2: all verts clipped
    case2_unclipped = (faces_num_clipped_verts == 3) | faces_culled
    # Case 3: two verts clipped
    case3_unclipped = (faces_num_clipped_verts == 2) & faces_unculled
    case3_unclipped_idx = case3_unclipped.nonzero(as_tuple=True)[0]
    # Case 4: one vert clipped
    case4_unclipped = (faces_num_clipped_verts == 1) & faces_unculled
    case4_unclipped_idx = case4_unclipped.nonzero(as_tuple=True)[0]

    # faces_unclipped_to_clipped_idx is an (F) dim tensor storing the index of each
    # face to the corresponding face in face_verts_clipped.
    # Each case 2 triangle will be culled (deleted from face_verts_clipped),
    # while each case 4 triangle will be split into two smaller triangles
    # (replaced by two consecutive triangles in face_verts_clipped)

    # case2_unclipped is an (F,) dim 0/1 tensor of all the case2 faces
    # case4_unclipped is an (F,) dim 0/1 tensor of all the case4 faces
    faces_delta = case4_unclipped.int() - case2_unclipped.int()
    # faces_delta_cum gives the per face change in index. Faces which are
    # clipped in the original mesh are mapped to the closest non clipped face
    # in face_verts_clipped (this doesn't matter as they are not used
    # during rasterization anyway).
    faces_delta_cum = faces_delta.cumsum(0) - faces_delta
    delta = 1 + case4_unclipped.int() - case2_unclipped.int()
    faces_unclipped_to_clipped_idx = delta.cumsum(0) - delta

    ###########################################
    # Allocate tensors for the output Meshes.
    # These will then be filled in for each case.
    ###########################################
    F_clipped = (
        F
        # pyre-fixme[58]: `+` is not supported for operand types `int` and
        #  `Union[bool, float, int]`.
        + faces_delta_cum[-1].item()
        # pyre-fixme[58]: `+` is not supported for operand types `int` and
        #  `Union[bool, float, int]`.
        + faces_delta[-1].item()
    )  # Total number of faces in the new Meshes
    face_verts_clipped = torch.zeros(
        (F_clipped, 3, 3), dtype=face_verts_unclipped.dtype, device=device
    )
    faces_clipped_to_unclipped_idx = torch.zeros(
        [F_clipped], dtype=torch.int64, device=device
    )

    # Update version of mesh_to_face_first_idx and num_faces_per_mesh applicable to
    # face_verts_clipped
    mesh_to_face_first_idx_clipped = faces_unclipped_to_clipped_idx[
        mesh_to_face_first_idx
    ]
    F_clipped_t = torch.full([1], F_clipped, dtype=torch.int64, device=device)
    num_faces_next = torch.cat((mesh_to_face_first_idx_clipped[1:], F_clipped_t))
    num_faces_per_mesh_clipped = num_faces_next - mesh_to_face_first_idx_clipped

    ################# Start Case 1 ########################################

    # Case 1: Triangles are fully visible, copy unchanged triangles into the
    # appropriate position in the new list of faces
    case1_clipped_idx = faces_unclipped_to_clipped_idx[case1_unclipped_idx]
    face_verts_clipped[case1_clipped_idx] = face_verts_unclipped[case1_unclipped_idx]
    faces_clipped_to_unclipped_idx[case1_clipped_idx] = case1_unclipped_idx

    # If no triangles need to be clipped but some triangles were culled, avoid
    # unnecessary clipping computation
    if case3_unclipped_idx.shape[0] + case4_unclipped_idx.shape[0] == 0:
        return ClippedFaces(
            face_verts=face_verts_clipped,
            mesh_to_face_first_idx=mesh_to_face_first_idx_clipped,
            num_faces_per_mesh=num_faces_per_mesh_clipped,
            faces_clipped_to_unclipped_idx=faces_clipped_to_unclipped_idx,
        )

    ################# End Case 1 ##########################################

    ################# Start Case 3 ########################################

    # Case 3: exactly two vertices are behind the camera, clipping the triangle into a
    # triangle.  In the diagram below, we clip the bottom part of the triangle, and add
    # new vertices p4 and p5 by intersecting with the clipping plane.  The updated
    # triangle is the triangle between p4, p1, p5
    #
    #                   p1  (unclipped vertex)
    #                   /\
    #                  /  \
    #                 /  t \
    # _____________p4/______\p5__________ clip_value
    # xxxxxxxxxxxxxx/        \xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxx/____      \xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxx p2 xxxx---____\p3 xxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    faces_case3 = face_verts_unclipped[case3_unclipped_idx]

    # index (0, 1, or 2) of the vertex in front of the clipping plane
    p1_face_ind = torch.where(~faces_clipped_verts[case3_unclipped_idx])[1]

    # Solve for the points p4, p5 that intersect the clipping plane
    p, p_barycentric = _find_verts_intersecting_clipping_plane(
        faces_case3, p1_face_ind, z_clip_value, perspective_correct
    )

    p1, _, _, p4, p5 = p
    p1_barycentric, _, _, p4_barycentric, p5_barycentric = p_barycentric

    # Store clipped triangle
    case3_clipped_idx = faces_unclipped_to_clipped_idx[case3_unclipped_idx]
    t_barycentric = torch.stack((p4_barycentric, p5_barycentric, p1_barycentric), 2)
    face_verts_clipped[case3_clipped_idx] = torch.stack((p4, p5, p1), 1)
    faces_clipped_to_unclipped_idx[case3_clipped_idx] = case3_unclipped_idx

    ################# End Case 3 ##########################################

    ################# Start Case 4 ########################################

    # Case 4: exactly one vertex is behind the camera, clip the triangle into a
    # quadrilateral.  In the diagram below, we clip the bottom part of the triangle,
    # and add new vertices p4 and p5 by intersecting with the cliiping plane.  The
    # unclipped region is a quadrilateral, which is split into two triangles:
    #   t1: p4, p2, p5
    #   t2: p5, p2, p3
    #
    #            p3_____________________p2
    #              \               __--/
    #               \    t2    __--   /
    #                \     __--  t1  /
    # ______________p5\__--_________/p4_________clip_value
    # xxxxxxxxxxxxxxxxx\           /xxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxxxxx\         /xxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxxxxxx\       /xxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxxxxxxx\     /xxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxxxxxxxx\   /xxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxxxxxxxxx\ /xxxxxxxxxxxxxxxxxxxxx
    #                      p1 (clipped vertex)

    faces_case4 = face_verts_unclipped[case4_unclipped_idx]

    # index (0, 1, or 2) of the vertex behind the clipping plane
    p1_face_ind = torch.where(faces_clipped_verts[case4_unclipped_idx])[1]

    # Solve for the points p4, p5 that intersect the clipping plane
    p, p_barycentric = _find_verts_intersecting_clipping_plane(
        faces_case4, p1_face_ind, z_clip_value, perspective_correct
    )
    _, p2, p3, p4, p5 = p
    _, p2_barycentric, p3_barycentric, p4_barycentric, p5_barycentric = p_barycentric

    # Store clipped triangles
    case4_clipped_idx = faces_unclipped_to_clipped_idx[case4_unclipped_idx]
    face_verts_clipped[case4_clipped_idx] = torch.stack((p4, p2, p5), 1)
    face_verts_clipped[case4_clipped_idx + 1] = torch.stack((p5, p2, p3), 1)
    t1_barycentric = torch.stack((p4_barycentric, p2_barycentric, p5_barycentric), 2)
    t2_barycentric = torch.stack((p5_barycentric, p2_barycentric, p3_barycentric), 2)
    faces_clipped_to_unclipped_idx[case4_clipped_idx] = case4_unclipped_idx
    faces_clipped_to_unclipped_idx[case4_clipped_idx + 1] = case4_unclipped_idx

    ##################### End Case 4 #########################

    # Triangles that were clipped (case 3 & case 4) will require conversion of
    # barycentric coordinates from being in terms of the smaller clipped triangle to in terms
    # of the original big triangle.  If there are T clipped triangles,
    # barycentric_conversion is a (T, 3, 3) tensor, where barycentric_conversion[i, :, k]
    # stores the barycentric weights in terms of the world coordinates of the original
    # (big) triangle for the kth vertex in the clipped (small) triangle.  If our
    # rasterizer then expresses some NDC coordinate in terms of barycentric
    # world coordinates for the clipped (small) triangle as alpha_clipped[i,:],
    #   alpha_unclipped[i, :] = barycentric_conversion[i, :, :]*alpha_clipped[i, :]
    barycentric_conversion = torch.cat((t_barycentric, t1_barycentric, t2_barycentric))

    # faces_clipped_to_conversion_idx is an (F_clipped,) shape tensor mapping each output
    # face to the applicable row of barycentric_conversion (or set to -1 if conversion is
    # not needed)
    faces_to_convert_idx = torch.cat(
        (case3_clipped_idx, case4_clipped_idx, case4_clipped_idx + 1), 0
    )
    barycentric_idx = torch.arange(
        barycentric_conversion.shape[0], dtype=torch.int64, device=device
    )
    faces_clipped_to_conversion_idx = torch.full(
        [F_clipped], -1, dtype=torch.int64, device=device
    )
    faces_clipped_to_conversion_idx[faces_to_convert_idx] = barycentric_idx

    # clipped_faces_quadrilateral_ind is an (F_clipped) dim tensor
    # For case 4 clipped triangles (where a big triangle is split in two smaller triangles),
    # store the index of the neighboring clipped triangle.
    # This will be needed because if the soft rasterizer includes both
    # triangles in the list of top K nearest triangles, we
    # should only use the one with the smaller distance.
    clipped_faces_neighbor_idx = torch.full(
        [F_clipped], -1, dtype=torch.int64, device=device
    )
    clipped_faces_neighbor_idx[case4_clipped_idx] = case4_clipped_idx + 1
    clipped_faces_neighbor_idx[case4_clipped_idx + 1] = case4_clipped_idx

    clipped_faces = ClippedFaces(
        face_verts=face_verts_clipped,
        mesh_to_face_first_idx=mesh_to_face_first_idx_clipped,
        num_faces_per_mesh=num_faces_per_mesh_clipped,
        faces_clipped_to_unclipped_idx=faces_clipped_to_unclipped_idx,
        barycentric_conversion=barycentric_conversion,
        faces_clipped_to_conversion_idx=faces_clipped_to_conversion_idx,
        clipped_faces_neighbor_idx=clipped_faces_neighbor_idx,
    )
    return clipped_faces


def convert_clipped_rasterization_to_original_faces(
    pix_to_face_clipped, bary_coords_clipped, clipped_faces: ClippedFaces
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert rasterization Fragments (expressed as pix_to_face_clipped,
    bary_coords_clipped, dists_clipped) of clipped Meshes computed using clip_faces()
    to the corresponding rasterization Fragments where barycentric coordinates and
    face indices are in terms of the original unclipped Meshes. The distances are
    handled in the rasterizer C++/CUDA kernels (i.e. for Cases 1/3 the distance
    can be used directly and for Case 4 triangles the distance of the pixel to
    the closest of the two subdivided triangles is used).

    Args:
        pix_to_face_clipped: LongTensor of shape (N, image_size, image_size,
            faces_per_pixel) giving the indices of the nearest faces at each pixel,
            sorted in ascending z-order. Concretely
            ``pix_to_face_clipped[n, y, x, k] = f`` means that ``faces_verts_clipped[f]``
            is the kth closest face (in the z-direction) to pixel (y, x). Pixels that
            are hit by fewer than faces_per_pixel are padded with -1.
        bary_coords_clipped: FloatTensor of shape
            (N, image_size, image_size, faces_per_pixel, 3) giving the barycentric
            coordinates in world coordinates of the nearest faces at each pixel, sorted
            in ascending z-order.  Concretely, if ``pix_to_face_clipped[n, y, x, k] = f``
            then ``[w0, w1, w2] = bary_coords_clipped[n, y, x, k]`` gives the
            barycentric coords for pixel (y, x) relative to the face defined by
            ``unproject(face_verts_clipped[f])``. Pixels hit by fewer than
            faces_per_pixel are padded with -1.
        clipped_faces: an instance of ClippedFaces class giving the auxillary variables
            for converting rasterization outputs from clipped to unclipped Meshes.

    Returns:
        3-tuple: (pix_to_face_unclipped, bary_coords_unclipped, dists_unclipped) that
        have the same definition as (pix_to_face_clipped, bary_coords_clipped,
        dists_clipped) except that they pertain to faces_verts_unclipped instead of
        faces_verts_clipped (i.e the original meshes as opposed to the modified meshes)
    """
    faces_clipped_to_unclipped_idx = clipped_faces.faces_clipped_to_unclipped_idx

    # If no clipping then return inputs
    if (
        faces_clipped_to_unclipped_idx is None
        or faces_clipped_to_unclipped_idx.numel() == 0
    ):
        return pix_to_face_clipped, bary_coords_clipped

    device = pix_to_face_clipped.device

    # Convert pix_to_face indices to now refer to the faces in the unclipped Meshes.
    # Init empty tensor to fill in all the background values which have pix_to_face=-1.
    empty = torch.full(pix_to_face_clipped.shape, -1, device=device, dtype=torch.int64)
    pix_to_face_unclipped = torch.where(
        pix_to_face_clipped != -1,
        faces_clipped_to_unclipped_idx[pix_to_face_clipped],
        empty,
    )

    # For triangles that were clipped into smaller triangle(s), convert barycentric
    # coordinates from being in terms of the clipped triangle to being in terms of the
    # original unclipped triangle.

    # barycentric_conversion is a (T, 3, 3) tensor such that
    # alpha_unclipped[i, :] = barycentric_conversion[i, :, :]*alpha_clipped[i, :]
    barycentric_conversion = clipped_faces.barycentric_conversion

    # faces_clipped_to_conversion_idx is an (F_clipped,) shape tensor mapping each output
    # face to the applicable row of barycentric_conversion (or set to -1 if conversion is
    # not needed)
    faces_clipped_to_conversion_idx = clipped_faces.faces_clipped_to_conversion_idx

    if barycentric_conversion is not None:
        bary_coords_unclipped = bary_coords_clipped.clone()

        # Select the subset of faces that require conversion, where N is the sum
        # number of case3/case4 triangles that are in the closest k triangles to some
        # rasterized pixel.
        pix_to_conversion_idx = torch.where(
            pix_to_face_clipped != -1,
            faces_clipped_to_conversion_idx[pix_to_face_clipped],
            empty,
        )
        faces_to_convert_mask = pix_to_conversion_idx != -1
        N = faces_to_convert_mask.sum().item()

        # Expand to (N, H, W, K, 3) to be the same shape as barycentric coordinates
        faces_to_convert_mask_expanded = faces_to_convert_mask[:, :, :, :, None].expand(
            -1, -1, -1, -1, 3
        )

        # An (N,) dim tensor of indices into barycentric_conversion
        conversion_idx_subset = pix_to_conversion_idx[faces_to_convert_mask]

        # An (N, 3, 1) tensor of barycentric coordinates in terms of the clipped triangles
        bary_coords_clipped_subset = bary_coords_clipped[faces_to_convert_mask_expanded]
        bary_coords_clipped_subset = bary_coords_clipped_subset.reshape((N, 3, 1))

        # An (N, 3, 3) tensor storing matrices to convert from clipped to unclipped
        # barycentric coordinates
        bary_conversion_subset = barycentric_conversion[conversion_idx_subset]

        # An (N, 3, 1) tensor of barycentric coordinates in terms of the unclipped triangle
        bary_coords_unclipped_subset = bary_conversion_subset.bmm(
            bary_coords_clipped_subset
        )

        bary_coords_unclipped_subset = bary_coords_unclipped_subset.reshape([N * 3])
        bary_coords_unclipped[
            faces_to_convert_mask_expanded
        ] = bary_coords_unclipped_subset

        # dists for case 4 faces will be handled in the rasterizer
        # so no need to modify them here.
    else:
        bary_coords_unclipped = bary_coords_clipped

    return pix_to_face_unclipped, bary_coords_unclipped
