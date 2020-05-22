# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


from typing import Optional

import numpy as np
import torch
from pytorch3d import _C


# TODO make the epsilon user configurable
kEpsilon = 1e-8

# Maxinum number of faces per bins for
# coarse-to-fine rasterization
kMaxFacesPerBin = 22


def rasterize_meshes(
    meshes,
    image_size: int = 256,
    blur_radius: float = 0.0,
    faces_per_pixel: int = 8,
    bin_size: Optional[int] = None,
    max_faces_per_bin: Optional[int] = None,
    perspective_correct: bool = False,
    cull_backfaces: bool = False,
):
    """
    Rasterize a batch of meshes given the shape of the desired output image.
    Each mesh is rasterized onto a separate image of shape
    (image_size, image_size).

    Args:
        meshes: A Meshes object representing a batch of meshes, batch size N.
        image_size: Size in pixels of the output raster image for each mesh
            in the batch. Assumes square images.
        blur_radius: Float distance in the range [0, 2] used to expand the face
            bounding boxes for rasterization. Setting blur radius
            results in blurred edges around the shape instead of a
            hard boundary. Set to 0 for no blur.
        faces_per_pixel (Optional): Number of faces to save per pixel, returning
            the nearest faces_per_pixel points along the z-axis.
        bin_size: Size of bins to use for coarse-to-fine rasterization. Setting
            bin_size=0 uses naive rasterization; setting bin_size=None attempts to
            set it heuristically based on the shape of the input. This should not
            affect the output, but can affect the speed of the forward pass.
        faces_per_bin: Only applicable when using coarse-to-fine rasterization
            (bin_size > 0); this is the maxiumum number of faces allowed within each
            bin. If more than this many faces actually fall into a bin, an error
            will be raised. This should not affect the output values, but can affect
            the memory usage in the forward pass.
        perspective_correct: Bool, Whether to apply perspective correction when computing
            barycentric coordinates for pixels.
        cull_backfaces: Bool, Whether to only rasterize mesh faces which are
            visible to the camera.  This assumes that vertices of
            front-facing triangles are ordered in an anti-clockwise
            fashion, and triangles that face away from the camera are
            in a clockwise order relative to the current view
            direction. NOTE: This will only work if the mesh faces are
            consistently defined with counter-clockwise ordering when
            viewed from the outside.

    Returns:
        4-element tuple containing

        - **pix_to_face**: LongTensor of shape
          (N, image_size, image_size, faces_per_pixel)
          giving the indices of the nearest faces at each pixel,
          sorted in ascending z-order.
          Concretely ``pix_to_face[n, y, x, k] = f`` means that
          ``faces_verts[f]`` is the kth closest face (in the z-direction)
          to pixel (y, x). Pixels that are hit by fewer than
          faces_per_pixel are padded with -1.
        - **zbuf**: FloatTensor of shape (N, image_size, image_size, faces_per_pixel)
          giving the NDC z-coordinates of the nearest faces at each pixel,
          sorted in ascending z-order.
          Concretely, if ``pix_to_face[n, y, x, k] = f`` then
          ``zbuf[n, y, x, k] = face_verts[f, 2]``. Pixels hit by fewer than
          faces_per_pixel are padded with -1.
        - **barycentric**: FloatTensor of shape
          (N, image_size, image_size, faces_per_pixel, 3)
          giving the barycentric coordinates in NDC units of the
          nearest faces at each pixel, sorted in ascending z-order.
          Concretely, if ``pix_to_face[n, y, x, k] = f`` then
          ``[w0, w1, w2] = barycentric[n, y, x, k]`` gives
          the barycentric coords for pixel (y, x) relative to the face
          defined by ``face_verts[f]``. Pixels hit by fewer than
          faces_per_pixel are padded with -1.
        - **pix_dists**: FloatTensor of shape
          (N, image_size, image_size, faces_per_pixel)
          giving the signed Euclidean distance (in NDC units) in the
          x/y plane of each point closest to the pixel. Concretely if
          ``pix_to_face[n, y, x, k] = f`` then ``pix_dists[n, y, x, k]`` is the
          squared distance between the pixel (y, x) and the face given
          by vertices ``face_verts[f]``. Pixels hit with fewer than
          ``faces_per_pixel`` are padded with -1.
    """
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    face_verts = verts_packed[faces_packed]
    mesh_to_face_first_idx = meshes.mesh_to_faces_packed_first_idx()
    num_faces_per_mesh = meshes.num_faces_per_mesh()

    # TODO: Choose naive vs coarse-to-fine based on mesh size and image size.
    if bin_size is None:
        if not verts_packed.is_cuda:
            # Binned CPU rasterization is not supported.
            bin_size = 0
        else:
            # TODO better heuristics for bin size.
            if image_size <= 64:
                bin_size = 8
            else:
                # Heuristic based formula maps image_size -> bin_size as follows:
                # image_size < 64 -> 8
                # 16 < image_size < 256 -> 16
                # 256 < image_size < 512 -> 32
                # 512 < image_size < 1024 -> 64
                # 1024 < image_size < 2048 -> 128
                bin_size = int(2 ** max(np.ceil(np.log2(image_size)) - 4, 4))

    if bin_size != 0:
        # There is a limit on the number of faces per bin in the cuda kernel.
        faces_per_bin = 1 + (image_size - 1) // bin_size
        if faces_per_bin >= kMaxFacesPerBin:
            raise ValueError(
                "bin_size too small, number of faces per bin must be less than %d; got %d"
                % (kMaxFacesPerBin, faces_per_bin)
            )

    if max_faces_per_bin is None:
        max_faces_per_bin = int(max(10000, verts_packed.shape[0] / 5))

    # pyre-fixme[16]: `_RasterizeFaceVerts` has no attribute `apply`.
    return _RasterizeFaceVerts.apply(
        face_verts,
        mesh_to_face_first_idx,
        num_faces_per_mesh,
        image_size,
        blur_radius,
        faces_per_pixel,
        bin_size,
        max_faces_per_bin,
        perspective_correct,
        cull_backfaces,
    )


class _RasterizeFaceVerts(torch.autograd.Function):
    """
    Torch autograd wrapper for forward and backward pass of rasterize_meshes
    implemented in C++/CUDA.

    Args:
        face_verts: Tensor of shape (F, 3, 3) giving (packed) vertex positions
            for faces in all the meshes in the batch. Concretely,
            face_verts[f, i] = [x, y, z] gives the coordinates for the
            ith vertex of the fth face. These vertices are expected to
            be in NDC coordinates in the range [-1, 1].
        mesh_to_face_first_idx: LongTensor of shape (N) giving the index in
            faces_verts of the first face in each mesh in
            the batch.
        num_faces_per_mesh: LongTensor of shape (N) giving the number of faces
            for each mesh in the batch.
        image_size, blur_radius, faces_per_pixel: same as rasterize_meshes.
        perspective_correct: same as rasterize_meshes.
        cull_backfaces: same as rasterize_meshes.

    Returns:
        same as rasterize_meshes function.
    """

    @staticmethod
    def forward(
        ctx,
        face_verts,
        mesh_to_face_first_idx,
        num_faces_per_mesh,
        image_size: int = 256,
        blur_radius: float = 0.01,
        faces_per_pixel: int = 0,
        bin_size: int = 0,
        max_faces_per_bin: int = 0,
        perspective_correct: bool = False,
        cull_backfaces: bool = False,
    ):
        # pyre-fixme[16]: Module `pytorch3d` has no attribute `_C`.
        pix_to_face, zbuf, barycentric_coords, dists = _C.rasterize_meshes(
            face_verts,
            mesh_to_face_first_idx,
            num_faces_per_mesh,
            image_size,
            blur_radius,
            faces_per_pixel,
            bin_size,
            max_faces_per_bin,
            perspective_correct,
            cull_backfaces,
        )
        ctx.save_for_backward(face_verts, pix_to_face)
        ctx.mark_non_differentiable(pix_to_face)
        ctx.perspective_correct = perspective_correct
        return pix_to_face, zbuf, barycentric_coords, dists

    @staticmethod
    def backward(ctx, grad_pix_to_face, grad_zbuf, grad_barycentric_coords, grad_dists):
        grad_face_verts = None
        grad_mesh_to_face_first_idx = None
        grad_num_faces_per_mesh = None
        grad_image_size = None
        grad_radius = None
        grad_faces_per_pixel = None
        grad_bin_size = None
        grad_max_faces_per_bin = None
        grad_perspective_correct = None
        grad_cull_backfaces = None
        face_verts, pix_to_face = ctx.saved_tensors
        grad_face_verts = _C.rasterize_meshes_backward(
            face_verts,
            pix_to_face,
            grad_zbuf,
            grad_barycentric_coords,
            grad_dists,
            ctx.perspective_correct,
        )
        grads = (
            grad_face_verts,
            grad_mesh_to_face_first_idx,
            grad_num_faces_per_mesh,
            grad_image_size,
            grad_radius,
            grad_faces_per_pixel,
            grad_bin_size,
            grad_max_faces_per_bin,
            grad_perspective_correct,
            grad_cull_backfaces,
        )
        return grads


def pix_to_ndc(i, S):
    # NDC x-offset + (i * pixel_width + half_pixel_width)
    return -1 + (2 * i + 1.0) / S


def rasterize_meshes_python(
    meshes,
    image_size: int = 256,
    blur_radius: float = 0.0,
    faces_per_pixel: int = 8,
    perspective_correct: bool = False,
    cull_backfaces: bool = False,
):
    """
    Naive PyTorch implementation of mesh rasterization with the same inputs and
    outputs as the rasterize_meshes function.

    This function is not optimized and is implemented as a comparison for the
    C++/CUDA implementations.
    """
    N = len(meshes)
    # Assume only square images.
    # TODO(T52813608) extend support for non-square images.
    H, W = image_size, image_size
    K = faces_per_pixel
    device = meshes.device

    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    faces_verts = verts_packed[faces_packed]
    mesh_to_face_first_idx = meshes.mesh_to_faces_packed_first_idx()
    num_faces_per_mesh = meshes.num_faces_per_mesh()

    # Intialize output tensors.
    face_idxs = torch.full(
        (N, H, W, K), fill_value=-1, dtype=torch.int64, device=device
    )
    zbuf = torch.full((N, H, W, K), fill_value=-1, dtype=torch.float32, device=device)
    bary_coords = torch.full(
        (N, H, W, K, 3), fill_value=-1, dtype=torch.float32, device=device
    )
    pix_dists = torch.full(
        (N, H, W, K), fill_value=-1, dtype=torch.float32, device=device
    )

    # Calculate all face bounding boxes.
    # pyre-fixme[16]: `Tuple` has no attribute `values`.
    x_mins = torch.min(faces_verts[:, :, 0], dim=1, keepdim=True).values
    x_maxs = torch.max(faces_verts[:, :, 0], dim=1, keepdim=True).values
    y_mins = torch.min(faces_verts[:, :, 1], dim=1, keepdim=True).values
    y_maxs = torch.max(faces_verts[:, :, 1], dim=1, keepdim=True).values

    # Expand by blur radius.
    x_mins = x_mins - np.sqrt(blur_radius) - kEpsilon
    x_maxs = x_maxs + np.sqrt(blur_radius) + kEpsilon
    y_mins = y_mins - np.sqrt(blur_radius) - kEpsilon
    y_maxs = y_maxs + np.sqrt(blur_radius) + kEpsilon

    # Loop through meshes in the batch.
    for n in range(N):
        face_start_idx = mesh_to_face_first_idx[n]
        face_stop_idx = face_start_idx + num_faces_per_mesh[n]

        # Iterate through the horizontal lines of the image from top to bottom.
        for yi in range(H):
            # Y coordinate of one end of the image. Reverse the ordering
            # of yi so that +Y is pointing up in the image.
            yfix = H - 1 - yi
            yf = pix_to_ndc(yfix, H)

            # Iterate through pixels on this horizontal line, left to right.
            for xi in range(W):
                # X coordinate of one end of the image. Reverse the ordering
                # of xi so that +X is pointing to the left in the image.
                xfix = W - 1 - xi
                xf = pix_to_ndc(xfix, W)
                top_k_points = []

                # Check whether each face in the mesh affects this pixel.
                for f in range(face_start_idx, face_stop_idx):
                    face = faces_verts[f].squeeze()
                    v0, v1, v2 = face.unbind(0)

                    face_area = edge_function(v0, v1, v2)

                    # Ignore triangles facing away from the camera.
                    back_face = face_area < 0
                    if cull_backfaces and back_face:
                        continue

                    # Ignore faces which have zero area.
                    if face_area == 0.0:
                        continue

                    outside_bbox = (
                        xf < x_mins[f]
                        or xf > x_maxs[f]
                        or yf < y_mins[f]
                        or yf > y_maxs[f]
                    )

                    # Check if pixel is outside of face bbox.
                    if outside_bbox:
                        continue

                    # Compute barycentric coordinates and pixel z distance.
                    pxy = torch.tensor([xf, yf], dtype=torch.float32, device=device)

                    bary = barycentric_coordinates(pxy, v0[:2], v1[:2], v2[:2])
                    if perspective_correct:
                        z0, z1, z2 = v0[2], v1[2], v2[2]
                        l0, l1, l2 = bary[0], bary[1], bary[2]
                        top0 = l0 * z1 * z2
                        top1 = z0 * l1 * z2
                        top2 = z0 * z1 * l2
                        bot = top0 + top1 + top2
                        bary = torch.stack([top0 / bot, top1 / bot, top2 / bot])
                    pz = bary[0] * v0[2] + bary[1] * v1[2] + bary[2] * v2[2]

                    # Check if point is behind the image.
                    if pz < 0:
                        continue

                    # Calculate signed 2D distance from point to face.
                    # Points inside the triangle have negative distance.
                    dist = point_triangle_distance(pxy, v0[:2], v1[:2], v2[:2])
                    inside = all(x > 0.0 for x in bary)

                    signed_dist = dist * -1.0 if inside else dist

                    # Add an epsilon to prevent errors when comparing distance
                    # to blur radius.
                    if not inside and dist >= blur_radius:
                        continue

                    top_k_points.append((pz, f, bary, signed_dist))
                    top_k_points.sort()
                    if len(top_k_points) > K:
                        top_k_points = top_k_points[:K]

                # Save to output tensors.
                for k, (pz, f, bary, dist) in enumerate(top_k_points):
                    zbuf[n, yi, xi, k] = pz
                    face_idxs[n, yi, xi, k] = f
                    bary_coords[n, yi, xi, k, 0] = bary[0]
                    bary_coords[n, yi, xi, k, 1] = bary[1]
                    bary_coords[n, yi, xi, k, 2] = bary[2]
                    pix_dists[n, yi, xi, k] = dist

    return face_idxs, zbuf, bary_coords, pix_dists


def edge_function(p, v0, v1):
    r"""
    Determines whether a point p is on the right side of a 2D line segment
    given by the end points v0, v1.

    Args:
        p: (x, y) Coordinates of a point.
        v0, v1: (x, y) Coordinates of the end points of the edge.

    Returns:
        area: The signed area of the parallelogram given by the vectors

              .. code-block:: python

                  B = p - v0
                  A = v1 - v0

                        v1 ________
                          /\      /
                      A  /  \    /
                        /    \  /
                    v0 /______\/
                          B    p

             The area can also be interpreted as the cross product A x B.
             If the sign of the area is positive, the point p is on the
             right side of the edge. Negative area indicates the point is on
             the left side of the edge. i.e. for an edge v1 - v0

             .. code-block:: python

                             v1
                            /
                           /
                    -     /    +
                         /
                        /
                      v0
    """
    return (p[0] - v0[0]) * (v1[1] - v0[1]) - (p[1] - v0[1]) * (v1[0] - v0[0])


def barycentric_coordinates(p, v0, v1, v2):
    """
    Compute the barycentric coordinates of a point relative to a triangle.

    Args:
        p: Coordinates of a point.
        v0, v1, v2: Coordinates of the triangle vertices.

    Returns
        bary: (w0, w1, w2) barycentric coordinates in the range [0, 1].
    """
    area = edge_function(v2, v0, v1) + kEpsilon  # 2 x face area.
    w0 = edge_function(p, v1, v2) / area
    w1 = edge_function(p, v2, v0) / area
    w2 = edge_function(p, v0, v1) / area
    return (w0, w1, w2)


def point_line_distance(p, v0, v1):
    """
    Return minimum distance between line segment (v1 - v0) and point p.

    Args:
        p: Coordinates of a point.
        v0, v1: Coordinates of the end points of the line segment.

    Returns:
        non-square distance to the boundary of the triangle.

    Consider the line extending the segment - this can be parameterized as
    ``v0 + t (v1 - v0)``.

    First find the projection of point p onto the line. It falls where
    ``t = [(p - v0) . (v1 - v0)] / |v1 - v0|^2``
    where . is the dot product.

    The parameter t is clamped from [0, 1] to handle points outside the
    segment (v1 - v0).

    Once the projection of the point on the segment is known, the distance from
    p to the projection gives the minimum distance to the segment.
    """
    if p.shape != v0.shape != v1.shape:
        raise ValueError("All points must have the same number of coordinates")

    v1v0 = v1 - v0
    l2 = v1v0.dot(v1v0)  # |v1 - v0|^2
    if l2 <= kEpsilon:
        return (p - v1).dot(p - v1)  # v0 == v1

    t = v1v0.dot(p - v0) / l2
    t = torch.clamp(t, min=0.0, max=1.0)
    p_proj = v0 + t * v1v0
    delta_p = p_proj - p
    return delta_p.dot(delta_p)


def point_triangle_distance(p, v0, v1, v2):
    """
    Return shortest distance between a point and a triangle.

    Args:
        p: Coordinates of a point.
        v0, v1, v2: Coordinates of the three triangle vertices.

    Returns:
        shortest absolute distance from the point to the triangle.
    """
    if p.shape != v0.shape != v1.shape != v2.shape:
        raise ValueError("All points must have the same number of coordinates")

    e01_dist = point_line_distance(p, v0, v1)
    e02_dist = point_line_distance(p, v0, v2)
    e12_dist = point_line_distance(p, v1, v2)
    edge_dists_min = torch.min(torch.min(e01_dist, e02_dist), e12_dist)

    return edge_dists_min
