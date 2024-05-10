# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from pytorch3d import _C

from ..utils import parse_image_size

from .clip import (
    clip_faces,
    ClipFrustum,
    convert_clipped_rasterization_to_original_faces,
)


# TODO make the epsilon user configurable
kEpsilon = 1e-8

# Maximum number of faces per bins for
# coarse-to-fine rasterization
kMaxFacesPerBin = 22


def rasterize_meshes(
    meshes,
    image_size: Union[int, List[int], Tuple[int, int]] = 256,
    blur_radius: float = 0.0,
    faces_per_pixel: int = 8,
    bin_size: Optional[int] = None,
    max_faces_per_bin: Optional[int] = None,
    perspective_correct: bool = False,
    clip_barycentric_coords: bool = False,
    cull_backfaces: bool = False,
    z_clip_value: Optional[float] = None,
    cull_to_frustum: bool = False,
):
    """
    Rasterize a batch of meshes given the shape of the desired output image.
    Each mesh is rasterized onto a separate image of shape
    (H, W) if `image_size` is a tuple or (image_size, image_size) if it
    is an int.

    If the desired image size is non square (i.e. a tuple of (H, W) where H != W)
    the aspect ratio needs special consideration. There are two aspect ratios
    to be aware of:
        - the aspect ratio of each pixel
        - the aspect ratio of the output image
    The camera can be used to set the pixel aspect ratio. In the rasterizer,
    we assume square pixels, but variable image aspect ratio (i.e rectangle images).

    In most cases you will want to set the camera aspect ratio to
    1.0 (i.e. square pixels) and only vary the
    `image_size` (i.e. the output image dimensions in pixels).

    Args:
        meshes: A Meshes object representing a batch of meshes, batch size N.
        image_size: Size in pixels of the output image to be rasterized.
            Can optionally be a tuple of (H, W) in the case of non square images.
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
        max_faces_per_bin: Only applicable when using coarse-to-fine rasterization
            (bin_size > 0); this is the maximum number of faces allowed within each
            bin. This should not affect the output values, but can affect
            the memory usage in the forward pass.
        perspective_correct: Bool, Whether to apply perspective correction when computing
            barycentric coordinates for pixels. This should be set to True if a perspective
            camera is used.
        clip_barycentric_coords: Whether, after any perspective correction is applied
            but before the depth is calculated (e.g. for z clipping),
            to "correct" a location outside the face (i.e. with a negative
            barycentric coordinate) to a position on the edge of the face.
        cull_backfaces: Bool, Whether to only rasterize mesh faces which are
            visible to the camera.  This assumes that vertices of
            front-facing triangles are ordered in an anti-clockwise
            fashion, and triangles that face away from the camera are
            in a clockwise order relative to the current view
            direction. NOTE: This will only work if the mesh faces are
            consistently defined with counter-clockwise ordering when
            viewed from the outside.
        z_clip_value: if not None, then triangles will be clipped (and possibly
            subdivided into smaller triangles) such that z >= z_clip_value.
            This avoids camera projections that go to infinity as z->0.
            Default is None as clipping affects rasterization speed and
            should only be turned on if explicitly needed.
            See clip.py for all the extra computation that is required.
        cull_to_frustum: if True, triangles outside the view frustum will be culled.
            Culling involves removing all faces which fall outside view frustum.
            Default is False so that it is turned on only when needed.

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

        In the case that image_size is a tuple of (H, W) then the outputs
        will be of shape `(N, H, W, ...)`.
    """
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    face_verts = verts_packed[faces_packed]
    mesh_to_face_first_idx = meshes.mesh_to_faces_packed_first_idx()
    num_faces_per_mesh = meshes.num_faces_per_mesh()

    # In the case that H != W use the max image size to set the bin_size
    # to accommodate the num bins constraint in the coarse rasterizer.
    # If the ratio of H:W is large this might cause issues as the smaller
    # dimension will have fewer bins.
    # TODO: consider a better way of setting the bin size.
    im_size = parse_image_size(image_size)
    max_image_size = max(*im_size)

    clipped_faces_neighbor_idx = None

    if z_clip_value is not None or cull_to_frustum:
        # Cull faces outside the view frustum, and clip faces that are partially
        # behind the camera into the portion of the triangle in front of the
        # camera.  This may change the number of faces
        frustum = ClipFrustum(
            left=-1,
            right=1,
            top=-1,
            bottom=1,
            perspective_correct=perspective_correct,
            z_clip_value=z_clip_value,
            cull=cull_to_frustum,
        )
        clipped_faces = clip_faces(
            face_verts, mesh_to_face_first_idx, num_faces_per_mesh, frustum=frustum
        )
        face_verts = clipped_faces.face_verts
        mesh_to_face_first_idx = clipped_faces.mesh_to_face_first_idx
        num_faces_per_mesh = clipped_faces.num_faces_per_mesh

        # For case 4 clipped triangles (where a big triangle is split in two smaller triangles),
        # need the index of the neighboring clipped triangle as only one can be in
        # in the top K closest faces in the rasterization step.
        clipped_faces_neighbor_idx = clipped_faces.clipped_faces_neighbor_idx

    if clipped_faces_neighbor_idx is None:
        # Set to the default which is all -1s.
        clipped_faces_neighbor_idx = torch.full(
            size=(face_verts.shape[0],),
            fill_value=-1,
            device=meshes.device,
            dtype=torch.int64,
        )

    # TODO: Choose naive vs coarse-to-fine based on mesh size and image size.
    if bin_size is None:
        if not verts_packed.is_cuda:
            # Binned CPU rasterization is not supported.
            bin_size = 0
        else:
            # TODO better heuristics for bin size.
            if max_image_size <= 64:
                bin_size = 8
            else:
                # Heuristic based formula maps max_image_size -> bin_size as follows:
                # max_image_size < 64 -> 8
                # 16 < max_image_size < 256 -> 16
                # 256 < max_image_size < 512 -> 32
                # 512 < max_image_size < 1024 -> 64
                # 1024 < max_image_size < 2048 -> 128
                bin_size = int(2 ** max(np.ceil(np.log2(max_image_size)) - 4, 4))

    if bin_size != 0:
        # There is a limit on the number of faces per bin in the cuda kernel.
        faces_per_bin = 1 + (max_image_size - 1) // bin_size
        if faces_per_bin >= kMaxFacesPerBin:
            raise ValueError(
                "bin_size too small, number of faces per bin must be less than %d; got %d"
                % (kMaxFacesPerBin, faces_per_bin)
            )

    if max_faces_per_bin is None:
        max_faces_per_bin = int(max(10000, meshes._F / 5))

    pix_to_face, zbuf, barycentric_coords, dists = _RasterizeFaceVerts.apply(
        face_verts,
        mesh_to_face_first_idx,
        num_faces_per_mesh,
        clipped_faces_neighbor_idx,
        im_size,
        blur_radius,
        faces_per_pixel,
        bin_size,
        max_faces_per_bin,
        perspective_correct,
        clip_barycentric_coords,
        cull_backfaces,
    )

    if z_clip_value is not None or cull_to_frustum:
        # If faces were clipped, map the rasterization result to be in terms of the
        # original unclipped faces.  This may involve converting barycentric
        # coordinates
        outputs = convert_clipped_rasterization_to_original_faces(
            pix_to_face,
            barycentric_coords,
            # pyre-fixme[61]: `clipped_faces` may not be initialized here.
            clipped_faces,
        )
        pix_to_face, barycentric_coords = outputs

    return pix_to_face, zbuf, barycentric_coords, dists


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
        face_verts: torch.Tensor,
        mesh_to_face_first_idx: torch.Tensor,
        num_faces_per_mesh: torch.Tensor,
        clipped_faces_neighbor_idx: torch.Tensor,
        image_size: Union[List[int], Tuple[int, int]] = (256, 256),
        blur_radius: float = 0.01,
        faces_per_pixel: int = 0,
        bin_size: int = 0,
        max_faces_per_bin: int = 0,
        perspective_correct: bool = False,
        clip_barycentric_coords: bool = False,
        cull_backfaces: bool = False,
        z_clip_value: Optional[float] = None,
        cull_to_frustum: bool = True,
    ):
        # pyre-fixme[16]: Module `pytorch3d` has no attribute `_C`.
        pix_to_face, zbuf, barycentric_coords, dists = _C.rasterize_meshes(
            face_verts,
            mesh_to_face_first_idx,
            num_faces_per_mesh,
            clipped_faces_neighbor_idx,
            image_size,
            blur_radius,
            faces_per_pixel,
            bin_size,
            max_faces_per_bin,
            perspective_correct,
            clip_barycentric_coords,
            cull_backfaces,
        )

        ctx.save_for_backward(face_verts, pix_to_face)
        ctx.mark_non_differentiable(pix_to_face)
        ctx.perspective_correct = perspective_correct
        ctx.clip_barycentric_coords = clip_barycentric_coords
        return pix_to_face, zbuf, barycentric_coords, dists

    @staticmethod
    def backward(ctx, grad_pix_to_face, grad_zbuf, grad_barycentric_coords, grad_dists):
        grad_face_verts = None
        grad_mesh_to_face_first_idx = None
        grad_num_faces_per_mesh = None
        grad_clipped_faces_neighbor_idx = None
        grad_image_size = None
        grad_radius = None
        grad_faces_per_pixel = None
        grad_bin_size = None
        grad_max_faces_per_bin = None
        grad_perspective_correct = None
        grad_clip_barycentric_coords = None
        grad_cull_backfaces = None
        face_verts, pix_to_face = ctx.saved_tensors
        grad_face_verts = _C.rasterize_meshes_backward(
            face_verts,
            pix_to_face,
            grad_zbuf,
            grad_barycentric_coords,
            grad_dists,
            ctx.perspective_correct,
            ctx.clip_barycentric_coords,
        )
        grads = (
            grad_face_verts,
            grad_mesh_to_face_first_idx,
            grad_num_faces_per_mesh,
            grad_clipped_faces_neighbor_idx,
            grad_image_size,
            grad_radius,
            grad_faces_per_pixel,
            grad_bin_size,
            grad_max_faces_per_bin,
            grad_perspective_correct,
            grad_clip_barycentric_coords,
            grad_cull_backfaces,
        )
        return grads


def non_square_ndc_range(S1, S2):
    """
    In the case of non square images, we scale the NDC range
    to maintain the aspect ratio. The smaller dimension has NDC
    range of 2.0.

    Args:
        S1: dimension along with the NDC range is needed
        S2: the other image dimension

    Returns:
        ndc_range: NDC range for dimension S1
    """
    ndc_range = 2.0
    if S1 > S2:
        ndc_range = (S1 / S2) * ndc_range
    return ndc_range


def pix_to_non_square_ndc(i, S1, S2):
    """
    The default value of the NDC range is [-1, 1].
    However in the case of non square images, we scale the NDC range
    to maintain the aspect ratio. The smaller dimension has NDC
    range from [-1, 1] and the other dimension is scaled by
    the ratio of H:W.
    e.g. for image size (H, W) = (64, 128)
       Height NDC range: [-1, 1]
       Width NDC range: [-2, 2]

    Args:
        i: pixel position on axes S1
        S1: dimension along with i is given
        S2: the other image dimension

    Returns:
        pixel: NDC coordinate of point i for dimension S1
    """
    # NDC: x-offset + (i * pixel_width + half_pixel_width)
    ndc_range = non_square_ndc_range(S1, S2)
    offset = ndc_range / 2.0
    return -offset + (ndc_range * i + offset) / S1


def rasterize_meshes_python(  # noqa: C901
    meshes,
    image_size: Union[int, Tuple[int, int]] = 256,
    blur_radius: float = 0.0,
    faces_per_pixel: int = 8,
    perspective_correct: bool = False,
    clip_barycentric_coords: bool = False,
    cull_backfaces: bool = False,
    z_clip_value: Optional[float] = None,
    cull_to_frustum: bool = True,
    clipped_faces_neighbor_idx: Optional[torch.Tensor] = None,
):
    """
    Naive PyTorch implementation of mesh rasterization with the same inputs and
    outputs as the rasterize_meshes function.

    This function is not optimized and is implemented as a comparison for the
    C++/CUDA implementations.
    """
    N = len(meshes)
    H, W = image_size if isinstance(image_size, tuple) else (image_size, image_size)

    K = faces_per_pixel
    device = meshes.device

    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    faces_verts = verts_packed[faces_packed]
    mesh_to_face_first_idx = meshes.mesh_to_faces_packed_first_idx()
    num_faces_per_mesh = meshes.num_faces_per_mesh()

    if z_clip_value is not None or cull_to_frustum:
        # Cull faces outside the view frustum, and clip faces that are partially
        # behind the camera into the portion of the triangle in front of the
        # camera.  This may change the number of faces
        frustum = ClipFrustum(
            left=-1,
            right=1,
            top=-1,
            bottom=1,
            perspective_correct=perspective_correct,
            z_clip_value=z_clip_value,
            cull=cull_to_frustum,
        )
        clipped_faces = clip_faces(
            faces_verts, mesh_to_face_first_idx, num_faces_per_mesh, frustum=frustum
        )
        faces_verts = clipped_faces.face_verts
        mesh_to_face_first_idx = clipped_faces.mesh_to_face_first_idx
        num_faces_per_mesh = clipped_faces.num_faces_per_mesh

    # Initialize output tensors.
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
    x_mins = torch.min(faces_verts[:, :, 0], dim=1, keepdim=True).values
    x_maxs = torch.max(faces_verts[:, :, 0], dim=1, keepdim=True).values
    y_mins = torch.min(faces_verts[:, :, 1], dim=1, keepdim=True).values
    y_maxs = torch.max(faces_verts[:, :, 1], dim=1, keepdim=True).values
    z_mins = torch.min(faces_verts[:, :, 2], dim=1, keepdim=True).values

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
            yf = pix_to_non_square_ndc(yfix, H, W)

            # Iterate through pixels on this horizontal line, left to right.
            for xi in range(W):
                # X coordinate of one end of the image. Reverse the ordering
                # of xi so that +X is pointing to the left in the image.
                xfix = W - 1 - xi
                xf = pix_to_non_square_ndc(xfix, W, H)
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

                    # Faces with at least one vertex behind the camera won't
                    # render correctly and should be removed or clipped before
                    # calling the rasterizer
                    if z_mins[f] < kEpsilon:
                        continue

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

                    # Check if inside before clipping
                    inside = all(x > 0.0 for x in bary)

                    # Barycentric clipping
                    if clip_barycentric_coords:
                        bary = barycentric_coordinates_clip(bary)
                    # use clipped barycentric coords to calculate the z value
                    pz = bary[0] * v0[2] + bary[1] * v1[2] + bary[2] * v2[2]

                    # Check if point is behind the image.
                    if pz < 0:
                        continue

                    # Calculate signed 2D distance from point to face.
                    # Points inside the triangle have negative distance.
                    dist = point_triangle_distance(pxy, v0[:2], v1[:2], v2[:2])

                    # Add an epsilon to prevent errors when comparing distance
                    # to blur radius.
                    if not inside and dist >= blur_radius:
                        continue

                    # Handle the case where a face (f) partially behind the image plane is
                    # clipped to a quadrilateral and then split into two faces (t1, t2).
                    top_k_idx = -1
                    if (
                        clipped_faces_neighbor_idx is not None
                        and clipped_faces_neighbor_idx[f] != -1
                    ):
                        neighbor_idx = clipped_faces_neighbor_idx[f]
                        # See if neighbor_idx is in top_k and find index
                        top_k_idx = [
                            i
                            for i, val in enumerate(top_k_points)
                            if val[1] == neighbor_idx
                        ]
                        top_k_idx = top_k_idx[0] if len(top_k_idx) > 0 else -1

                    if top_k_idx != -1 and dist < top_k_points[top_k_idx][3]:
                        # Overwrite the neighbor with current face info
                        top_k_points[top_k_idx] = (pz, f, bary, dist, inside)
                    else:
                        # Handle as a normal face
                        top_k_points.append((pz, f, bary, dist, inside))

                    top_k_points.sort()
                    if len(top_k_points) > K:
                        top_k_points = top_k_points[:K]

                # Save to output tensors.
                for k, (pz, f, bary, dist, inside) in enumerate(top_k_points):
                    zbuf[n, yi, xi, k] = pz
                    face_idxs[n, yi, xi, k] = f
                    bary_coords[n, yi, xi, k, 0] = bary[0]
                    bary_coords[n, yi, xi, k, 1] = bary[1]
                    bary_coords[n, yi, xi, k, 2] = bary[2]
                    # Write the signed distance
                    pix_dists[n, yi, xi, k] = -dist if inside else dist

    if z_clip_value is not None or cull_to_frustum:
        # If faces were clipped, map the rasterization result to be in terms of the
        # original unclipped faces.  This may involve converting barycentric
        # coordinates
        (
            face_idxs,
            bary_coords,
        ) = convert_clipped_rasterization_to_original_faces(
            face_idxs,
            bary_coords,
            # pyre-fixme[61]: `clipped_faces` may not be initialized here.
            clipped_faces,
        )

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


def barycentric_coordinates_clip(bary):
    """
    Clip negative barycentric coordinates to 0.0 and renormalize so
    the barycentric coordinates for a point sum to 1. When the blur_radius
    is greater than 0, a face will still be recorded as overlapping a pixel
    if the pixel is outside the face. In this case at least one of the
    barycentric coordinates for the pixel relative to the face will be negative.
    Clipping will ensure that the texture and z buffer are interpolated correctly.

    Args:
        bary: tuple of barycentric coordinates

    Returns
        bary_clip: (w0, w1, w2) barycentric coordinates with no negative values.
    """
    # Only negative values are clamped to 0.0.
    w0_clip = torch.clamp(bary[0], min=0.0)
    w1_clip = torch.clamp(bary[1], min=0.0)
    w2_clip = torch.clamp(bary[2], min=0.0)
    bary_sum = torch.clamp(w0_clip + w1_clip + w2_clip, min=1e-5)
    w0_clip = w0_clip / bary_sum
    w1_clip = w1_clip / bary_sum
    w2_clip = w2_clip / bary_sum

    return (w0_clip, w1_clip, w2_clip)


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
