# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""
This module implements utility functions for sampling points from
an obj having multiple textures.
"""
from typing import Tuple, Optional, Dict

import numpy as np
import torch

from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
from pytorch3d.ops.packed_to_padded import packed_to_padded
from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.ops.sample_points_from_meshes import (
    _sample_points,
    _sample_normals,
    _sample_textures,
)
from pytorch3d.utils.obj_utils import parse_obj_to_mesh_by_texture, _validate_obj


def sample_points_from_obj(
    verts: torch.Tensor,
    faces: torch.Tensor,
    verts_uvs: torch.Tensor = None,
    faces_uvs: torch.Tensor = None,
    texture_images: Optional[Dict[str, torch.tensor]] = None,
    materials_idx: Optional[torch.Tensor] = None,
    texture_atlas: Optional[torch.Tensor] = None,
    use_texture_atlas: Optional[bool] = False,
    num_samples: Optional[int] = None,
    sample_all_faces: Optional[bool] = False,
    sampling_factors: Optional[torch.Tensor] = None,
    min_sampling_factor: Optional[int] = 1,
    return_normals: Optional[bool] = False,
    return_textures: Optional[bool] = False,
    return_mappers: Optional[bool] = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert mesh faces to a pointcloud by uniformly sampling points
    from surfaces with probability proportional to the face area.
    Provides support for faces with multiple textures and materials
    by sampling the input as submeshes by texture. Allows forcing at
    least one sample per face, regardless of surface area. The expected
    input data structure is a pytorch3d obj that includes verts, faces,
    and aux data; however, this function will generically support any
    mesh defined by these data structures as the underlying data
    are converted to a pytorch3d meshes object

    Args:
        verts: A tensor of verts. For objs, typically the tensor associated with obj[0].
        faces: A tensor of faces cooresponding to verts. For objs, typically the tensor associated with obj[1].verts_idx.
        verts_uvs: A tensor of uv coords per vertex. For objs, typically the tensor associated with obj[2].verts_uvs.
        faces_uvs: A tensor giving the index into verts_uvs for each vertex. For objs, typically the tensor associated with obj[1].textures_idx.
        texture_images: A dictionary keyed by material name with the texture image tensor. For objs, typically the tensor associated with obj[2].texture_images.
        materials_idx: A tensor giving the material index to each face to texture in texture_images. For objs, typically the tensor associated with obj[1].materials_idx.
        texture_atlas: A tensor representing the RxR texture map for each face. For objs, typically the tensor associated with obj[2].texture_atlas.
        use_texture_atlas: If true, sample from texture atlas instead of texture images.
        num_samples: If None, the sample size per mesh in the obj defaults to a factor proportial to face area.
            If num_samples is provided, a fixed number of samples is picked for all obj submeshes.
        sample_all_faces: If True, at least one point is sampled per face, regardless of face area.
        sampling_factors: Default to None. Optionally, specify the number of samples to produce per obj texture as a LongTensor.
        min_sampling_factor: When auto sampling (num_samples == None), sample sizes are picked proportional to face area by multiplying
            min_sampling_factor by surface area. This value can be used as a floor value to sample from each face which can arbitrarily increase the number of points sampled.
            A range of 1 to 10000 is recommended where 1 is least dense and 10000 is very dense.
        return_normals: If True, return normals for the sampled points.
        return_textures: If True, return textures for the sampled points.
        return_mappers: If True, return mappers for each point to its origin face in the input obj.
            mappers is tensor where the tensor index references the point index in the pointcloud and the
            tensor value references the face index in the input mesh; mappers.shape[0] == num_samples.
    Returns:
        A 4-element tuple, where batch size B is always 1 for 1 OBj input, and total_samples is the
        combined size of all concatenated samples taken. The total samples returned may increase if
        any are True: num_samples is None (auto sampling), sample_all_faces is True.

        - **samples**: FloatTensor of shape (B, total_samples, 3) giving the
          coordinates of sampled points for each mesh in the batch.
        - **normals**: None or FloatTensor of shape (B, total_samples, 3) giving a normal vector
          to each sampled point. Only returned if return_normals is True.
        - **textures**: None or FloatTensor of shape (B, total_samples, C) giving a C-dimensional
          texture vector to each sampled point. Only returned if return_textures is True.
          - **mappers**: None or IntTensor of shape (B, total_samples) providing a point to face mapping
          for each point's origin face in the sample.

    """
    device = verts.device
    auto_samples = False

    _validate_obj(
        verts=verts,
        faces=faces,
        faces_uvs=faces_uvs,
        verts_uvs=verts_uvs,
        texture_images=texture_images,
        materials_idx=materials_idx,
    )

    if verts.shape[0] == 0 or faces.shape[0] == 0:
        raise ValueError("OBJ is empty.")

    if not torch.isfinite(verts).all():
        raise ValueError("Verts contain nan or inf.")

    if num_samples is None:
        auto_samples = True

    if return_textures and None in [
        verts_uvs,
        faces_uvs,
        materials_idx,
        texture_images,
    ]:
        return_textures = False

    if use_texture_atlas and texture_atlas is None:
        use_texture_atlas = False

    if min_sampling_factor is None or min_sampling_factor < 0:
        min_sampling_factor = 1

    if not return_textures:
        meshes = join_meshes_as_batch([Meshes(verts=[verts], faces=[faces])])
    else:
        meshes = parse_obj_to_mesh_by_texture(
            verts=verts,
            faces=faces,
            verts_uvs=verts_uvs,
            faces_uvs=faces_uvs,
            texture_images=texture_images,
            device=device,
            materials_idx=materials_idx,
            texture_atlas=texture_atlas,
            use_texture_atlas=use_texture_atlas,
        )

        meshes = join_meshes_as_batch(meshes)

    sampling_sizes, areas_padded = _pick_sampling_sizes(
        meshes=meshes,
        min_sampling_factor=min_sampling_factor,
        sampling_factors=sampling_factors,
        return_areas=True,
    )

    if not auto_samples:
        # force _sample_meshes helper to sample at least num_samples from each mesh
        sampling_sizes = torch.full(sampling_sizes.shape, num_samples).type(
            torch.LongTensor
        )

    if sampling_sizes.shape[0] != len(meshes):
        message = "sampling_sizes.shape[0] != len(meshes); check sampling_factors"
        raise ValueError(message)

    (samples, normals, textures, mappers) = _sample_meshes(
        meshes=meshes,
        sampling_sizes=sampling_sizes,
        areas_padded=areas_padded,
        sample_all_faces=sample_all_faces,
        return_normals=return_normals,
        return_textures=return_textures,
        return_mappers=return_mappers,
    )

    return samples, normals, textures, mappers


@torch.no_grad()
def _pick_sampling_sizes(
    meshes: Meshes,
    min_sampling_factor: Optional[int] = 1,
    sampling_factors: Optional[torch.Tensor] = None,
    return_areas: Optional[bool] = False,
) -> torch.Tensor:
    """This is a helper function that picks varying num_samples proportional
    to the area of each mesh in a batch of meshes.

    num_samples is equal to sampling_factor * the sum of the mesh area.

    sampling_factors for each mesh in the input batch can be set at absolute values
    or by picking a small number, relative to the number of faces in a mesh. In
    this implementation, sampling factors is given by the number of faces
    in each mesh to the 1/4th power, divided by three, with a floor value equal to
    min_sampling_factor.

    For example, in practice, a mesh having 159220 faces and a total sum area of 19050 will
    have sample factor of 6 and a resulting sample size of 114300. In this case, most faces
    with non-zero areas may be represented in the point cloud with one or more points based
    on the face area. If there are fewer faces or smaller areas, the sampling factor will be
    at least min_sampling_factor times the surface area.

    Args:
        meshes: A batch of N meshes to sample.
        min_sampling_factor: A minimum value, default to 1, to multiply against the area of each face
            and produce number of points to sample from each face.
        sampling_factors: Optionally, pick an aribitrary set of sampling factors for each mesh.
        return_areas: Whether to return the areas_padded tensor.
    Returns:
        a 2-Element Tuple of:

        - **sampling_sizes** : An Nx1 tensor specifying the sampling size for each mesh.
        - **areas_padded**: An NxMaxFaces tensor that provides the areas for each face, padded to the longest tensor in the batch.
    """

    areas, _ = mesh_face_areas_normals(meshes.verts_packed(), meshes.faces_packed())
    num_faces_per_mesh = meshes.num_faces_per_mesh()
    max_faces = num_faces_per_mesh.max().item()
    mesh_to_face = meshes.mesh_to_faces_packed_first_idx()
    areas_padded = packed_to_padded(
        areas, mesh_to_face[meshes.valid], max_faces
    )  # (N, F)

    if sampling_factors is None:
        sampling_factors = (
            torch.pow(num_faces_per_mesh, 1 / 4)
            .div(3)
            .clamp(min=min_sampling_factor)
            .type(dtype=torch.LongTensor)
            .to(areas_padded.device)
        )
    # pad sizes for estimation by one unit
    areas_sum = torch.sum(areas_padded, dim=1)
    sampling_sizes = (
        torch.mul(areas_sum + 1, sampling_factors).ceil().type(torch.LongTensor)
    )

    if return_areas:
        return sampling_sizes, areas_padded
    else:
        return sampling_sizes


@torch.no_grad()
def _sample_meshes(
    meshes: torch.Tensor,
    sampling_sizes: torch.Tensor,
    areas_padded: torch.Tensor,
    sample_all_faces: bool,
    return_normals: bool,
    return_textures: bool,
    return_mappers: bool,
) -> Tuple[Tuple[torch.Tensor, torch.tensor, torch.tensor, torch.tensor]]:
    """This is a helper function that implements the original functionality in
    sample_points_from_meshes with additional features to sample from obj inputs.

    Given a batch of meshes, point clouds of aribitrary sizes
    are sampled from each mesh and combined into a single point cloud sample.
    This function is appropriate when the batch of meshes represent a
    single scene or object and it is desirable to sample a single
    point cloud that represents a combination of the input.

    For example, this function is designed to support sampling from an input obj
    having multiple textures and selecting num_samples proportional to the face
    area in each mesh. This feature is convenient if the meshes vary greatly in
    size, shape, or area and a padded sample of fixed size is not desired.
    In addition, this function provides a feature to sample at least one point per
    face where face areas are zero or close enough to zero that standard sampling
    will skip. This feature can be desireable if a point cloud representation is
    used to learn a model of meshes and at least one point per mesh is necessary.

    Since the samples are not done in batches, this function is designed to accumulate
    tensors and face index offsets by iterating over the input batch.

    Args:
        meshes: A batch of N meshes to sample.
        sampling_sizes: An Nx1 tensor specifying the sampling size for each mesh.
        areas_padded: An NxMaxFaces tensor that provides the areas for each face, padded to the longest tensor in the batch.
        sample_all_faces: If True, at least one point per face is sampled, regardless of area.
        return_normals: If True, samples normals.
        return_textures: If True, samples textures.
        return_mappers: If True, samples mappers.
    Returns:
        A 4-element tuple, where batch size B is always 1 for 1 OBj input, and total_samples is the
        combined size of all concatenated samples taken.

        - **samples**: FloatTensor of shape (B, total_samples, 3) giving the
          coordinates of sampled points for each mesh in the batch.
        - **normals**: None or FloatTensor of shape (B, total_samples, 3) giving a normal vector
          to each sampled point. Only returned if return_normals is True.
        - **textures**: None or FloatTensor of shape (B, total_samples, C) giving a C-dimensional
          texture vector to each sampled point. Only returned if return_textures is True.
          - **mappers**: None or IntTensor of shape (B, total_samples) providing a point to face mapping
          for each point's origin face in the sample.

    """
    # initialize default values for returned data
    samples, normals, textures, mappers = None, None, None, None
    # initialize an offset for each mesh by the ith num_faces_per_mesh
    mappers_offset = 0
    num_faces_per_mesh = meshes.num_faces_per_mesh()

    # iterate through each mesh in the mesh batch, sampling a varying number of points
    for i, j in enumerate(sampling_sizes):
        # i index and j sample size
        n_faces = num_faces_per_mesh[i].item()
        # initiate sample_face_idxs for only non-padded face areas
        sample_face_idxs = areas_padded[i][..., :n_faces].multinomial(
            j, replacement=True
        )

        if sample_all_faces:
            # if at least one point per face is desired, check to add all face indices
            represented_faces = torch.unique(sample_face_idxs)
            # check difference of sets and add concat any values not present
            if represented_faces.shape[0] < n_faces:
                unpresented_faces = np.setdiff1d(
                    np.arange(start=0, stop=n_faces),
                    list(set(represented_faces.tolist())),
                )
                sample_face_idxs, _ = torch.cat(
                    (
                        sample_face_idxs,
                        torch.LongTensor(unpresented_faces).to(sample_face_idxs.device),
                    )
                ).sort()
        # curr_samples as the number of points to sample from the ith mesh
        curr_samples = sample_face_idxs.shape[0]
        verts = meshes[i].verts_packed()
        # fill the empty sample tensor and face verts/bary coords
        (_samples, (v0, v1, v2), (w0, w1, w2)) = _sample_points(
            meshes[i],
            curr_samples,
            sample_face_idxs,
            verts,
            meshes[i].faces_packed(),
        )
        # accumulate samples in concat tensor
        samples = (
            torch.cat([samples, _samples], dim=1) if samples is not None else _samples
        )
        # sample and concat normals if requested
        if return_normals:
            _normals = _sample_normals(
                meshes[i],
                curr_samples,
                sample_face_idxs,
                v0,
                v1,
                v2,
            )
            normals = (
                torch.cat([normals, _normals], dim=1)
                if normals is not None
                else _normals
            )
        # sample and concat textures if requested
        if return_textures:
            _textures = _sample_textures(
                meshes[i],
                curr_samples,
                sample_face_idxs,
                w0,
                w1,
                w2,
            )
            textures = (
                torch.cat([textures, _textures], dim=1)
                if textures is not None
                else _textures
            )
        # sample and concat mappers if requested
        if return_mappers:
            _mappers = sample_face_idxs.unsqueeze(0) + mappers_offset
            mappers = (
                torch.cat([mappers, _mappers], dim=1)
                if mappers is not None
                else _mappers
            )

            mappers_offset += n_faces

    return samples, normals, textures, mappers
