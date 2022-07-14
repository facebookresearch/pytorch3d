# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List

import numpy as np
import torch
from pytorch3d.common.datatypes import Device
from pytorch3d.datasets.utils import collate_batched_meshes
from pytorch3d.ops import cubify
from pytorch3d.renderer import (
    HardPhongShader,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    TexturesVertex,
)
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.transforms import Transform3d


# Empirical min and max over the dataset from meshrcnn.
# https://github.com/facebookresearch/meshrcnn/blob/main/shapenet/utils/coords.py#L9
SHAPENET_MIN_ZMIN = 0.67
SHAPENET_MAX_ZMAX = 0.92
# Threshold for cubify from meshrcnn:
# https://github.com/facebookresearch/meshrcnn/blob/main/configs/shapenet/voxmesh_R50.yaml#L11
CUBIFY_THRESH = 0.2

# Default values of rotation, translation and intrinsic matrices for BlenderCamera.
r = np.expand_dims(np.eye(3), axis=0)  # (1, 3, 3)
t = np.expand_dims(np.zeros(3), axis=0)  # (1, 3)
k = np.expand_dims(np.eye(4), axis=0)  # (1, 4, 4)


def collate_batched_R2N2(batch: List[Dict]):  # pragma: no cover
    """
    Take a list of objects in the form of dictionaries and merge them
    into a single dictionary. This function can be used with a Dataset
    object to create a torch.utils.data.Dataloader which directly
    returns Meshes objects.
    TODO: Add support for textures.

    Args:
        batch: List of dictionaries containing information about objects
            in the dataset.

    Returns:
        collated_dict: Dictionary of collated lists. If batch contains both
            verts and faces, a collated mesh batch is also returned.
    """
    collated_dict = collate_batched_meshes(batch)

    # If collate_batched_meshes receives R2N2 items with images and that
    # all models have the same number of views V, stack the batches of
    # views of each model into a new batch of shape (N, V, H, W, 3).
    # Otherwise leave it as a list.
    if "images" in collated_dict:
        try:
            collated_dict["images"] = torch.stack(collated_dict["images"])
        except RuntimeError:
            print(
                "Models don't have the same number of views. Now returning "
                "lists of images instead of batches."
            )

    # If collate_batched_meshes receives R2N2 items with camera calibration
    # matrices and that all models have the same number of views V, stack each
    # type of matrices into a new batch of shape (N, V, ...).
    # Otherwise leave them as lists.
    if all(x in collated_dict for x in ["R", "T", "K"]):
        try:
            collated_dict["R"] = torch.stack(collated_dict["R"])  # (N, V, 3, 3)
            collated_dict["T"] = torch.stack(collated_dict["T"])  # (N, V, 3)
            collated_dict["K"] = torch.stack(collated_dict["K"])  # (N, V, 4, 4)
        except RuntimeError:
            print(
                "Models don't have the same number of views. Now returning "
                "lists of calibration matrices instead of a batched tensor."
            )

    # If collate_batched_meshes receives voxels and all models have the same
    # number of views V, stack the batches of voxels into a new batch of shape
    # (N, V, S, S, S), where S is the voxel size.
    if "voxels" in collated_dict:
        try:
            collated_dict["voxels"] = torch.stack(collated_dict["voxels"])
        except RuntimeError:
            print(
                "Models don't have the same number of views. Now returning "
                "lists of voxels instead of a batched tensor."
            )
    return collated_dict


def compute_extrinsic_matrix(
    azimuth: float, elevation: float, distance: float
):  # pragma: no cover
    """
    Copied from meshrcnn codebase:
    https://github.com/facebookresearch/meshrcnn/blob/main/shapenet/utils/coords.py#L96

    Compute 4x4 extrinsic matrix that converts from homogeneous world coordinates
    to homogeneous camera coordinates. We assume that the camera is looking at the
    origin.
    Used in R2N2 Dataset when computing calibration matrices.

    Args:
        azimuth: Rotation about the z-axis, in degrees.
        elevation: Rotation above the xy-plane, in degrees.
        distance: Distance from the origin.

    Returns:
        FloatTensor of shape (4, 4).
    """
    azimuth, elevation, distance = float(azimuth), float(elevation), float(distance)

    az_rad = -math.pi * azimuth / 180.0
    el_rad = -math.pi * elevation / 180.0
    sa = math.sin(az_rad)
    ca = math.cos(az_rad)
    se = math.sin(el_rad)
    ce = math.cos(el_rad)
    R_world2obj = torch.tensor(
        [[ca * ce, sa * ce, -se], [-sa, ca, 0], [ca * se, sa * se, ce]]
    )
    R_obj2cam = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    R_world2cam = R_obj2cam.mm(R_world2obj)
    cam_location = torch.tensor([[distance, 0, 0]]).t()
    T_world2cam = -(R_obj2cam.mm(cam_location))
    RT = torch.cat([R_world2cam, T_world2cam], dim=1)
    RT = torch.cat([RT, torch.tensor([[0.0, 0, 0, 1]])])

    # Georgia: For some reason I cannot fathom, when Blender loads a .obj file it
    # rotates the model 90 degrees about the x axis. To compensate for this quirk we
    # roll that rotation into the extrinsic matrix here
    rot = torch.tensor([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    RT = RT.mm(rot.to(RT))

    return RT


def read_binvox_coords(
    f,
    integer_division: bool = True,
    dtype: torch.dtype = torch.float32,
):  # pragma: no cover
    """
    Copied from meshrcnn codebase:
    https://github.com/facebookresearch/meshrcnn/blob/main/shapenet/utils/binvox_torch.py#L5

    Read a binvox file and return the indices of all nonzero voxels.

    This matches the behavior of binvox_rw.read_as_coord_array
    (https://github.com/dimatura/binvox-rw-py/blob/public/binvox_rw.py#L153)
    but this implementation uses torch rather than numpy, and is more efficient
    due to improved vectorization.

    Georgia: I think that binvox_rw.read_as_coord_array actually has a bug; when converting
    linear indices into three-dimensional indices, they use floating-point
    division instead of integer division. We can reproduce their incorrect
    implementation by passing integer_division=False.

    Args:
      f (str): A file pointer to the binvox file to read
      integer_division (bool): If False, then match the buggy implementation from binvox_rw
      dtype: Datatype of the output tensor. Use float64 to match binvox_rw

    Returns:
      coords (tensor): A tensor of shape (N, 3) where N is the number of nonzero voxels,
           and coords[i] = (x, y, z) gives the index of the ith nonzero voxel. If the
           voxel grid has shape (V, V, V) then we have 0 <= x, y, z < V.
    """
    size, translation, scale = _read_binvox_header(f)
    storage = torch.ByteStorage.from_buffer(f.read())
    data = torch.tensor([], dtype=torch.uint8)
    # pyre-fixme[28]: Unexpected keyword argument `source`.
    data.set_(source=storage)
    vals, counts = data[::2], data[1::2]
    idxs = _compute_idxs(vals, counts)
    if not integer_division:
        idxs = idxs.to(dtype)
    x_idxs = idxs // (size * size)
    zy_idxs = idxs % (size * size)
    z_idxs = zy_idxs // size
    y_idxs = zy_idxs % size
    coords = torch.stack([x_idxs, y_idxs, z_idxs], dim=1)
    return coords.to(dtype)


def _compute_idxs(vals, counts):  # pragma: no cover
    """
    Copied from meshrcnn codebase:
    https://github.com/facebookresearch/meshrcnn/blob/main/shapenet/utils/binvox_torch.py#L58

    Fast vectorized version of index computation.

    Args:
        vals: tensor of binary values indicating voxel presence in a dense format.
        counts: tensor of number of occurrence of each value in vals.

    Returns:
        idxs: A tensor of shape (N), where N is the number of nonzero voxels.
    """
    # Consider an example where:
    # vals   = [0, 1, 0, 1, 1]
    # counts = [2, 3, 3, 2, 1]
    #
    # These values of counts and vals mean that the dense binary grid is:
    # [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
    #
    # So the nonzero indices we want to return are:
    # [2, 3, 4, 8, 9, 10]

    # After the cumsum we will have:
    # end_idxs = [2, 5, 8, 10, 11]
    end_idxs = counts.cumsum(dim=0)

    # After masking and computing start_idx we have:
    # end_idxs   = [5, 10, 11]
    # counts     = [3,  2,  1]
    # start_idxs = [2,  8, 10]
    mask = vals == 1
    end_idxs = end_idxs[mask]
    counts = counts[mask].to(end_idxs)
    start_idxs = end_idxs - counts

    # We initialize delta as:
    # [2, 1, 1, 1, 1, 1]
    delta = torch.ones(counts.sum().item(), dtype=torch.int64)
    delta[0] = start_idxs[0]

    # We compute pos = [3, 5], val = [3, 0]; then delta is
    # [2, 1, 1, 4, 1, 1]
    pos = counts.cumsum(dim=0)[:-1]
    val = start_idxs[1:] - end_idxs[:-1]
    delta[pos] += val

    # A final cumsum gives the idx we want: [2, 3, 4, 8, 9, 10]
    idxs = delta.cumsum(dim=0)
    return idxs


def _read_binvox_header(f):  # pragma: no cover
    """
    Copied from meshrcnn codebase:
    https://github.com/facebookresearch/meshrcnn/blob/main/shapenet/utils/binvox_torch.py#L99

    Read binvox header and extract information regarding voxel sizes and translations
    to original voxel coordinates.

    Args:
        f (str): A file pointer to the binvox file to read.

    Returns:
        size (int): size of voxel.
        translation (tuple(float)): translation to original voxel coordinates.
        scale (float): scale to original voxel coordinates.
    """
    # First line of the header should be "#binvox 1"
    line = f.readline().strip()
    if line != b"#binvox 1":
        raise ValueError("Invalid header (line 1)")

    # Second line of the header should be "dim [int] [int] [int]"
    # and all three int should be the same
    line = f.readline().strip()
    if not line.startswith(b"dim "):
        raise ValueError("Invalid header (line 2)")
    dims = line.split(b" ")
    try:
        dims = [int(d) for d in dims[1:]]
    except ValueError:
        raise ValueError("Invalid header (line 2)") from None
    if len(dims) != 3 or dims[0] != dims[1] or dims[0] != dims[2]:
        raise ValueError("Invalid header (line 2)")
    size = dims[0]

    # Third line of the header should be "translate [float] [float] [float]"
    line = f.readline().strip()
    if not line.startswith(b"translate "):
        raise ValueError("Invalid header (line 3)")
    translation = line.split(b" ")
    if len(translation) != 4:
        raise ValueError("Invalid header (line 3)")
    try:
        translation = tuple(float(t) for t in translation[1:])
    except ValueError:
        raise ValueError("Invalid header (line 3)") from None

    # Fourth line of the header should be "scale [float]"
    line = f.readline().strip()
    if not line.startswith(b"scale "):
        raise ValueError("Invalid header (line 4)")
    line = line.split(b" ")
    if not len(line) == 2:
        raise ValueError("Invalid header (line 4)")
    scale = float(line[1])

    # Fifth line of the header should be "data"
    line = f.readline().strip()
    if not line == b"data":
        raise ValueError("Invalid header (line 5)")

    return size, translation, scale


def align_bbox(src, tgt):  # pragma: no cover
    """
    Copied from meshrcnn codebase:
    https://github.com/facebookresearch/meshrcnn/blob/main/tools/preprocess_shapenet.py#L263

    Return a copy of src points in the coordinate system of tgt by applying a
    scale and shift along each coordinate axis to make the min / max values align.

    Args:
        src, tgt: Torch Tensor of shape (N, 3)

    Returns:
        out: Torch Tensor of shape (N, 3)
    """
    if src.ndim != 2 or tgt.ndim != 2:
        raise ValueError("Both src and tgt need to have dimensions of 2.")
    if src.shape[-1] != 3 or tgt.shape[-1] != 3:
        raise ValueError(
            "Both src and tgt need to have sizes of 3 along the second dimension."
        )
    src_min = src.min(dim=0)[0]
    src_max = src.max(dim=0)[0]
    tgt_min = tgt.min(dim=0)[0]
    tgt_max = tgt.max(dim=0)[0]
    scale = (tgt_max - tgt_min) / (src_max - src_min)
    shift = tgt_min - scale * src_min
    out = scale * src + shift
    return out


def voxelize(voxel_coords, P, V):  # pragma: no cover
    """
    Copied from meshrcnn codebase:
    https://github.com/facebookresearch/meshrcnn/blob/main/tools/preprocess_shapenet.py#L284
    but changing flip y to flip x.

    Creating voxels of shape (D, D, D) from voxel_coords and projection matrix.

    Args:
        voxel_coords: FloatTensor of shape (V, 3) giving voxel's coordinates aligned to
            the vertices.
        P: FloatTensor of shape (4, 4) giving the projection matrix.
        V: Voxel size of the output.

    Returns:
        voxels: Tensor of shape (D, D, D) giving the voxelized result.
    """
    device = voxel_coords.device
    voxel_coords = project_verts(voxel_coords, P)

    # Using the actual zmin and zmax of the model is bad because we need them
    # to perform the inverse transform, which transform voxels back into world
    # space for refinement or evaluation. Instead we use an empirical min and
    # max over the dataset; that way it is consistent for all images.
    zmin = SHAPENET_MIN_ZMIN
    zmax = SHAPENET_MAX_ZMAX

    # Once we know zmin and zmax, we need to adjust the z coordinates so the
    # range [zmin, zmax] instead runs from [-1, 1]
    m = 2.0 / (zmax - zmin)
    b = -2.0 * zmin / (zmax - zmin) - 1
    voxel_coords[:, 2].mul_(m).add_(b)
    voxel_coords[:, 0].mul_(-1)  # Flip x

    # Now voxels are in [-1, 1]^3; map to [0, V-1)^3
    voxel_coords = 0.5 * (V - 1) * (voxel_coords + 1.0)
    voxel_coords = voxel_coords.round().to(torch.int64)
    valid = (0 <= voxel_coords) * (voxel_coords < V)
    valid = valid[:, 0] * valid[:, 1] * valid[:, 2]
    x, y, z = voxel_coords.unbind(dim=1)
    x, y, z = x[valid], y[valid], z[valid]
    voxels = torch.zeros(V, V, V, dtype=torch.uint8, device=device)
    voxels[z, y, x] = 1

    return voxels


def project_verts(verts, P, eps: float = 1e-1):  # pragma: no cover
    """
    Copied from meshrcnn codebase:
    https://github.com/facebookresearch/meshrcnn/blob/main/shapenet/utils/coords.py#L159

    Project vertices using a 4x4 transformation matrix.

    Args:
        verts: FloatTensor of shape (N, V, 3) giving a batch of vertex positions or of
            shape (V, 3) giving a single set of vertex positions.
        P: FloatTensor of shape (N, 4, 4) giving projection matrices or of shape (4, 4)
            giving a single projection matrix.

    Returns:
        verts_out: FloatTensor of shape (N, V, 3) giving vertex positions (x, y, z)
            where verts_out[i] is the result of transforming verts[i] by P[i].
    """
    # Handle unbatched inputs
    singleton = False
    if verts.dim() == 2:
        assert P.dim() == 2
        singleton = True
        verts, P = verts[None], P[None]

    N, V = verts.shape[0], verts.shape[1]
    dtype, device = verts.dtype, verts.device

    # Add an extra row of ones to the world-space coordinates of verts before
    # multiplying by the projection matrix. We could avoid this allocation by
    # instead multiplying by a 4x3 submatrix of the projection matrix, then
    # adding the remaining 4x1 vector. Not sure whether there will be much
    # performance difference between the two.
    ones = torch.ones(N, V, 1, dtype=dtype, device=device)
    verts_hom = torch.cat([verts, ones], dim=2)
    verts_cam_hom = torch.bmm(verts_hom, P.transpose(1, 2))

    # Avoid division by zero by clamping the absolute value
    w = verts_cam_hom[:, :, 3:]
    w_sign = w.sign()
    w_sign[w == 0] = 1
    w = w_sign * w.abs().clamp(min=eps)

    verts_proj = verts_cam_hom[:, :, :3] / w

    if singleton:
        return verts_proj[0]
    return verts_proj


class BlenderCamera(CamerasBase):  # pragma: no cover
    """
    Camera for rendering objects with calibration matrices from the R2N2 dataset
    (which uses Blender for rendering the views for each model).
    """

    def __init__(self, R=r, T=t, K=k, device: Device = "cpu") -> None:
        """
        Args:
            R: Rotation matrix of shape (N, 3, 3).
            T: Translation matrix of shape (N, 3).
            K: Intrinsic matrix of shape (N, 4, 4).
            device: Device (as str or torch.device).
        """
        # The initializer formats all inputs to torch tensors and broadcasts
        # all the inputs to have the same batch dimension where necessary.
        super().__init__(device=device, R=R, T=T, K=K)

    def get_projection_transform(self, **kwargs) -> Transform3d:
        transform = Transform3d(device=self.device)
        transform._matrix = self.K.transpose(1, 2).contiguous()
        return transform

    def is_perspective(self):
        return False

    def in_ndc(self):
        return True


def render_cubified_voxels(
    voxels: torch.Tensor, shader_type=HardPhongShader, device: Device = "cpu", **kwargs
):  # pragma: no cover
    """
    Use the Cubify operator to convert inputs voxels to a mesh and then render that mesh.

    Args:
        voxels: FloatTensor of shape (N, D, D, D) where N is the batch size and
            D is the number of voxels along each dimension.
        shader_type: shader_type: shader_type: Shader to use for rendering. Examples
            include HardPhongShader (default), SoftPhongShader etc or any other type
            of valid Shader class.
        device: Device (as str or torch.device) on which the tensors should be located.
        **kwargs: Accepts any of the kwargs that the renderer supports.
    Returns:
        Batch of rendered images of shape (N, H, W, 3).
    """
    cubified_voxels = cubify(voxels, CUBIFY_THRESH).to(device)
    cubified_voxels.textures = TexturesVertex(
        verts_features=torch.ones_like(cubified_voxels.verts_padded(), device=device)
    )
    cameras = BlenderCamera(device=device)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=kwargs.get("raster_settings", RasterizationSettings()),
        ),
        shader=shader_type(
            device=device,
            cameras=cameras,
            lights=kwargs.get("lights", PointLights()).to(device),
        ),
    )
    return renderer(cubified_voxels)
