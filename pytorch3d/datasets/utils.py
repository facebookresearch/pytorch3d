# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import math
from typing import Dict, List

import torch
from pytorch3d.structures import Meshes


def collate_batched_meshes(batch: List[Dict]):
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
    if batch is None or len(batch) == 0:
        return None
    collated_dict = {}
    for k in batch[0].keys():
        collated_dict[k] = [d[k] for d in batch]

    collated_dict["mesh"] = None
    if {"verts", "faces"}.issubset(collated_dict.keys()):
        collated_dict["mesh"] = Meshes(
            verts=collated_dict["verts"], faces=collated_dict["faces"]
        )

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
                "lists of calibration matrices instead of batches."
            )

    return collated_dict


def compute_extrinsic_matrix(azimuth, elevation, distance):
    """
    Copied from meshrcnn codebase:
    https://github.com/facebookresearch/meshrcnn/blob/master/shapenet/utils/coords.py#L96

    Compute 4x4 extrinsic matrix that converts from homogenous world coordinates
    to homogenous camera coordinates. We assume that the camera is looking at the
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
