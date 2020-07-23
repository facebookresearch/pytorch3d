# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

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

    # If collate_batched_meshes receives R2N2 items, stack the batches of
    # views of each model into a new batch of shape (N, V, H, W, 3) where
    # V is the number of views.
    if "images" in collated_dict:
        collated_dict["images"] = torch.stack(collated_dict["images"])

    return collated_dict
