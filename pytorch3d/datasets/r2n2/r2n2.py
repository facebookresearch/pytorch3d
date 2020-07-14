# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import json
import warnings
from os import path
from pathlib import Path
from typing import Dict

from pytorch3d.datasets.shapenet_base import ShapeNetBase
from pytorch3d.io import load_obj


SYNSET_DICT_DIR = Path(__file__).resolve().parent


class R2N2(ShapeNetBase):
    """
    This class loads the R2N2 dataset from a given directory into a Dataset object.
    The R2N2 dataset contains 13 categories that are a subset of the ShapeNetCore v.1
    dataset. The R2N2 dataset also contains its own 24 renderings of each object and
    voxelized models.
    """

    def __init__(self, split, shapenet_dir, r2n2_dir, splits_file):
        """
        Store each object's synset id and models id the given directories.
        Args:
            split (str): One of (train, val, test).
            shapenet_dir (path): Path to ShapeNet core v1.
            r2n2_dir (path): Path to the R2N2 dataset.
            splits_file (path): File containing the train/val/test splits.
        """
        super().__init__()
        self.shapenet_dir = shapenet_dir
        self.r2n2_dir = r2n2_dir
        # Examine if split is valid.
        if split not in ["train", "val", "test"]:
            raise ValueError("split has to be one of (train, val, test).")
        # Synset dictionary mapping synset offsets in R2N2 to corresponding labels.
        with open(
            path.join(SYNSET_DICT_DIR, "r2n2_synset_dict.json"), "r"
        ) as read_dict:
            self.synset_dict = json.load(read_dict)
        # Inverse dicitonary mapping synset labels to corresponding offsets.
        self.synset_inv = {label: offset for offset, label in self.synset_dict.items()}

        # Store synset and model ids of objects mentioned in the splits_file.
        with open(splits_file) as splits:
            split_dict = json.load(splits)[split]

        synset_set = set()
        for synset in split_dict.keys():
            # Examine if the given synset is present in the ShapeNetCore dataset
            # and is also part of the standard R2N2 dataset.
            if not (
                path.isdir(path.join(shapenet_dir, synset))
                and synset in self.synset_dict
            ):
                msg = (
                    "Synset category %s from the splits file is either not "
                    "present in %s or not part of the standard R2N2 dataset."
                ) % (synset, shapenet_dir)
                warnings.warn(msg)
                continue

            synset_set.add(synset)
            models = split_dict[synset].keys()
            for model in models:
                # Examine if the given model is present in the ShapeNetCore path.
                shapenet_path = path.join(shapenet_dir, synset, model)
                if not path.isdir(shapenet_path):
                    msg = "Model %s from category %s is not present in %s." % (
                        model,
                        synset,
                        shapenet_dir,
                    )
                    warnings.warn(msg)
                    continue
                self.synset_ids.append(synset)
                self.model_ids.append(model)

        # Examine if all the synsets in the standard R2N2 mapping are present.
        # Update self.synset_inv so that it only includes the loaded categories.
        synset_not_present = [
            self.synset_inv.pop(self.synset_dict[synset])
            for synset in self.synset_dict.keys()
            if synset not in synset_set
        ]
        if len(synset_not_present) > 0:
            msg = (
                "The following categories are included in R2N2's"
                "official mapping but not found in the dataset location %s: %s"
            ) % (shapenet_dir, ", ".join(synset_not_present))
            warnings.warn(msg)

    def __getitem__(self, idx: int) -> Dict:
        """
        Read a model by the given index.

        Args:
            idx: The idx of the model to be retrieved in the dataset.

        Returns:
            dictionary with following keys:
            - verts: FloatTensor of shape (V, 3).
            - faces: faces.verts_idx, LongTensor of shape (F, 3).
            - synset_id (str): synset id.
            - model_id (str): model id.
            - label (str): synset label.
        """
        model = self._get_item_ids(idx)
        model_path = path.join(
            self.shapenet_dir, model["synset_id"], model["model_id"], "model.obj"
        )
        model["verts"], faces, _ = load_obj(model_path)
        model["faces"] = faces.verts_idx
        model["label"] = self.synset_dict[model["synset_id"]]
        return model
