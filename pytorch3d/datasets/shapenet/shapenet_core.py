# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import json
import os
import warnings
from os import path
from pathlib import Path
from typing import Dict

from pytorch3d.datasets.shapenet_base import ShapeNetBase


SYNSET_DICT_DIR = Path(__file__).resolve().parent


class ShapeNetCore(ShapeNetBase):  # pragma: no cover
    """
    This class loads ShapeNetCore from a given directory into a Dataset object.
    ShapeNetCore is a subset of the ShapeNet dataset and can be downloaded from
    https://www.shapenet.org/.
    """

    def __init__(
        self,
        data_dir,
        synsets=None,
        version: int = 1,
        load_textures: bool = True,
        texture_resolution: int = 4,
    ) -> None:
        """
        Store each object's synset id and models id from data_dir.

        Args:
            data_dir: Path to ShapeNetCore data.
            synsets: List of synset categories to load from ShapeNetCore in the form of
                synset offsets or labels. A combination of both is also accepted.
                When no category is specified, all categories in data_dir are loaded.
            version: (int) version of ShapeNetCore data in data_dir, 1 or 2.
                Default is set to be 1. Version 1 has 57 categories and version 2 has 55
                categories.
                Note: version 1 has two categories 02858304(boat) and 02992529(cellphone)
                that are hyponyms of categories 04530566(watercraft) and 04401088(telephone)
                respectively. You can combine the categories manually if needed.
                Version 2 doesn't have 02858304(boat) or 02834778(bicycle) compared to
                version 1.
            load_textures: Boolean indicating whether textures should loaded for the model.
                Textures will be of type TexturesAtlas i.e. a texture map per face.
            texture_resolution: Int specifying the resolution of the texture map per face
                created using the textures in the obj file. A
                (texture_resolution, texture_resolution, 3) map is created per face.
        """
        super().__init__()
        self.shapenet_dir = data_dir
        self.load_textures = load_textures
        self.texture_resolution = texture_resolution

        if version not in [1, 2]:
            raise ValueError("Version number must be either 1 or 2.")
        self.model_dir = "model.obj" if version == 1 else "models/model_normalized.obj"

        # Synset dictionary mapping synset offsets to corresponding labels.
        dict_file = "shapenet_synset_dict_v%d.json" % version
        with open(path.join(SYNSET_DICT_DIR, dict_file), "r") as read_dict:
            self.synset_dict = json.load(read_dict)
        # Inverse dictionary mapping synset labels to corresponding offsets.
        self.synset_inv = {label: offset for offset, label in self.synset_dict.items()}

        # If categories are specified, check if each category is in the form of either
        # synset offset or synset label, and if the category exists in the given directory.
        if synsets is not None:
            # Set of categories to load in the form of synset offsets.
            synset_set = set()
            for synset in synsets:
                if (synset in self.synset_dict.keys()) and (
                    path.isdir(path.join(data_dir, synset))
                ):
                    synset_set.add(synset)
                elif (synset in self.synset_inv.keys()) and (
                    path.isdir(path.join(data_dir, self.synset_inv[synset]))
                ):
                    synset_set.add(self.synset_inv[synset])
                else:
                    msg = (
                        "Synset category %s either not part of ShapeNetCore dataset "
                        "or cannot be found in %s."
                    ) % (synset, data_dir)
                    warnings.warn(msg)
        # If no category is given, load every category in the given directory.
        # Ignore synset folders not included in the official mapping.
        else:
            synset_set = {
                synset
                for synset in os.listdir(data_dir)
                if path.isdir(path.join(data_dir, synset))
                and synset in self.synset_dict
            }

        # Check if there are any categories in the official mapping that are not loaded.
        # Update self.synset_inv so that it only includes the loaded categories.
        synset_not_present = set(self.synset_dict.keys()).difference(synset_set)
        [self.synset_inv.pop(self.synset_dict[synset]) for synset in synset_not_present]

        if len(synset_not_present) > 0:
            msg = (
                "The following categories are included in ShapeNetCore ver.%d's "
                "official mapping but not found in the dataset location %s: %s"
                ""
            ) % (version, data_dir, ", ".join(synset_not_present))
            warnings.warn(msg)

        # Extract model_id of each object from directory names.
        # Each grandchildren directory of data_dir contains an object, and the name
        # of the directory is the object's model_id.
        for synset in synset_set:
            self.synset_start_idxs[synset] = len(self.synset_ids)
            for model in os.listdir(path.join(data_dir, synset)):
                if not path.exists(path.join(data_dir, synset, model, self.model_dir)):
                    msg = (
                        "Object file not found in the model directory %s "
                        "under synset directory %s."
                    ) % (model, synset)
                    warnings.warn(msg)
                    continue
                self.synset_ids.append(synset)
                self.model_ids.append(model)
            model_count = len(self.synset_ids) - self.synset_start_idxs[synset]
            self.synset_num_models[synset] = model_count

    def __getitem__(self, idx: int) -> Dict:
        """
        Read a model by the given index.

        Args:
            idx: The idx of the model to be retrieved in the dataset.

        Returns:
            dictionary with following keys:
            - verts: FloatTensor of shape (V, 3).
            - faces: LongTensor of shape (F, 3) which indexes into the verts tensor.
            - synset_id (str): synset id
            - model_id (str): model id
            - label (str): synset label.
        """
        model = self._get_item_ids(idx)
        model_path = path.join(
            self.shapenet_dir, model["synset_id"], model["model_id"], self.model_dir
        )
        verts, faces, textures = self._load_mesh(model_path)
        model["verts"] = verts
        model["faces"] = faces
        model["textures"] = textures
        model["label"] = self.synset_dict[model["synset_id"]]
        return model
