# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import os
import warnings
from os import path

import torch
from pytorch3d.io import load_obj


class ShapeNetCore(torch.utils.data.Dataset):
    """
    This class loads ShapeNet v.1 from a given directory into a Dataset object.
    """

    def __init__(self, data_dir):
        """
        Stores each object's synset id and models id from data_dir.
        Args:
            data_dir (path): Path to shapenet data
        """
        self.data_dir = data_dir

        # List of subdirectories of data_dir each containing a category of models.
        # The name of each subdirectory is the wordnet synset offset of that category.
        wnsynset_list = [
            wnsynset
            for wnsynset in os.listdir(data_dir)
            if path.isdir(path.join(data_dir, wnsynset))
        ]

        # Extract synset_id and model_id of each object from directory names.
        # Each grandchildren directory of data_dir contains an object, and the name
        # of the directory is the object's model_id.
        self.synset_ids = []
        self.model_ids = []
        for synset in wnsynset_list:
            for model in os.listdir(path.join(data_dir, synset)):
                if not path.exists(path.join(data_dir, synset, model, "model.obj")):
                    msg = """ model.obj not found in the model directory %s
                        under synset directory %s.""" % (
                        model,
                        synset,
                    )
                    warnings.warn(msg)
                else:
                    self.synset_ids.append(synset)
                    self.model_ids.append(model)

    def __len__(self):
        """
        Returns # of total models in shapenet core
        """
        return len(self.model_ids)

    def __getitem__(self, idx):
        """
        Read a model by the given index.
        Returns:
            dictionary with following keys:
            - verts: FloatTensor of shape (V, 3).
            - faces: LongTensor of shape (F, 3) which indexes into the verts tensor.
            - synset_id (str): synset id
            - model_id (str): model id
        """
        model = {}
        model["synset_id"] = self.synset_ids[idx]
        model["model_id"] = self.model_ids[idx]
        model_path = path.join(
            self.data_dir, model["synset_id"], model["model_id"], "model.obj"
        )
        model["verts"], faces, _ = load_obj(model_path)
        model["faces"] = faces.verts_idx
        return model
