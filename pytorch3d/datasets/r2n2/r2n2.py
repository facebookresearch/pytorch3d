# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import json
import warnings
from os import path
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from pytorch3d.datasets.shapenet_base import ShapeNetBase
from pytorch3d.io import load_obj
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.transforms import Transform3d
from tabulate import tabulate


SYNSET_DICT_DIR = Path(__file__).resolve().parent

# Default values of rotation, translation and intrinsic matrices for BlenderCamera.
r = np.expand_dims(np.eye(3), axis=0)  # (1, 3, 3)
t = np.expand_dims(np.zeros(3), axis=0)  # (1, 3)
k = np.expand_dims(np.eye(4), axis=0)  # (1, 4, 4)


class R2N2(ShapeNetBase):
    """
    This class loads the R2N2 dataset from a given directory into a Dataset object.
    The R2N2 dataset contains 13 categories that are a subset of the ShapeNetCore v.1
    dataset. The R2N2 dataset also contains its own 24 renderings of each object and
    voxelized models. Most of the models have all 24 views in the same split, but there
    are eight of them that divide their views between train and test splits.
    """

    def __init__(
        self,
        split: str,
        shapenet_dir,
        r2n2_dir,
        splits_file,
        return_all_views: bool = True,
    ):
        """
        Store each object's synset id and models id the given directories.

        Args:
            split (str): One of (train, val, test).
            shapenet_dir (path): Path to ShapeNet core v1.
            r2n2_dir (path): Path to the R2N2 dataset.
            splits_file (path): File containing the train/val/test splits.
            return_all_views (bool): Indicator of whether or not to load all the views in
                the split. If set to False, one of the views in the split will be randomly
                selected and loaded.
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

        self.return_images = True
        # Check if the folder containing R2N2 renderings is included in r2n2_dir.
        if not path.isdir(path.join(r2n2_dir, "ShapeNetRendering")):
            self.return_images = False
            msg = (
                "ShapeNetRendering not found in %s. R2N2 renderings will "
                "be skipped when returning models."
            ) % (r2n2_dir)
            warnings.warn(msg)

        synset_set = set()
        # Store lists of views of each model in a list.
        self.views_per_model_list = []
        # Store tuples of synset label and total number of views in each category in a list.
        synset_num_instances = []
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
            self.synset_start_idxs[synset] = len(self.synset_ids)
            # Start counting total number of views in the current category.
            synset_view_count = 0
            for model in split_dict[synset]:
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

                model_views = split_dict[synset][model]
                # Randomly select a view index if return_all_views set to False.
                if not return_all_views:
                    rand_idx = torch.randint(len(model_views), (1,))
                    model_views = [model_views[rand_idx]]
                self.views_per_model_list.append(model_views)
                synset_view_count += len(model_views)
            synset_num_instances.append((self.synset_dict[synset], synset_view_count))
            model_count = len(self.synset_ids) - self.synset_start_idxs[synset]
            self.synset_num_models[synset] = model_count
        headers = ["category", "#instances"]
        synset_num_instances.append(("total", sum(n for _, n in synset_num_instances)))
        print(
            tabulate(synset_num_instances, headers, numalign="left", stralign="center")
        )

        # Examine if all the synsets in the standard R2N2 mapping are present.
        # Update self.synset_inv so that it only includes the loaded categories.
        synset_not_present = [
            self.synset_inv.pop(self.synset_dict[synset])
            for synset in self.synset_dict
            if synset not in synset_set
        ]
        if len(synset_not_present) > 0:
            msg = (
                "The following categories are included in R2N2's"
                "official mapping but not found in the dataset location %s: %s"
            ) % (shapenet_dir, ", ".join(synset_not_present))
            warnings.warn(msg)

    def __getitem__(self, model_idx, view_idxs: Optional[List[int]] = None) -> Dict:
        """
        Read a model by the given index.

        Args:
            model_idx: The idx of the model to be retrieved in the dataset.
            view_idx: List of indices of the view to be returned. Each index needs to be
                contained in the loaded split (always between 0 and 23, inclusive). If
                an invalid index is supplied, view_idx will be ignored and all the loaded
                views will be returned.

        Returns:
            dictionary with following keys:
            - verts: FloatTensor of shape (V, 3).
            - faces: faces.verts_idx, LongTensor of shape (F, 3).
            - synset_id (str): synset id.
            - model_id (str): model id.
            - label (str): synset label.
            - images: FloatTensor of shape (V, H, W, C), where V is number of views
                returned. Returns a batch of the renderings of the models from the R2N2 dataset.
        """
        if isinstance(model_idx, tuple):
            model_idx, view_idxs = model_idx
        if view_idxs is not None:
            if isinstance(view_idxs, int):
                view_idxs = [view_idxs]
            if not isinstance(view_idxs, list) and not torch.is_tensor(view_idxs):
                raise TypeError(
                    "view_idxs is of type %s but it needs to be a list."
                    % type(view_idxs)
                )

        model_views = self.views_per_model_list[model_idx]
        if view_idxs is not None and any(
            idx not in self.views_per_model_list[model_idx] for idx in view_idxs
        ):
            msg = """At least one of the indices in view_idxs is not available.
                Specified view of the model needs to be contained in the
                loaded split. If return_all_views is set to False, only one
                random view is loaded. Try accessing the specified view(s)
                after loading the dataset with self.return_all_views set to True.
                Now returning all view(s) in the loaded dataset."""
            warnings.warn(msg)
        elif view_idxs is not None:
            model_views = view_idxs

        model = self._get_item_ids(model_idx)
        model_path = path.join(
            self.shapenet_dir, model["synset_id"], model["model_id"], "model.obj"
        )
        model["verts"], faces, _ = load_obj(model_path)
        model["faces"] = faces.verts_idx
        model["label"] = self.synset_dict[model["synset_id"]]

        model["images"] = None
        # Retrieve R2N2's renderings if required.
        if self.return_images:
            rendering_path = path.join(
                self.r2n2_dir,
                "ShapeNetRendering",
                model["synset_id"],
                model["model_id"],
                "rendering",
            )

            images = []
            for i in model_views:
                # Read image.
                image_path = path.join(rendering_path, "%02d.png" % i)
                raw_img = Image.open(image_path)
                image = torch.from_numpy(np.array(raw_img) / 255.0)[..., :3]
                images.append(image.to(dtype=torch.float32))

            model["images"] = torch.stack(images)

        return model


class BlenderCamera(CamerasBase):
    """
    Camera for rendering objects with calibration matrices from the R2N2 dataset
    (which uses Blender for rendering the views for each model).
    """

    def __init__(self, R=r, T=t, K=k, device="cpu"):
        """
        Args:
            R: Rotation matrix of shape (N, 3, 3).
            T: Translation matrix of shape (N, 3).
            K: Intrinsic matrix of shape (N, 4, 4).
            device: torch.device or str.
        """
        # The initializer formats all inputs to torch tensors and broadcasts
        # all the inputs to have the same batch dimension where necessary.
        super().__init__(device=device, R=R, T=T, K=K)

    def get_projection_transform(self, **kwargs) -> Transform3d:
        transform = Transform3d(device=self.device)
        transform._matrix = self.K.transpose(1, 2).contiguous()  # pyre-ignore[16]
        return transform
