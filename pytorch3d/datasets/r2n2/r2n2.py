# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import json
import warnings
from os import path
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from pytorch3d.common.datatypes import Device
from pytorch3d.datasets.shapenet_base import ShapeNetBase
from pytorch3d.renderer import HardPhongShader
from tabulate import tabulate

from .utils import (
    align_bbox,
    BlenderCamera,
    compute_extrinsic_matrix,
    read_binvox_coords,
    voxelize,
)


SYNSET_DICT_DIR = Path(__file__).resolve().parent
MAX_CAMERA_DISTANCE = 1.75  # Constant from R2N2.
VOXEL_SIZE = 128
# Intrinsic matrix extracted from Blender. Taken from meshrcnn codebase:
# https://github.com/facebookresearch/meshrcnn/blob/main/shapenet/utils/coords.py
BLENDER_INTRINSIC = torch.tensor(
    [
        [2.1875, 0.0, 0.0, 0.0],
        [0.0, 2.1875, 0.0, 0.0],
        [0.0, 0.0, -1.002002, -0.2002002],
        [0.0, 0.0, -1.0, 0.0],
    ]
)


class R2N2(ShapeNetBase):  # pragma: no cover
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
        shapenet_dir: str,
        r2n2_dir: str,
        splits_file: str,
        return_all_views: bool = True,
        return_voxels: bool = False,
        views_rel_path: str = "ShapeNetRendering",
        voxels_rel_path: str = "ShapeNetVoxels",
        load_textures: bool = True,
        texture_resolution: int = 4,
    ) -> None:
        """
        Store each object's synset id and models id the given directories.

        Args:
            split (str): One of (train, val, test).
            shapenet_dir (str): Path to ShapeNet core v1.
            r2n2_dir (str): Path to the R2N2 dataset.
            splits_file (str): File containing the train/val/test splits.
            return_all_views (bool): Indicator of whether or not to load all the views in
                the split. If set to False, one of the views in the split will be randomly
                selected and loaded.
            return_voxels(bool): Indicator of whether or not to return voxels as a tensor
                of shape (D, D, D) where D is the number of voxels along each dimension.
            views_rel_path: path to rendered views within the r2n2_dir. If not specified,
                the renderings are assumed to be at os.path.join(rn2n_dir, "ShapeNetRendering").
            voxels_rel_path: path to rendered views within the r2n2_dir. If not specified,
                the renderings are assumed to be at os.path.join(rn2n_dir, "ShapeNetVoxels").
            load_textures: Boolean indicating whether textures should loaded for the model.
                Textures will be of type TexturesAtlas i.e. a texture map per face.
            texture_resolution: Int specifying the resolution of the texture map per face
                created using the textures in the obj file. A
                (texture_resolution, texture_resolution, 3) map is created per face.

        """
        super().__init__()
        self.shapenet_dir = shapenet_dir
        self.r2n2_dir = r2n2_dir
        self.views_rel_path = views_rel_path
        self.voxels_rel_path = voxels_rel_path
        self.load_textures = load_textures
        self.texture_resolution = texture_resolution
        # Examine if split is valid.
        if split not in ["train", "val", "test"]:
            raise ValueError("split has to be one of (train, val, test).")
        # Synset dictionary mapping synset offsets in R2N2 to corresponding labels.
        with open(
            path.join(SYNSET_DICT_DIR, "r2n2_synset_dict.json"), "r"
        ) as read_dict:
            self.synset_dict = json.load(read_dict)
        # Inverse dictionary mapping synset labels to corresponding offsets.
        self.synset_inv = {label: offset for offset, label in self.synset_dict.items()}

        # Store synset and model ids of objects mentioned in the splits_file.
        with open(splits_file) as splits:
            split_dict = json.load(splits)[split]

        self.return_images = True
        # Check if the folder containing R2N2 renderings is included in r2n2_dir.
        if not path.isdir(path.join(r2n2_dir, views_rel_path)):
            self.return_images = False
            msg = (
                "%s not found in %s. R2N2 renderings will "
                "be skipped when returning models."
            ) % (views_rel_path, r2n2_dir)
            warnings.warn(msg)

        self.return_voxels = return_voxels
        # Check if the folder containing voxel coordinates is included in r2n2_dir.
        if not path.isdir(path.join(r2n2_dir, voxels_rel_path)):
            self.return_voxels = False
            msg = (
                "%s not found in %s. Voxel coordinates will "
                "be skipped when returning models."
            ) % (voxels_rel_path, r2n2_dir)
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
            - R: Rotation matrix of shape (V, 3, 3), where V is number of views returned.
            - T: Translation matrix of shape (V, 3), where V is number of views returned.
            - K: Intrinsic matrix of shape (V, 4, 4), where V is number of views returned.
            - voxels: Voxels of shape (D, D, D), where D is the number of voxels along each
                dimension.
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

        verts, faces, textures = self._load_mesh(model_path)
        model["verts"] = verts
        model["faces"] = faces
        model["textures"] = textures
        model["label"] = self.synset_dict[model["synset_id"]]

        model["images"] = None
        images, Rs, Ts, voxel_RTs = [], [], [], []
        # Retrieve R2N2's renderings if required.
        if self.return_images:
            rendering_path = path.join(
                self.r2n2_dir,
                self.views_rel_path,
                model["synset_id"],
                model["model_id"],
                "rendering",
            )
            # Read metadata file to obtain params for calibration matrices.
            with open(path.join(rendering_path, "rendering_metadata.txt"), "r") as f:
                metadata_lines = f.readlines()
            for i in model_views:
                # Read image.
                image_path = path.join(rendering_path, "%02d.png" % i)
                raw_img = Image.open(image_path)
                image = torch.from_numpy(np.array(raw_img) / 255.0)[..., :3]
                images.append(image.to(dtype=torch.float32))

                # Get camera calibration.
                azim, elev, yaw, dist_ratio, fov = [
                    float(v) for v in metadata_lines[i].strip().split(" ")
                ]
                dist = dist_ratio * MAX_CAMERA_DISTANCE
                # Extrinsic matrix before transformation to PyTorch3D world space.
                RT = compute_extrinsic_matrix(azim, elev, dist)
                R, T = self._compute_camera_calibration(RT)
                Rs.append(R)
                Ts.append(T)
                voxel_RTs.append(RT)

            # Intrinsic matrix extracted from the Blender with slight modification to work with
            # PyTorch3D world space. Taken from meshrcnn codebase:
            # https://github.com/facebookresearch/meshrcnn/blob/main/shapenet/utils/coords.py
            K = torch.tensor(
                [
                    [2.1875, 0.0, 0.0, 0.0],
                    [0.0, 2.1875, 0.0, 0.0],
                    [0.0, 0.0, -1.002002, -0.2002002],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            )
            model["images"] = torch.stack(images)
            model["R"] = torch.stack(Rs)
            model["T"] = torch.stack(Ts)
            model["K"] = K.expand(len(model_views), 4, 4)

        voxels_list = []

        # Read voxels if required.
        voxel_path = path.join(
            self.r2n2_dir,
            self.voxels_rel_path,
            model["synset_id"],
            model["model_id"],
            "model.binvox",
        )
        if self.return_voxels:
            if not path.isfile(voxel_path):
                msg = "Voxel file not found for model %s from category %s."
                raise FileNotFoundError(msg % (model["model_id"], model["synset_id"]))

            with open(voxel_path, "rb") as f:
                # Read voxel coordinates as a tensor of shape (N, 3).
                voxel_coords = read_binvox_coords(f)
            # Align voxels to the same coordinate system as mesh verts.
            voxel_coords = align_bbox(voxel_coords, model["verts"])
            for RT in voxel_RTs:
                # Compute projection matrix.
                P = BLENDER_INTRINSIC.mm(RT)
                # Convert voxel coordinates of shape (N, 3) to voxels of shape (D, D, D).
                voxels = voxelize(voxel_coords, P, VOXEL_SIZE)
                voxels_list.append(voxels)
            model["voxels"] = torch.stack(voxels_list)

        return model

    def _compute_camera_calibration(self, RT):
        """
        Helper function for calculating rotation and translation matrices from ShapeNet
        to camera transformation and ShapeNet to PyTorch3D transformation.

        Args:
            RT: Extrinsic matrix that performs ShapeNet world view to camera view
                transformation.

        Returns:
            R: Rotation matrix of shape (3, 3).
            T: Translation matrix of shape (3).
        """
        # Transform the mesh vertices from shapenet world to pytorch3d world.
        shapenet_to_pytorch3d = torch.tensor(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        RT = torch.transpose(RT, 0, 1).mm(shapenet_to_pytorch3d)  # (4, 4)
        # Extract rotation and translation matrices from RT.
        R = RT[:3, :3]
        T = RT[3, :3]
        return R, T

    def render(
        self,
        model_ids: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        sample_nums: Optional[List[int]] = None,
        idxs: Optional[List[int]] = None,
        view_idxs: Optional[List[int]] = None,
        shader_type=HardPhongShader,
        device: Device = "cpu",
        **kwargs,
    ) -> torch.Tensor:
        """
        Render models with BlenderCamera by default to achieve the same orientations as the
        R2N2 renderings. Also accepts other types of cameras and any of the args that the
        render function in the ShapeNetBase class accepts.

        Args:
            view_idxs: each model will be rendered with the orientation(s) of the specified
                views. Only render by view_idxs if no camera or args for BlenderCamera is
                supplied.
            Accepts any of the args of the render function in ShapeNetBase:
            model_ids: List[str] of model_ids of models intended to be rendered.
            categories: List[str] of categories intended to be rendered. categories
                and sample_nums must be specified at the same time. categories can be given
                in the form of synset offsets or labels, or a combination of both.
            sample_nums: List[int] of number of models to be randomly sampled from
                each category. Could also contain one single integer, in which case it
                will be broadcasted for every category.
            idxs: List[int] of indices of models to be rendered in the dataset.
            shader_type: Shader to use for rendering. Examples include HardPhongShader
            (default), SoftPhongShader etc or any other type of valid Shader class.
            device: Device (as str or torch.device) on which the tensors should be located.
            **kwargs: Accepts any of the kwargs that the renderer supports and any of the
                args that BlenderCamera supports.

        Returns:
            Batch of rendered images of shape (N, H, W, 3).
        """
        idxs = self._handle_render_inputs(model_ids, categories, sample_nums, idxs)
        r = torch.cat([self[idxs[i], view_idxs]["R"] for i in range(len(idxs))])
        t = torch.cat([self[idxs[i], view_idxs]["T"] for i in range(len(idxs))])
        k = torch.cat([self[idxs[i], view_idxs]["K"] for i in range(len(idxs))])
        # Initialize default camera using R, T, K from kwargs or R, T, K of the specified views.
        blend_cameras = BlenderCamera(
            R=kwargs.get("R", r),
            T=kwargs.get("T", t),
            K=kwargs.get("K", k),
            device=device,
        )
        cameras = kwargs.get("cameras", blend_cameras).to(device)
        kwargs.pop("cameras", None)
        # pass down all the same inputs
        return super().render(
            idxs=idxs, shader_type=shader_type, device=device, cameras=cameras, **kwargs
        )
