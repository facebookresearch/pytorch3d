---
hide_title: true
sidebar_label: Data loaders
---

# Data loaders for common 3D Datasets

### ShapetNetCore

ShapeNet is a dataset of 3D CAD models. ShapeNetCore is a subset of the ShapeNet dataset and can be downloaded from https://www.shapenet.org/. There are two versions ShapeNetCore: v1 (55 categories) and v2 (57 categories).

The PyTorch3D [ShapeNetCore data loader](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/datasets/shapenet/shapenet_core.py) inherits from `torch.utils.data.Dataset`. It takes the path where the ShapeNetCore dataset is stored locally and loads models in the dataset. The ShapeNetCore class loads and returns models with their `categories`, `model_ids`, `vertices` and `faces`. The `ShapeNetCore` data loader also has a customized `render` function that renders models by the specified `model_ids (List[int])`, `categories (List[str])` or `indices (List[int])` with PyTorch3D's differentiable renderer.

The loaded dataset can be passed to `torch.utils.data.DataLoader` with PyTorch3D's customized collate_fn: `collate_batched_meshes` from the `pytorch3d.dataset.utils` module. The `vertices` and `faces` of the models are used to construct a [Meshes](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/structures/meshes.py) object representing the batched meshes. This `Meshes` representation can be easily used with other ops and rendering in PyTorch3D.

### R2N2

The R2N2 dataset contains 13 categories that are a subset of the ShapeNetCore v.1 dataset. The R2N2 dataset also contains its own 24 renderings of each object and voxelized models. The R2N2 Dataset can be downloaded following the instructions [here](http://3d-r2n2.stanford.edu/).

The PyTorch3D [R2N2 data loader](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/datasets/r2n2/r2n2.py) is initialized with the paths to the ShapeNet dataset, the R2N2 dataset and the splits file for R2N2. Just like `ShapeNetCore`, it can be passed to `torch.utils.data.DataLoader` with a customized collate_fn: `collate_batched_R2N2` from the `pytorch3d.dataset.r2n2.utils` module. It returns all the data that `ShapeNetCore` returns, and in addition, it returns the R2N2 renderings (24 views for each model) along with the camera calibration matrices and a voxel representation for each model. Similar to `ShapeNetCore`, it has a customized `render` function that supports rendering specified models with the PyTorch3D differentiable renderer. In addition, it supports rendering models with the same orientations as R2N2's original renderings.
