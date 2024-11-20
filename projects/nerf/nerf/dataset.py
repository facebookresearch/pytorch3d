# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import List, Optional, Tuple

import numpy as np
import requests
import torch
from PIL import Image
from pytorch3d.renderer import PerspectiveCameras
from torch.utils.data import Dataset


DEFAULT_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "data"
)

DEFAULT_URL_ROOT = "https://dl.fbaipublicfiles.com/pytorch3d_nerf_data"

ALL_DATASETS = ("lego", "fern", "pt3logo")


def trivial_collate(batch):
    """
    A trivial collate function that merely returns the uncollated batch.
    """
    return batch


class ListDataset(Dataset):
    """
    A simple dataset made of a list of entries.
    """

    def __init__(self, entries: List) -> None:
        """
        Args:
            entries: The list of dataset entries.
        """
        self._entries = entries

    def __len__(
        self,
    ) -> int:
        return len(self._entries)

    def __getitem__(self, index):
        return self._entries[index]


def get_nerf_datasets(
    dataset_name: str,  # 'lego | fern'
    image_size: Tuple[int, int],
    data_root: str = DEFAULT_DATA_ROOT,
    autodownload: bool = True,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Obtains the training and validation dataset object for a dataset specified
    with the `dataset_name` argument.

    Args:
        dataset_name: The name of the dataset to load.
        image_size: A tuple (height, width) denoting the sizes of the loaded dataset images.
        data_root: The root folder at which the data is stored.
        autodownload: Auto-download the dataset files in case they are missing.

    Returns:
        train_dataset: The training dataset object.
        val_dataset: The validation dataset object.
        test_dataset: The testing dataset object.
    """

    if dataset_name not in ALL_DATASETS:
        raise ValueError(f"'{dataset_name}'' does not refer to a known dataset.")

    print(f"Loading dataset {dataset_name}, image size={str(image_size)} ...")

    cameras_path = os.path.join(data_root, dataset_name + ".pth")
    image_path = cameras_path.replace(".pth", ".png")

    if autodownload and any(not os.path.isfile(p) for p in (cameras_path, image_path)):
        # Automatically download the data files if missing.
        download_data([dataset_name], data_root=data_root)

    train_data = torch.load(cameras_path, weights_only=True)
    n_cameras = train_data["cameras"]["R"].shape[0]

    _image_max_image_pixels = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None  # The dataset image is very large ...
    images = torch.FloatTensor(np.array(Image.open(image_path))) / 255.0
    images = torch.stack(torch.chunk(images, n_cameras, dim=0))[..., :3]
    Image.MAX_IMAGE_PIXELS = _image_max_image_pixels

    scale_factors = [s_new / s for s, s_new in zip(images.shape[1:3], image_size)]
    if abs(scale_factors[0] - scale_factors[1]) > 1e-3:
        raise ValueError(
            "Non-isotropic scaling is not allowed. Consider changing the 'image_size' argument."
        )
    scale_factor = sum(scale_factors) * 0.5

    if scale_factor != 1.0:
        print(f"Rescaling dataset (factor={scale_factor})")
        images = torch.nn.functional.interpolate(
            images.permute(0, 3, 1, 2),
            size=tuple(image_size),
            mode="bilinear",
        ).permute(0, 2, 3, 1)

    cameras = [
        PerspectiveCameras(
            **{k: v[cami][None] for k, v in train_data["cameras"].items()}
        ).to("cpu")
        for cami in range(n_cameras)
    ]

    train_idx, val_idx, test_idx = train_data["split"]

    train_dataset, val_dataset, test_dataset = [
        ListDataset(
            [
                {"image": images[i], "camera": cameras[i], "camera_idx": int(i)}
                for i in idx
            ]
        )
        for idx in [train_idx, val_idx, test_idx]
    ]

    return train_dataset, val_dataset, test_dataset


def download_data(
    dataset_names: Optional[List[str]] = None,
    data_root: str = DEFAULT_DATA_ROOT,
    url_root: str = DEFAULT_URL_ROOT,
) -> None:
    """
    Downloads the relevant dataset files.

    Args:
        dataset_names: A list of the names of datasets to download. If `None`,
            downloads all available datasets.
    """

    if dataset_names is None:
        dataset_names = ALL_DATASETS

    os.makedirs(data_root, exist_ok=True)

    for dataset_name in dataset_names:
        cameras_file = dataset_name + ".pth"
        images_file = cameras_file.replace(".pth", ".png")
        license_file = cameras_file.replace(".pth", "_license.txt")

        for fl in (cameras_file, images_file, license_file):
            local_fl = os.path.join(data_root, fl)
            remote_fl = os.path.join(url_root, fl)

            print(f"Downloading dataset {dataset_name} from {remote_fl} to {local_fl}.")

            r = requests.get(remote_fl)
            with open(local_fl, "wb") as f:
                f.write(r.content)
