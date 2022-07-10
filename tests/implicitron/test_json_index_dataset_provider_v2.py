# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import json
import os
import random
import tempfile
import unittest
from typing import List

import numpy as np

import torch
import torchvision
from PIL import Image
from pytorch3d.implicitron.dataset.dataset_base import FrameData
from pytorch3d.implicitron.dataset.json_index_dataset_map_provider_v2 import (
    JsonIndexDatasetMapProviderV2,
)
from pytorch3d.implicitron.dataset.types import (
    dump_dataclass_jgzip,
    FrameAnnotation,
    ImageAnnotation,
    MaskAnnotation,
    SequenceAnnotation,
)
from pytorch3d.implicitron.tools.config import expand_args_fields


class TestJsonIndexDatasetProviderV2(unittest.TestCase):
    def test_random_dataset(self):
        # store random frame annotations
        expand_args_fields(JsonIndexDatasetMapProviderV2)
        categories = ["A", "B"]
        subset_name = "test"
        with tempfile.TemporaryDirectory() as tmpd:
            _make_random_json_dataset_map_provider_v2_data(tmpd, categories)
            for category in categories:
                dataset_provider = JsonIndexDatasetMapProviderV2(
                    category=category,
                    subset_name="test",
                    dataset_root=tmpd,
                )
                dataset_map = dataset_provider.get_dataset_map()
                for set_ in ["train", "val", "test"]:
                    dataloader = torch.utils.data.DataLoader(
                        getattr(dataset_map, set_),
                        batch_size=3,
                        shuffle=True,
                        collate_fn=FrameData.collate,
                    )
                    for _ in dataloader:
                        pass
                category_to_subset_list = (
                    dataset_provider.get_category_to_subset_name_list()
                )
                category_to_subset_list_ = {c: [subset_name] for c in categories}
                self.assertTrue(category_to_subset_list == category_to_subset_list_)


def _make_random_json_dataset_map_provider_v2_data(
    root: str,
    categories: List[str],
    n_frames: int = 8,
    n_sequences: int = 5,
    H: int = 50,
    W: int = 30,
    subset_name: str = "test",
):
    os.makedirs(root, exist_ok=True)
    category_to_subset_list = {}
    for category in categories:
        frame_annotations = []
        sequence_annotations = []
        frame_index = []
        for seq_i in range(n_sequences):
            seq_name = str(seq_i)
            for i in range(n_frames):
                # generate and store image
                imdir = os.path.join(root, category, seq_name, "images")
                os.makedirs(imdir, exist_ok=True)
                img_path = os.path.join(imdir, f"frame{i:05d}.jpg")
                img = torch.rand(3, H, W)
                torchvision.utils.save_image(img, img_path)

                # generate and store mask
                maskdir = os.path.join(root, category, seq_name, "masks")
                os.makedirs(maskdir, exist_ok=True)
                mask_path = os.path.join(maskdir, f"frame{i:05d}.png")
                mask = np.zeros((H, W))
                mask[H // 2 :, W // 2 :] = 1
                Image.fromarray((mask * 255.0).astype(np.uint8), mode="L",).convert(
                    "L"
                ).save(mask_path)

                fa = FrameAnnotation(
                    sequence_name=seq_name,
                    frame_number=i,
                    frame_timestamp=float(i),
                    image=ImageAnnotation(
                        path=img_path.replace(os.path.normpath(root) + "/", ""),
                        size=list(img.shape[-2:]),
                    ),
                    mask=MaskAnnotation(
                        path=mask_path.replace(os.path.normpath(root) + "/", ""),
                        mass=mask.sum().item(),
                    ),
                )
                frame_annotations.append(fa)
                frame_index.append((seq_name, i, fa.image.path))

            sequence_annotations.append(
                SequenceAnnotation(
                    sequence_name=seq_name,
                    category=category,
                )
            )

        dump_dataclass_jgzip(
            os.path.join(root, category, "frame_annotations.jgz"),
            frame_annotations,
        )
        dump_dataclass_jgzip(
            os.path.join(root, category, "sequence_annotations.jgz"),
            sequence_annotations,
        )

        test_frame_index = frame_index[2::3]

        set_list = {
            "train": frame_index[0::3],
            "val": frame_index[1::3],
            "test": test_frame_index,
        }
        set_lists_dir = os.path.join(root, category, "set_lists")
        os.makedirs(set_lists_dir, exist_ok=True)
        set_list_file = os.path.join(set_lists_dir, f"set_lists_{subset_name}.json")
        with open(set_list_file, "w") as f:
            json.dump(set_list, f)

        eval_batches = [random.sample(test_frame_index, 5) for _ in range(10)]
        eval_b_dir = os.path.join(root, category, "eval_batches")
        os.makedirs(eval_b_dir, exist_ok=True)
        eval_b_file = os.path.join(eval_b_dir, f"eval_batches_{subset_name}.json")
        with open(eval_b_file, "w") as f:
            json.dump(eval_batches, f)

        category_to_subset_list[category] = [subset_name]

    with open(os.path.join(root, "category_to_subset_name_list.json"), "w") as f:
        json.dump(category_to_subset_list, f)
