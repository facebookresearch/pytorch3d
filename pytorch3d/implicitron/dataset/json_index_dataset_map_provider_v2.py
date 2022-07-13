# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import json
import logging
import os
import warnings
from typing import Dict, List, Optional, Type

from pytorch3d.implicitron.dataset.dataset_map_provider import (
    DatasetMap,
    DatasetMapProviderBase,
    PathManagerFactory,
    Task,
)
from pytorch3d.implicitron.dataset.json_index_dataset import JsonIndexDataset
from pytorch3d.implicitron.tools.config import (
    expand_args_fields,
    registry,
    run_auto_creation,
)

from pytorch3d.renderer.cameras import CamerasBase


_CO3DV2_DATASET_ROOT: str = os.getenv("CO3DV2_DATASET_ROOT", "")


logger = logging.getLogger(__name__)


@registry.register
class JsonIndexDatasetMapProviderV2(DatasetMapProviderBase):  # pyre-ignore [13]
    """
    Generates the training, validation, and testing dataset objects for
    a dataset laid out on disk like CO3Dv2, with annotations in gzipped json files.

    The dataset is organized in the filesystem as follows:
        ```
        self.dataset_root
            ├── <category_0>
            │   ├── <sequence_name_0>
            │   │   ├── depth_masks
            │   │   ├── depths
            │   │   ├── images
            │   │   ├── masks
            │   │   └── pointcloud.ply
            │   ├── <sequence_name_1>
            │   │   ├── depth_masks
            │   │   ├── depths
            │   │   ├── images
            │   │   ├── masks
            │   │   └── pointcloud.ply
            │   ├── ...
            │   ├── <sequence_name_N>
            │   ├── set_lists
            │       ├── set_lists_<subset_name_0>.json
            │       ├── set_lists_<subset_name_1>.json
            │       ├── ...
            │       ├── set_lists_<subset_name_M>.json
            │   ├── eval_batches
            │   │   ├── eval_batches_<subset_name_0>.json
            │   │   ├── eval_batches_<subset_name_1>.json
            │   │   ├── ...
            │   │   ├── eval_batches_<subset_name_M>.json
            │   ├── frame_annotations.jgz
            │   ├── sequence_annotations.jgz
            ├── <category_1>
            ├── ...
            ├── <category_K>
        ```

    The dataset contains sequences named `<sequence_name_i>` from `K` categories with
    names `<category_j>`. Each category comprises sequence folders
    `<category_k>/<sequence_name_i>` containing the list of sequence images, depth maps,
    foreground masks, and valid-depth masks `images`, `depths`, `masks`, and `depth_masks`
    respectively. Furthermore, `<category_k>/<sequence_name_i>/set_lists/` stores `M`
    json files `set_lists_<subset_name_l>.json`, each describing a certain sequence subset.

    Users specify the loaded dataset subset by setting `self.subset_name` to one of the
    available subset names `<subset_name_l>`.

    `frame_annotations.jgz` and `sequence_annotations.jgz` are gzipped json files containing
    the list of all frames and sequences of the given category stored as lists of
    `FrameAnnotation` and `SequenceAnnotation` objects respectivelly.

    Each `set_lists_<subset_name_l>.json` file contains the following dictionary:
        ```
        {
            "train": [
                (sequence_name: str, frame_number: int, image_path: str),
                ...
            ],
            "val": [
                (sequence_name: str, frame_number: int, image_path: str),
                ...
            ],
            "test": [
                (sequence_name: str, frame_number: int, image_path: str),
                ...
            ],
        ]
        ```
    defining the list of frames (identified with their `sequence_name` and `frame_number`)
    in the "train", "val", and "test" subsets of the dataset.
    Note that `frame_number` can be obtained only from `frame_annotations.jgz` and
    does not necesarrily correspond to the numeric suffix of the corresponding image
    file name (e.g. a file `<category_0>/<sequence_name_0>/images/frame00005.jpg` can
    have its frame number set to `20`, not 5).

    Each `eval_batches_<subset_name_l>.json` file contains a list of evaluation examples
    in the following form:
        ```
        [
            [  # batch 1
                (sequence_name: str, frame_number: int, image_path: str),
                ...
            ],
            [  # batch 1
                (sequence_name: str, frame_number: int, image_path: str),
                ...
            ],
        ]
        ```
    Note that the evaluation examples always come from the `"test"` subset of the dataset.
    (test frames can repeat across batches).

    Args:
        category: The object category of the dataset.
        subset_name: The name of the dataset subset. For CO3Dv2, these include
            e.g. "manyview_dev_0", "fewview_test", ...
        dataset_root: The root folder of the dataset.
        test_on_train: Construct validation and test datasets from
            the training subset.
        only_test_set: Load only the test set. Incompatible with `test_on_train`.
        load_eval_batches: Load the file containing eval batches pointing to the
            test dataset.
        dataset_args: Specifies additional arguments to the
            JsonIndexDataset constructor call.
        path_manager_factory: (Optional) An object that generates an instance of
            PathManager that can translate provided file paths.
        path_manager_factory_class_type: The class type of `path_manager_factory`.
    """

    category: str
    subset_name: str
    dataset_root: str = _CO3DV2_DATASET_ROOT

    test_on_train: bool = False
    only_test_set: bool = False
    load_eval_batches: bool = True

    dataset_class_type: str = "JsonIndexDataset"
    dataset: JsonIndexDataset

    path_manager_factory: PathManagerFactory
    path_manager_factory_class_type: str = "PathManagerFactory"

    def __post_init__(self):
        super().__init__()
        run_auto_creation(self)

        if self.only_test_set and self.test_on_train:
            raise ValueError("Cannot have only_test_set and test_on_train")

        frame_file = os.path.join(
            self.dataset_root, self.category, "frame_annotations.jgz"
        )
        sequence_file = os.path.join(
            self.dataset_root, self.category, "sequence_annotations.jgz"
        )

        path_manager = self.path_manager_factory.get()

        # setup the common dataset arguments
        common_dataset_kwargs = getattr(self, f"dataset_{self.dataset_class_type}_args")
        common_dataset_kwargs = {
            **common_dataset_kwargs,
            "dataset_root": self.dataset_root,
            "frame_annotations_file": frame_file,
            "sequence_annotations_file": sequence_file,
            "subsets": None,
            "subset_lists_file": "",
            "path_manager": path_manager,
        }

        # get the used dataset type
        dataset_type: Type[JsonIndexDataset] = registry.get(
            JsonIndexDataset, self.dataset_class_type
        )
        expand_args_fields(dataset_type)

        dataset = dataset_type(**common_dataset_kwargs)

        available_subset_names = self._get_available_subset_names()
        logger.debug(f"Available subset names: {str(available_subset_names)}.")
        if self.subset_name not in available_subset_names:
            raise ValueError(
                f"Unknown subset name {self.subset_name}."
                + f" Choose one of available subsets: {str(available_subset_names)}."
            )

        # load the list of train/val/test frames
        subset_mapping = self._load_annotation_json(
            os.path.join(
                self.category, "set_lists", f"set_lists_{self.subset_name}.json"
            )
        )

        # load the evaluation batches
        if self.load_eval_batches:
            eval_batch_index = self._load_annotation_json(
                os.path.join(
                    self.category,
                    "eval_batches",
                    f"eval_batches_{self.subset_name}.json",
                )
            )

        train_dataset = None
        if not self.only_test_set:
            # load the training set
            logger.debug("Extracting train dataset.")
            train_dataset = dataset.subset_from_frame_index(subset_mapping["train"])
            logger.info(f"Train dataset: {str(train_dataset)}")

        if self.test_on_train:
            assert train_dataset is not None
            val_dataset = test_dataset = train_dataset
        else:
            # load the val and test sets
            logger.debug("Extracting val dataset.")
            val_dataset = dataset.subset_from_frame_index(subset_mapping["val"])
            logger.info(f"Val dataset: {str(val_dataset)}")
            logger.debug("Extracting test dataset.")
            test_dataset = dataset.subset_from_frame_index(subset_mapping["test"])
            logger.info(f"Test dataset: {str(test_dataset)}")
            if self.load_eval_batches:
                # load the eval batches
                logger.debug("Extracting eval batches.")
                try:
                    test_dataset.eval_batches = (
                        test_dataset.seq_frame_index_to_dataset_index(
                            eval_batch_index,
                        )
                    )
                except IndexError:
                    warnings.warn(
                        "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n"
                        + "Some eval batches are missing from the test dataset.\n"
                        + "The evaluation results will be incomparable to the\n"
                        + "evaluation results calculated on the original dataset.\n"
                        + "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
                    )
                    test_dataset.eval_batches = (
                        test_dataset.seq_frame_index_to_dataset_index(
                            eval_batch_index,
                            allow_missing_indices=True,
                            remove_missing_indices=True,
                        )
                    )
                logger.info(f"# eval batches: {len(test_dataset.eval_batches)}")

        self.dataset_map = DatasetMap(
            train=train_dataset, val=val_dataset, test=test_dataset
        )

    def create_dataset(self):
        # The dataset object is created inside `self.get_dataset_map`
        pass

    def get_dataset_map(self) -> DatasetMap:
        return self.dataset_map  # pyre-ignore [16]

    def get_category_to_subset_name_list(self) -> Dict[str, List[str]]:
        """
        Returns a global dataset index containing the available subset names per category
        as a dictionary.

        Returns:
            category_to_subset_name_list: A dictionary containing subset names available
                per category of the following form:
                    ```
                    {
                        category_0: [category_0_subset_name_0, category_0_subset_name_1, ...],
                        category_1: [category_1_subset_name_0, category_1_subset_name_1, ...],
                        ...
                    }
                    ```

        """
        category_to_subset_name_list_json = "category_to_subset_name_list.json"
        category_to_subset_name_list = self._load_annotation_json(
            category_to_subset_name_list_json
        )
        return category_to_subset_name_list

    def get_task(self) -> Task:  # TODO: we plan to get rid of tasks
        return {
            "manyview": Task.SINGLE_SEQUENCE,
            "fewview": Task.MULTI_SEQUENCE,
        }[self.subset_name.split("_")[0]]

    def get_all_train_cameras(self) -> Optional[CamerasBase]:
        # pyre-ignore[16]
        train_dataset = self.dataset_map.train
        assert isinstance(train_dataset, JsonIndexDataset)
        return train_dataset.get_all_train_cameras()

    def _load_annotation_json(self, json_filename: str):
        full_path = os.path.join(
            self.dataset_root,
            json_filename,
        )
        logger.info(f"Loading frame index json from {full_path}.")
        path_manager = self.path_manager_factory.get()
        if path_manager is not None:
            full_path = path_manager.get_local_path(full_path)
        if not os.path.isfile(full_path):
            # The batch indices file does not exist.
            # Most probably the user has not specified the root folder.
            raise ValueError(
                f"Looking for dataset json file in {full_path}. "
                + "Please specify a correct dataset_root folder."
            )
        with open(full_path, "r") as f:
            data = json.load(f)
        return data

    def _get_available_subset_names(self):
        path_manager = self.path_manager_factory.get()
        if path_manager is not None:
            dataset_root = path_manager.get_local_path(self.dataset_root)
        else:
            dataset_root = self.dataset_root
        return get_available_subset_names(dataset_root, self.category)


def get_available_subset_names(dataset_root: str, category: str) -> List[str]:
    """
    Get the available subset names for a given category folder inside a root dataset
    folder `dataset_root`.
    """
    category_dir = os.path.join(dataset_root, category)
    if not os.path.isdir(category_dir):
        raise ValueError(
            f"Looking for dataset files in {category_dir}. "
            + "Please specify a correct dataset_root folder."
        )
    set_list_jsons = os.listdir(os.path.join(category_dir, "set_lists"))
    return [
        json_file.replace("set_lists_", "").replace(".json", "")
        for json_file in set_list_jsons
    ]
