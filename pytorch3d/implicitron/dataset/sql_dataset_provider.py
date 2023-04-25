# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
from typing import List, Optional, Tuple, Type

import numpy as np

from omegaconf import DictConfig, OmegaConf

from pytorch3d.implicitron.dataset.dataset_map_provider import (
    DatasetMap,
    DatasetMapProviderBase,
    PathManagerFactory,
)
from pytorch3d.implicitron.tools.config import (
    expand_args_fields,
    registry,
    run_auto_creation,
)

from .sql_dataset import SqlIndexDataset


_CO3D_SQL_DATASET_ROOT: str = os.getenv("CO3D_SQL_DATASET_ROOT", "")

# _NEED_CONTROL is a list of those elements of SqlIndexDataset which
# are not directly specified for it in the config but come from the
# DatasetMapProvider.
_NEED_CONTROL: Tuple[str, ...] = (
    "path_manager",
    "subsets",
    "sqlite_metadata_file",
    "subset_lists_file",
)

logger = logging.getLogger(__name__)


@registry.register
class SqlIndexDatasetMapProvider(DatasetMapProviderBase):  # pyre-ignore [13]
    """
    Generates the training, validation, and testing dataset objects for
    a dataset laid out on disk like SQL-CO3D, with annotations in an SQLite data base.

    The dataset is organized in the filesystem as follows::

        self.dataset_root
            ├── <possible/partition/0>
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
            │       ├── <subset_base_name_0>.json
            │       ├── <subset_base_name_1>.json
            │       ├── ...
            │       ├── <subset_base_name_2>.json
            │   ├── eval_batches
            │   │   ├── <eval_batches_base_name_0>.json
            │   │   ├── <eval_batches_base_name_1>.json
            │   │   ├── ...
            │   │   ├── <eval_batches_base_name_M>.json
            │   ├── frame_annotations.jgz
            │   ├── sequence_annotations.jgz
            ├── <possible/partition/1>
            ├── ...
            ├── <possible/partition/K>
            ├── set_lists
                ├── <subset_base_name_0>.sqlite
                ├── <subset_base_name_1>.sqlite
                ├── ...
                ├── <subset_base_name_2>.sqlite
            ├── eval_batches
            │   ├── <eval_batches_base_name_0>.json
            │   ├── <eval_batches_base_name_1>.json
            │   ├── ...
            │   ├── <eval_batches_base_name_M>.json

    The dataset contains sequences named `<sequence_name_i>` that may be partitioned by
    directories such as `<possible/partition/0>` e.g. representing categories but they
    can also be stored in a flat structure. Each sequence folder contains the list of
    sequence images, depth maps, foreground masks, and valid-depth masks
    `images`, `depths`, `masks`, and `depth_masks` respectively. Furthermore,
    `set_lists/` dirtectories (with partitions or global) store json or sqlite files
    `<subset_base_name_l>.<ext>`, each describing a certain sequence subset.
    These subset path conventions are not hard-coded and arbitrary relative path can be
    specified by setting `self.subset_lists_path` to the relative path w.r.t.
    dataset root.

    Each `<subset_base_name_l>.json` file contains the following dictionary::

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

    defining the list of frames (identified with their `sequence_name` and
    `frame_number`) in the "train", "val", and "test" subsets of the dataset. In case of
    SQLite format, `<subset_base_name_l>.sqlite` contains a table with the header::

        | sequence_name | frame_number | image_path | subset |

    Note that `frame_number` can be obtained only from the metadata and
    does not necesarrily correspond to the numeric suffix of the corresponding image
    file name (e.g. a file `<partition_0>/<sequence_name_0>/images/frame00005.jpg` can
    have its frame number set to `20`, not 5).

    Each `<eval_batches_base_name_M>.json` file contains a list of evaluation examples
    in the following form::

        [
            [  # batch 1
                (sequence_name: str, frame_number: int, image_path: str),
                ...
            ],
            [  # batch 2
                (sequence_name: str, frame_number: int, image_path: str),
                ...
            ],
        ]

    Note that the evaluation examples always come from the `"test"` subset of the dataset.
    (test frames can repeat across batches). The batches can contain single element,
    which is typical in case of regular radiance field fitting.

    Args:
        subset_lists_path: The relative path to the dataset subset definition.
            For CO3D, these include e.g. "skateboard/set_lists/set_lists_manyview_dev_0.json".
            By default (None), dataset is not partitioned to subsets (in that case, setting
            `ignore_subsets` will speed up construction)
        dataset_root: The root folder of the dataset.
        metadata_basename: name of the SQL metadata file in dataset_root;
            not expected to be changed by users
        test_on_train: Construct validation and test datasets from
            the training subset; note that in practice, in this
            case all subset dataset objects will be same
        only_test_set: Load only the test set. Incompatible with `test_on_train`.
        ignore_subsets: Don’t filter by subsets in the dataset; note that in this
            case all subset datasets will be same
        eval_batch_num_training_frames: Add a certain number of training frames to each
            eval batch. Useful for evaluating models that require
            source views as input (e.g. NeRF-WCE / PixelNeRF).
        dataset_args: Specifies additional arguments to the
            JsonIndexDataset constructor call.
        path_manager_factory: (Optional) An object that generates an instance of
            PathManager that can translate provided file paths.
        path_manager_factory_class_type: The class type of `path_manager_factory`.
    """

    category: Optional[str] = None
    subset_list_name: Optional[str] = None  # TODO: docs
    # OR
    subset_lists_path: Optional[str] = None
    eval_batches_path: Optional[str] = None

    dataset_root: str = _CO3D_SQL_DATASET_ROOT
    metadata_basename: str = "metadata.sqlite"

    test_on_train: bool = False
    only_test_set: bool = False
    ignore_subsets: bool = False
    train_subsets: Tuple[str, ...] = ("train",)
    val_subsets: Tuple[str, ...] = ("val",)
    test_subsets: Tuple[str, ...] = ("test",)

    eval_batch_num_training_frames: int = 0

    # this is a mould that is never constructed, used to build self._dataset_map values
    dataset_class_type: str = "SqlIndexDataset"
    dataset: SqlIndexDataset

    path_manager_factory: PathManagerFactory
    path_manager_factory_class_type: str = "PathManagerFactory"

    def __post_init__(self):
        super().__init__()
        run_auto_creation(self)

        if self.only_test_set and self.test_on_train:
            raise ValueError("Cannot have only_test_set and test_on_train")

        if self.ignore_subsets and not self.only_test_set:
            self.test_on_train = True  # no point in loading same data 3 times

        path_manager = self.path_manager_factory.get()

        sqlite_metadata_file = os.path.join(self.dataset_root, self.metadata_basename)
        sqlite_metadata_file = _local_path(path_manager, sqlite_metadata_file)

        if not os.path.isfile(sqlite_metadata_file):
            # The sqlite_metadata_file does not exist.
            # Most probably the user has not specified the root folder.
            raise ValueError(
                f"Looking for frame annotations in {sqlite_metadata_file}."
                + " Please specify a correct dataset_root folder."
                + " Note: By default the root folder is taken from the"
                + " CO3D_SQL_DATASET_ROOT environment variable."
            )

        if self.subset_lists_path and self.subset_list_name:
            raise ValueError(
                "subset_lists_path and subset_list_name cannot be both set"
            )

        subset_lists_file = self._get_lists_file("set_lists")

        # setup the common dataset arguments
        common_dataset_kwargs = {
            **getattr(self, f"dataset_{self.dataset_class_type}_args"),
            "sqlite_metadata_file": sqlite_metadata_file,
            "dataset_root": self.dataset_root,
            "subset_lists_file": subset_lists_file,
            "path_manager": path_manager,
        }

        if self.category:
            logger.info(f"Forcing category filter in the datasets to {self.category}")
            common_dataset_kwargs["pick_categories"] = self.category.split(",")

        # get the used dataset type
        dataset_type: Type[SqlIndexDataset] = registry.get(
            SqlIndexDataset, self.dataset_class_type
        )
        expand_args_fields(dataset_type)

        if subset_lists_file is not None and not os.path.isfile(subset_lists_file):
            available_subsets = self._get_available_subsets(
                OmegaConf.to_object(common_dataset_kwargs["pick_categories"])
            )
            msg = f"Cannot find subset list file {self.subset_lists_path}."
            if available_subsets:
                msg += f" Some of the available subsets: {str(available_subsets)}."
            raise ValueError(msg)

        train_dataset = None
        val_dataset = None
        if not self.only_test_set:
            # load the training set
            logger.debug("Constructing train dataset.")
            train_dataset = dataset_type(
                **common_dataset_kwargs, subsets=self._get_subsets(self.train_subsets)
            )
            logger.info(f"Train dataset: {str(train_dataset)}")

        if self.test_on_train:
            assert train_dataset is not None
            val_dataset = test_dataset = train_dataset
        else:
            # load the val and test sets
            if not self.only_test_set:
                # NOTE: this is always loaded in JsonProviderV2
                logger.debug("Extracting val dataset.")
                val_dataset = dataset_type(
                    **common_dataset_kwargs, subsets=self._get_subsets(self.val_subsets)
                )
                logger.info(f"Val dataset: {str(val_dataset)}")

            logger.debug("Extracting test dataset.")
            eval_batches_file = self._get_lists_file("eval_batches")
            del common_dataset_kwargs["eval_batches_file"]
            test_dataset = dataset_type(
                **common_dataset_kwargs,
                subsets=self._get_subsets(self.test_subsets, True),
                eval_batches_file=eval_batches_file,
            )
            logger.info(f"Test dataset: {str(test_dataset)}")

            if (
                eval_batches_file is not None
                and self.eval_batch_num_training_frames > 0
            ):
                self._extend_eval_batches(test_dataset)

        self._dataset_map = DatasetMap(
            train=train_dataset, val=val_dataset, test=test_dataset
        )

    def _get_subsets(self, subsets, is_eval: bool = False):
        if self.ignore_subsets:
            return None

        if is_eval and self.eval_batch_num_training_frames > 0:
            # we will need to have training frames for extended batches
            return list(subsets) + list(self.train_subsets)

        return subsets

    def _extend_eval_batches(self, test_dataset: SqlIndexDataset) -> None:
        rng = np.random.default_rng(seed=0)
        eval_batches = test_dataset.get_eval_batches()
        if eval_batches is None:
            raise ValueError("Eval batches were not loaded!")

        for batch in eval_batches:
            sequence = batch[0][0]
            seq_frames = list(
                test_dataset.sequence_frames_in_order(sequence, self.train_subsets)
            )
            idx_to_add = rng.permutation(len(seq_frames))[
                : self.eval_batch_num_training_frames
            ]
            batch.extend((sequence, seq_frames[a][1]) for a in idx_to_add)

    @classmethod
    def dataset_tweak_args(cls, type, args: DictConfig) -> None:
        """
        Called by get_default_args.
        Certain fields are not exposed on each dataset class
        but rather are controlled by this provider class.
        """
        for key in _NEED_CONTROL:
            del args[key]

    def create_dataset(self):
        # No `dataset` member of this class is created.
        # The dataset(s) live in `self.get_dataset_map`.
        pass

    def get_dataset_map(self) -> DatasetMap:
        return self._dataset_map  # pyre-ignore [16]

    def _get_available_subsets(self, categories: List[str]):
        """
        Get the available subset names for a given category folder (if given) inside
        a root dataset folder `dataset_root`.
        """
        path_manager = self.path_manager_factory.get()

        subsets: List[str] = []
        for prefix in [""] + categories:
            set_list_dir = os.path.join(self.dataset_root, prefix, "set_lists")
            if not (
                (path_manager is not None) and path_manager.isdir(set_list_dir)
            ) and not os.path.isdir(set_list_dir):
                continue

            set_list_files = (os.listdir if path_manager is None else path_manager.ls)(
                set_list_dir
            )
            subsets.extend(os.path.join(prefix, "set_lists", f) for f in set_list_files)

        return subsets

    def _get_lists_file(self, flavor: str) -> Optional[str]:
        if flavor == "eval_batches":
            subset_lists_path = self.eval_batches_path
        else:
            subset_lists_path = self.subset_lists_path

        if not subset_lists_path and not self.subset_list_name:
            return None

        category_elem = ""
        if self.category and "," not in self.category:
            # if multiple categories are given, looking for global set lists
            category_elem = self.category

        subset_lists_path = subset_lists_path or (
            os.path.join(
                category_elem, f"{flavor}", f"{flavor}_{self.subset_list_name}"
            )
        )

        assert subset_lists_path
        path_manager = self.path_manager_factory.get()
        # try absolute path first
        subset_lists_file = _get_local_path_check_extensions(
            subset_lists_path, path_manager
        )
        if subset_lists_file:
            return subset_lists_file

        full_path = os.path.join(self.dataset_root, subset_lists_path)
        subset_lists_file = _get_local_path_check_extensions(full_path, path_manager)

        if not subset_lists_file:
            raise FileNotFoundError(
                f"Subset lists path given but not found: {full_path}"
            )

        return subset_lists_file


def _get_local_path_check_extensions(
    path, path_manager, extensions=("", ".sqlite", ".json")
) -> Optional[str]:
    for ext in extensions:
        local = _local_path(path_manager, path + ext)
        if os.path.isfile(local):
            return local

    return None


def _local_path(path_manager, path: str) -> str:
    if path_manager is None:
        return path
    return path_manager.get_local_path(path)
