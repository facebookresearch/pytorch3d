# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import json
import os
from typing import Dict, List, Optional, Tuple, Type

from omegaconf import DictConfig
from pytorch3d.implicitron.tools.config import (
    expand_args_fields,
    registry,
    run_auto_creation,
)
from pytorch3d.renderer.cameras import CamerasBase

from .dataset_map_provider import DatasetMap, DatasetMapProviderBase, PathManagerFactory
from .json_index_dataset import JsonIndexDataset

from .utils import (
    DATASET_TYPE_KNOWN,
    DATASET_TYPE_TEST,
    DATASET_TYPE_TRAIN,
    DATASET_TYPE_UNKNOWN,
)


# fmt: off
CO3D_CATEGORIES: List[str] = list(reversed([
    "baseballbat", "banana", "bicycle", "microwave", "tv",
    "cellphone", "toilet", "hairdryer", "couch", "kite", "pizza",
    "umbrella", "wineglass", "laptop",
    "hotdog", "stopsign", "frisbee", "baseballglove",
    "cup", "parkingmeter", "backpack", "toyplane", "toybus",
    "handbag", "chair", "keyboard", "car", "motorcycle",
    "carrot", "bottle", "sandwich", "remote", "bowl", "skateboard",
    "toaster", "mouse", "toytrain", "book", "toytruck",
    "orange", "broccoli", "plant", "teddybear",
    "suitcase", "bench", "ball", "cake",
    "vase", "hydrant", "apple", "donut",
]))
# fmt: on

_CO3D_DATASET_ROOT: str = os.getenv("CO3D_DATASET_ROOT", "")

# _NEED_CONTROL is a list of those elements of JsonIndexDataset which
# are not directly specified for it in the config but come from the
# DatasetMapProvider.
_NEED_CONTROL: Tuple[str, ...] = (
    "dataset_root",
    "eval_batches",
    "eval_batch_index",
    "n_frames_per_sequence",
    "path_manager",
    "pick_sequence",
    "subsets",
    "frame_annotations_file",
    "sequence_annotations_file",
    "subset_lists_file",
)


@registry.register
class JsonIndexDatasetMapProvider(DatasetMapProviderBase):
    """
    Generates the training / validation and testing dataset objects for
    a dataset laid out on disk like Co3D, with annotations in json files.

    Args:
        category: The object category of the dataset.
        task_str: "multisequence" or "singlesequence".
        dataset_root: The root folder of the dataset.
        n_frames_per_sequence: Randomly sample #n_frames_per_sequence frames
            in each sequence.
        test_on_train: Construct validation and test datasets from
            the training subset.
        restrict_sequence_name: Restrict the dataset sequences to the ones
            present in the given list of names.
        test_restrict_sequence_id: The ID of the loaded sequence.
            Active for task_str='singlesequence'.
        assert_single_seq: Assert that only frames from a single sequence
            are present in all generated datasets.
        only_test_set: Load only the test set.
        dataset_class_type: name of class (JsonIndexDataset or a subclass)
                            to use for the dataset.
        dataset_X_args (e.g. dataset_JsonIndexDataset_args): arguments passed
            to all the dataset constructors.
        path_manager_factory: (Optional) An object that generates an instance of
            PathManager that can translate provided file paths.
        path_manager_factory_class_type: The class type of `path_manager_factory`.
    """

    # pyre-fixme[13]: Attribute `category` is never initialized.
    category: str
    task_str: str = "singlesequence"
    dataset_root: str = _CO3D_DATASET_ROOT
    n_frames_per_sequence: int = -1
    test_on_train: bool = False
    restrict_sequence_name: Tuple[str, ...] = ()
    test_restrict_sequence_id: int = -1
    assert_single_seq: bool = False
    only_test_set: bool = False
    # pyre-fixme[13]: Attribute `dataset` is never initialized.
    dataset: JsonIndexDataset
    dataset_class_type: str = "JsonIndexDataset"
    # pyre-fixme[13]: Attribute `path_manager_factory` is never initialized.
    path_manager_factory: PathManagerFactory
    path_manager_factory_class_type: str = "PathManagerFactory"

    @classmethod
    def dataset_tweak_args(cls, type, args: DictConfig) -> None:
        """
        Called by get_default_args(JsonIndexDatasetMapProvider) to
        not expose certain fields of each dataset class.
        """
        for key in _NEED_CONTROL:
            del args[key]

    def create_dataset(self):
        """
        Prevent the member named dataset from being created.
        """
        return

    def __post_init__(self):
        super().__init__()
        run_auto_creation(self)
        if self.only_test_set and self.test_on_train:
            raise ValueError("Cannot have only_test_set and test_on_train")

        path_manager = self.path_manager_factory.get()

        # TODO:
        # - implement loading multiple categories

        frame_file = os.path.join(
            self.dataset_root, self.category, "frame_annotations.jgz"
        )
        sequence_file = os.path.join(
            self.dataset_root, self.category, "sequence_annotations.jgz"
        )
        subset_lists_file = os.path.join(
            self.dataset_root, self.category, "set_lists.json"
        )
        common_kwargs = {
            "dataset_root": self.dataset_root,
            "path_manager": path_manager,
            "frame_annotations_file": frame_file,
            "sequence_annotations_file": sequence_file,
            "subset_lists_file": subset_lists_file,
            **getattr(self, f"dataset_{self.dataset_class_type}_args"),
        }

        # This maps the common names of the dataset subsets ("train"/"val"/"test")
        # to the names of the subsets in the CO3D dataset.
        set_names_mapping = _get_co3d_set_names_mapping(
            self.task_str,
            self.test_on_train,
            self.only_test_set,
        )

        # load the evaluation batches
        batch_indices_path = os.path.join(
            self.dataset_root,
            self.category,
            f"eval_batches_{self.task_str}.json",
        )
        if path_manager is not None:
            batch_indices_path = path_manager.get_local_path(batch_indices_path)
        if not os.path.isfile(batch_indices_path):
            # The batch indices file does not exist.
            # Most probably the user has not specified the root folder.
            raise ValueError(
                f"Looking for batch indices in {batch_indices_path}. "
                + "Please specify a correct dataset_root folder."
            )

        with open(batch_indices_path, "r") as f:
            eval_batch_index = json.load(f)
        restrict_sequence_name = self.restrict_sequence_name

        if self.task_str == "singlesequence":
            if (
                self.test_restrict_sequence_id is None
                or self.test_restrict_sequence_id < 0
            ):
                raise ValueError(
                    "Please specify an integer id 'test_restrict_sequence_id'"
                    + " of the sequence considered for 'singlesequence'"
                    + " training and evaluation."
                )
            if len(self.restrict_sequence_name) > 0:
                raise ValueError(
                    "For the 'singlesequence' task, the restrict_sequence_name has"
                    " to be unset while test_restrict_sequence_id has to be set to an"
                    " integer defining the order of the evaluation sequence."
                )
            # a sort-stable set() equivalent:
            eval_batches_sequence_names = list(
                {b[0][0]: None for b in eval_batch_index}.keys()
            )
            eval_sequence_name = eval_batches_sequence_names[
                self.test_restrict_sequence_id
            ]
            eval_batch_index = [
                b for b in eval_batch_index if b[0][0] == eval_sequence_name
            ]
            # overwrite the restrict_sequence_name
            restrict_sequence_name = [eval_sequence_name]
        if len(restrict_sequence_name) > 0:
            eval_batch_index = [
                b for b in eval_batch_index if b[0][0] in restrict_sequence_name
            ]

        dataset_type: Type[JsonIndexDataset] = registry.get(
            JsonIndexDataset, self.dataset_class_type
        )
        expand_args_fields(dataset_type)
        train_dataset = None
        if not self.only_test_set:
            train_dataset = dataset_type(
                n_frames_per_sequence=self.n_frames_per_sequence,
                subsets=set_names_mapping["train"],
                pick_sequence=restrict_sequence_name,
                **common_kwargs,
            )
        if self.test_on_train:
            assert train_dataset is not None
            val_dataset = test_dataset = train_dataset
        else:
            val_dataset = dataset_type(
                n_frames_per_sequence=-1,
                subsets=set_names_mapping["val"],
                pick_sequence=restrict_sequence_name,
                **common_kwargs,
            )
            test_dataset = dataset_type(
                n_frames_per_sequence=-1,
                subsets=set_names_mapping["test"],
                pick_sequence=restrict_sequence_name,
                eval_batch_index=eval_batch_index,
                **common_kwargs,
            )
        dataset_map = DatasetMap(
            train=train_dataset, val=val_dataset, test=test_dataset
        )

        if self.assert_single_seq:
            # check there's only one sequence in all datasets
            sequence_names = {
                sequence_name
                for dset in dataset_map.iter_datasets()
                for sequence_name in dset.sequence_names()
            }
            if len(sequence_names) > 1:
                raise ValueError("Multiple sequences loaded but expected one")

        self.dataset_map = dataset_map

    def get_dataset_map(self) -> DatasetMap:
        # pyre-ignore[16]
        return self.dataset_map

    def get_all_train_cameras(self) -> Optional[CamerasBase]:
        if self.task_str == "multisequence":
            return None

        assert self.task_str == "singlesequence"

        # pyre-ignore[16]
        train_dataset = self.dataset_map.train
        assert isinstance(train_dataset, JsonIndexDataset)
        return train_dataset.get_all_train_cameras()


def _get_co3d_set_names_mapping(
    task_str: str,
    test_on_train: bool,
    only_test: bool,
) -> Dict[str, List[str]]:
    """
    Returns the mapping of the common dataset subset names ("train"/"val"/"test")
    to the names of the corresponding subsets in the CO3D dataset
    ("test_known"/"test_unseen"/"train_known"/"train_unseen").

    The keys returned will be
        - train (if not only_test)
        - val (if not test_on_train)
        - test (if not test_on_train)
    """
    single_seq = task_str == "singlesequence"

    if only_test:
        set_names_mapping = {}
    else:
        set_names_mapping = {
            "train": [
                (DATASET_TYPE_TEST if single_seq else DATASET_TYPE_TRAIN)
                + "_"
                + DATASET_TYPE_KNOWN
            ]
        }
    if not test_on_train:
        prefixes = [DATASET_TYPE_TEST]
        if not single_seq:
            prefixes.append(DATASET_TYPE_TRAIN)
        set_names_mapping.update(
            {
                dset: [
                    p + "_" + t
                    for p in prefixes
                    for t in [DATASET_TYPE_KNOWN, DATASET_TYPE_UNKNOWN]
                ]
                for dset in ["val", "test"]
            }
        )

    return set_names_mapping
