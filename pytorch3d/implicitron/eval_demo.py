# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import copy
import dataclasses
import os
from typing import cast, Optional

import lpips
import torch
from pytorch3d.implicitron.dataset.dataloader_zoo import dataloader_zoo
from pytorch3d.implicitron.dataset.dataset_zoo import CO3D_CATEGORIES, dataset_zoo
from pytorch3d.implicitron.dataset.implicitron_dataset import (
    FrameData,
    ImplicitronDataset,
    ImplicitronDatasetBase,
)
from pytorch3d.implicitron.dataset.utils import is_known_frame
from pytorch3d.implicitron.evaluation.evaluate_new_view_synthesis import (
    aggregate_nvs_results,
    eval_batch,
    pretty_print_nvs_metrics,
    summarize_nvs_eval_results,
)
from pytorch3d.implicitron.models.model_dbir import ModelDBIR
from pytorch3d.implicitron.tools.utils import dataclass_to_cuda_
from tqdm import tqdm


def main() -> None:
    """
    Evaluates new view synthesis metrics of a simple depth-based image rendering
    (DBIR) model for multisequence/singlesequence tasks for several categories.

    The evaluation is conducted on the same data as in [1] and, hence, the results
    are directly comparable to the numbers reported in [1].

    References:
        [1] J. Reizenstein, R. Shapovalov, P. Henzler, L. Sbordone,
                P. Labatut, D. Novotny:
            Common Objects in 3D: Large-Scale Learning
                and Evaluation of Real-life 3D Category Reconstruction
    """

    task_results = {}
    for task in ("singlesequence", "multisequence"):
        task_results[task] = []
        for category in CO3D_CATEGORIES[: (20 if task == "singlesequence" else 10)]:
            for single_sequence_id in (0, 1) if task == "singlesequence" else (None,):
                category_result = evaluate_dbir_for_category(
                    category, task=task, single_sequence_id=single_sequence_id
                )
                print("")
                print(
                    f"Results for task={task}; category={category};"
                    + (
                        f" sequence={single_sequence_id}:"
                        if single_sequence_id is not None
                        else ":"
                    )
                )
                pretty_print_nvs_metrics(category_result)
                print("")

                task_results[task].append(category_result)
            _print_aggregate_results(task, task_results)

    for task in task_results:
        _print_aggregate_results(task, task_results)


def evaluate_dbir_for_category(
    category: str = "apple",
    bg_color: float = 0.0,
    task: str = "singlesequence",
    single_sequence_id: Optional[int] = None,
    num_workers: int = 16,
):
    """
    Evaluates new view synthesis metrics of a simple depth-based image rendering
    (DBIR) model for a given task, category, and sequence (in case task=='singlesequence').

    Args:
        category: Object category.
        bg_color: Background color of the renders.
        task: Evaluation task. Either singlesequence or multisequence.
        single_sequence_id: The ID of the evaluiation sequence for the singlesequence task.
        num_workers: The number of workers for the employed dataloaders.

    Returns:
        category_result: A dictionary of quantitative metrics.
    """

    single_sequence_id = single_sequence_id if single_sequence_id is not None else -1

    torch.manual_seed(42)

    if task not in ["multisequence", "singlesequence"]:
        raise ValueError("'task' has to be either 'multisequence' or 'singlesequence'")

    datasets = dataset_zoo(
        category=category,
        dataset_root=os.environ["CO3D_DATASET_ROOT"],
        assert_single_seq=task == "singlesequence",
        dataset_name=f"co3d_{task}",
        test_on_train=False,
        load_point_clouds=True,
        test_restrict_sequence_id=single_sequence_id,
    )

    dataloaders = dataloader_zoo(
        datasets,
        dataset_name=f"co3d_{task}",
    )

    test_dataset = datasets["test"]
    test_dataloader = dataloaders["test"]

    if task == "singlesequence":
        # all_source_cameras are needed for evaluation of the
        # target camera difficulty
        # pyre-fixme[16]: `ImplicitronDataset` has no attribute `frame_annots`.
        sequence_name = test_dataset.frame_annots[0]["frame_annotation"].sequence_name
        all_source_cameras = _get_all_source_cameras(
            test_dataset, sequence_name, num_workers=num_workers
        )
    else:
        all_source_cameras = None

    image_size = cast(ImplicitronDataset, test_dataset).image_width

    if image_size is None:
        raise ValueError("Image size should be set in the dataset")

    # init the simple DBIR model
    model = ModelDBIR(
        image_size=image_size,
        bg_color=bg_color,
        max_points=int(1e5),
    )
    model.cuda()

    # init the lpips model for eval
    lpips_model = lpips.LPIPS(net="vgg")
    lpips_model = lpips_model.cuda()

    per_batch_eval_results = []
    print("Evaluating DBIR model ...")
    for frame_data in tqdm(test_dataloader):
        frame_data = dataclass_to_cuda_(frame_data)
        preds = model(**dataclasses.asdict(frame_data))
        nvs_prediction = copy.deepcopy(preds["nvs_prediction"])
        per_batch_eval_results.append(
            eval_batch(
                frame_data,
                nvs_prediction,
                bg_color=bg_color,
                lpips_model=lpips_model,
                source_cameras=all_source_cameras,
            )
        )

    category_result_flat, category_result = summarize_nvs_eval_results(
        per_batch_eval_results, task
    )

    return category_result["results"]


def _print_aggregate_results(task, task_results) -> None:
    """
    Prints the aggregate metrics for a given task.
    """
    aggregate_task_result = aggregate_nvs_results(task_results[task])
    print("")
    print(f"Aggregate results for task={task}:")
    pretty_print_nvs_metrics(aggregate_task_result)
    print("")


def _get_all_source_cameras(
    dataset: ImplicitronDatasetBase, sequence_name: str, num_workers: int = 8
):
    """
    Loads all training cameras of a given sequence.

    The set of all seen cameras is needed for evaluating the viewpoint difficulty
    for the singlescene evaluation.

    Args:
        dataset: Co3D dataset object.
        sequence_name: The name of the sequence.
        num_workers: The number of for the utilized dataloader.
    """

    # load all source cameras of the sequence
    seq_idx = list(dataset.sequence_indices_in_order(sequence_name))
    dataset_for_loader = torch.utils.data.Subset(dataset, seq_idx)
    (all_frame_data,) = torch.utils.data.DataLoader(
        dataset_for_loader,
        shuffle=False,
        batch_size=len(dataset_for_loader),
        num_workers=num_workers,
        collate_fn=FrameData.collate,
    )
    is_known = is_known_frame(all_frame_data.frame_type)
    source_cameras = all_frame_data.camera[torch.where(is_known)[0]]
    return source_cameras


if __name__ == "__main__":
    main()
