# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import copy
import json
import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch

import tqdm
from pytorch3d.implicitron.evaluation import evaluate_new_view_synthesis as evaluate
from pytorch3d.implicitron.models.base_model import EvaluationMode, ImplicitronModelBase
from pytorch3d.implicitron.tools.config import (
    registry,
    ReplaceableBase,
    run_auto_creation,
)
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class EvaluatorBase(ReplaceableBase):
    """
    Evaluate a trained model on given data. Returns a dict of loss/objective
    names and their values.
    """

    is_multisequence: bool = False

    def run(
        self, model: ImplicitronModelBase, dataloader: DataLoader, **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate the results of Implicitron training.
        """
        raise NotImplementedError()


@registry.register
class ImplicitronEvaluator(EvaluatorBase):
    """
    Evaluate the results of Implicitron training.
    """

    # UNUSED; preserved for compatibility purposes
    camera_difficulty_bin_breaks: Tuple[float, ...] = 0.97, 0.98

    def __post_init__(self):
        run_auto_creation(self)

    # pyre-fixme[14]: `run` overrides method defined in `EvaluatorBase` inconsistently.
    def run(
        self,
        model: ImplicitronModelBase,
        dataloader: DataLoader,
        device: torch.device,
        dump_to_json: bool = False,
        exp_dir: Optional[str] = None,
        epoch: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Evaluate the results of Implicitron training. Optionally, dump results to
        exp_dir/results_test.json.

        Args:
            model: A (trained) model to evaluate.
            dataloader: A test dataloader.
            device: A torch device.
            dump_to_json: If True, will dump the results to a json file.
            exp_dir: Root expeirment directory.
            epoch: Evaluation epoch (to be stored in the results dict).

        Returns:
            A dictionary of results.
        """
        try:
            import lpips

            lpips_model = lpips.LPIPS(net="vgg")
            lpips_model = lpips_model.to(device)
        except ImportError:
            warnings.warn(
                "lpips library NOT FOUND. lpips losses will not be calculated"
            )
            lpips_model = None

        model.eval()

        per_batch_eval_results = []
        logger.info("Evaluating model ...")
        for frame_data in tqdm.tqdm(dataloader):
            frame_data = frame_data.to(device)

            # mask out the unknown images so that the model does not see them
            frame_data_for_eval = _get_eval_frame_data(frame_data)

            with torch.no_grad():
                preds = model(
                    **{
                        **frame_data_for_eval,
                        "evaluation_mode": EvaluationMode.EVALUATION,
                    }
                )
                implicitron_render = copy.deepcopy(preds["implicitron_render"])
                per_batch_eval_results.append(
                    evaluate.eval_batch(
                        frame_data,
                        implicitron_render,
                        bg_color="black",
                        lpips_model=lpips_model,
                    )
                )

        _, category_result = evaluate.summarize_nvs_eval_results(
            per_batch_eval_results,
            self.is_multisequence,
        )

        results = category_result["results"]
        evaluate.pretty_print_nvs_metrics(results)
        if dump_to_json:
            _dump_to_json(epoch, exp_dir, results)

        return category_result["results"]


def _dump_to_json(
    epoch: Optional[int], exp_dir: Optional[str], results: List[Dict[str, Any]]
) -> None:
    if epoch is not None:
        for r in results:
            r["eval_epoch"] = int(epoch)
    logger.info("Evaluation results")

    if exp_dir is None:
        raise ValueError("Cannot save results to json without a specified save path.")
    with open(os.path.join(exp_dir, "results_test.json"), "w") as f:
        json.dump(results, f)


def _get_eval_frame_data(frame_data: Any) -> Any:
    """
    Masks the target image data to make sure we cannot use it at model evaluation
    time. Assumes the first batch element is target, the rest are source.
    """
    frame_data_for_eval = copy.deepcopy(frame_data)
    for k in ("image_rgb", "depth_map", "fg_probability", "mask_crop"):
        value = getattr(frame_data_for_eval, k)
        value[0].zero_()
    return frame_data_for_eval
