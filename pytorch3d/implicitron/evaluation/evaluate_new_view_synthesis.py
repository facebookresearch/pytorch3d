# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import copy
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.implicitron.dataset.frame_data import FrameData
from pytorch3d.implicitron.dataset.utils import is_train_frame
from pytorch3d.implicitron.models.base_model import ImplicitronRender
from pytorch3d.implicitron.tools import vis_utils
from pytorch3d.implicitron.tools.image_utils import mask_background
from pytorch3d.implicitron.tools.metric_utils import calc_psnr, eval_depth, iou, rgb_l1
from pytorch3d.implicitron.tools.point_cloud_utils import get_rgbd_point_cloud
from pytorch3d.implicitron.tools.vis_utils import make_depth_image
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.vis.plotly_vis import plot_scene
from tabulate import tabulate

if TYPE_CHECKING:
    from visdom import Visdom


EVAL_N_SRC_VIEWS = [1, 3, 5, 7, 9]


@dataclass
class _Visualizer:
    image_render: torch.Tensor
    image_rgb_masked: torch.Tensor
    depth_render: torch.Tensor
    depth_map: Optional[torch.Tensor]
    depth_mask: Optional[torch.Tensor]

    visdom_env: str = "eval_debug"

    _viz: Optional["Visdom"] = field(init=False)

    def __post_init__(self):
        self._viz = vis_utils.get_visdom_connection()

    def show_rgb(
        self, loss_value: float, metric_name: str, loss_mask_now: torch.Tensor
    ):
        if self._viz is None:
            return
        self._viz.images(
            torch.cat(
                (
                    self.image_render,
                    self.image_rgb_masked,
                    loss_mask_now.repeat(1, 3, 1, 1),
                ),
                dim=3,
            ),
            env=self.visdom_env,
            win=metric_name,
            opts={"title": f"{metric_name}_{loss_value:1.2f}"},
        )

    def show_depth(
        self, depth_loss: float, name_postfix: str, loss_mask_now: torch.Tensor
    ):
        if self._viz is None:
            return
        viz = self._viz
        viz.images(
            torch.cat(
                (make_depth_image(self.depth_render, loss_mask_now),)
                + (
                    (make_depth_image(self.depth_map, loss_mask_now),)
                    if self.depth_map is not None
                    else ()
                ),
                dim=3,
            ),
            env=self.visdom_env,
            win="depth_abs" + name_postfix,
            opts={"title": f"depth_abs_{name_postfix}_{depth_loss:1.2f}"},
        )
        viz.images(
            loss_mask_now,
            env=self.visdom_env,
            win="depth_abs" + name_postfix + "_mask",
            opts={"title": f"depth_abs_{name_postfix}_{depth_loss:1.2f}_mask"},
        )
        if self.depth_mask is not None:
            viz.images(
                self.depth_mask,
                env=self.visdom_env,
                win="depth_abs" + name_postfix + "_maskd",
                opts={"title": f"depth_abs_{name_postfix}_{depth_loss:1.2f}_maskd"},
            )

        # show the 3D plot
        # pyre-fixme[9]: viewpoint_trivial has type `PerspectiveCameras`; used as
        #  `TensorProperties`.
        viewpoint_trivial: PerspectiveCameras = PerspectiveCameras().to(
            loss_mask_now.device
        )
        _pcls = {
            "pred_depth": get_rgbd_point_cloud(
                viewpoint_trivial,
                self.image_render,
                self.depth_render,
                # mask_crop,
                torch.ones_like(self.depth_render),
                # loss_mask_now,
            )
        }
        if self.depth_map is not None:
            _pcls["gt_depth"] = get_rgbd_point_cloud(
                viewpoint_trivial,
                self.image_rgb_masked,
                self.depth_map,
                # mask_crop,
                torch.ones_like(self.depth_map),
                # loss_mask_now,
            )

        _pcls = {pn: p for pn, p in _pcls.items() if int(p.num_points_per_cloud()) > 0}

        plotlyplot = plot_scene(
            {f"pcl{name_postfix}": _pcls},  # pyre-ignore
            camera_scale=1.0,
            pointcloud_max_points=10000,
            pointcloud_marker_size=1,
        )
        viz.plotlyplot(
            plotlyplot,
            env=self.visdom_env,
            win=f"pcl{name_postfix}",
        )


def eval_batch(
    frame_data: FrameData,
    implicitron_render: ImplicitronRender,
    bg_color: Union[torch.Tensor, Sequence, str, float] = "black",
    mask_thr: float = 0.5,
    lpips_model=None,
    visualize: bool = False,
    visualize_visdom_env: str = "eval_debug",
    break_after_visualising: bool = True,
) -> Dict[str, Any]:
    """
    Produce performance metrics for a single batch of new-view synthesis
    predictions.

    Given a set of known views (for which frame_data.frame_type.endswith('known')
    is True), a new-view synthesis method (NVS) is tasked to generate new views
    of the scene from the viewpoint of the target views (for which
    frame_data.frame_type.endswith('known') is False). The resulting
    synthesized new views, stored in `implicitron_render`, are compared to the
    target ground truth in `frame_data` in terms of geometry and appearance
    resulting in a dictionary of metrics returned by the `eval_batch` function.

    Args:
        frame_data: A FrameData object containing the input to the new view
            synthesis method.
        implicitron_render: The data describing the synthesized new views.
        bg_color: The background color of the generated new views and the
            ground truth.
        lpips_model: A pre-trained model for evaluating the LPIPS metric.
        visualize: If True, visualizes the results to Visdom.

    Returns:
        results: A dictionary holding evaluation metrics.

    Throws:
        ValueError if frame_data does not have frame_type, camera, or image_rgb
        ValueError if the batch has a mix of training and test samples
        ValueError if the batch frames are not [unseen, known, known, ...]
        ValueError if one of the required fields in implicitron_render is missing
    """
    frame_type = frame_data.frame_type
    if frame_type is None:
        raise ValueError("Frame type has not been set.")

    # we check that all those fields are not None but Pyre can't infer that properly
    # TODO: assign to local variables and simplify the code.
    if frame_data.image_rgb is None:
        raise ValueError("Image is not in the evaluation batch.")

    if frame_data.camera is None:
        raise ValueError("Camera is not in the evaluation batch.")

    # eval all results in the resolution of the frame_data image
    image_resol = tuple(frame_data.image_rgb.shape[2:])

    # Post-process the render:
    # 1) check implicitron_render for Nones,
    # 2) obtain copies to make sure we dont edit the original data,
    # 3) take only the 1st (target) image
    # 4) resize to match ground-truth resolution
    cloned_render: Dict[str, torch.Tensor] = {}
    for k in ["mask_render", "image_render", "depth_render"]:
        field = getattr(implicitron_render, k)
        if field is None:
            raise ValueError(f"A required predicted field {k} is missing")

        imode = "bilinear" if k == "image_render" else "nearest"
        cloned_render[k] = (
            F.interpolate(field[:1], size=image_resol, mode=imode).detach().clone()
        )

    frame_data = copy.deepcopy(frame_data)

    # mask the ground truth depth in case frame_data contains the depth mask
    if frame_data.depth_map is not None and frame_data.depth_mask is not None:
        frame_data.depth_map *= frame_data.depth_mask

    if not isinstance(frame_type, list):  # not batch FrameData
        frame_type = [frame_type]

    is_train = is_train_frame(frame_type)
    if len(is_train) > 1 and (is_train[1] != is_train[1:]).any():
        raise ValueError(
            "All (conditioning) frames in the eval batch have to be either train/test."
        )

    for k in [
        "depth_map",
        "image_rgb",
        "fg_probability",
        "mask_crop",
    ]:
        if not hasattr(frame_data, k) or getattr(frame_data, k) is None:
            continue
        setattr(frame_data, k, getattr(frame_data, k)[:1])

    if frame_data.depth_map is None or frame_data.depth_map.sum() <= 0:
        warnings.warn("Empty or missing depth map in evaluation!")

    if frame_data.mask_crop is None:
        warnings.warn("mask_crop is None, assuming the whole image is valid.")

    if frame_data.fg_probability is None:
        warnings.warn("fg_probability is None, assuming the whole image is fg.")

    # threshold the masks to make ground truth binary masks
    mask_fg = (
        frame_data.fg_probability >= mask_thr
        if frame_data.fg_probability is not None
        # pyre-ignore [16]
        else torch.ones_like(frame_data.image_rgb[:, :1, ...]).bool()
    )

    mask_crop = (
        frame_data.mask_crop
        if frame_data.mask_crop is not None
        else torch.ones_like(mask_fg)
    )

    # unmasked g.t. image
    image_rgb = frame_data.image_rgb

    # fg-masked g.t. image
    image_rgb_masked = mask_background(
        # pyre-fixme[6]: Expected `Tensor` for 1st param but got
        #  `Optional[torch.Tensor]`.
        frame_data.image_rgb,
        mask_fg,
        bg_color=bg_color,
    )

    # clamp predicted images
    image_render = cloned_render["image_render"].clamp(0.0, 1.0)

    if visualize:
        visualizer = _Visualizer(
            image_render=image_render,
            image_rgb_masked=image_rgb_masked,
            depth_render=cloned_render["depth_render"],
            depth_map=frame_data.depth_map,
            depth_mask=(
                frame_data.depth_mask[:1] if frame_data.depth_mask is not None else None
            ),
            visdom_env=visualize_visdom_env,
        )

    results: Dict[str, Any] = {}

    results["iou"] = iou(
        cloned_render["mask_render"],
        mask_fg,
        mask=mask_crop,
    )

    for loss_fg_mask, name_postfix in zip((mask_crop, mask_fg), ("_masked", "_fg")):
        loss_mask_now = mask_crop * loss_fg_mask

        for rgb_metric_name, rgb_metric_fun in zip(
            ("psnr", "rgb_l1"), (calc_psnr, rgb_l1)
        ):
            metric_name = rgb_metric_name + name_postfix
            results[metric_name] = rgb_metric_fun(
                image_render,
                image_rgb_masked,
                mask=loss_mask_now,
            )

            if visualize:
                visualizer.show_rgb(
                    results[metric_name].item(), metric_name, loss_mask_now
                )

        if name_postfix == "_fg" and frame_data.depth_map is not None:
            # only record depth metrics for the foreground
            _, abs_ = eval_depth(
                cloned_render["depth_render"],
                # pyre-fixme[6]: For 2nd param expected `Tensor` but got
                #  `Optional[Tensor]`.
                frame_data.depth_map,
                get_best_scale=True,
                mask=loss_mask_now,
                crop=5,
            )
            results["depth_abs" + name_postfix] = abs_.mean()

            if visualize:
                visualizer.show_depth(abs_.mean().item(), name_postfix, loss_mask_now)
                if break_after_visualising:
                    breakpoint()  # noqa: B601

    # add the rgb metrics between the render and the unmasked image
    for rgb_metric_name, rgb_metric_fun in zip(
        ("psnr_full_image", "rgb_l1_full_image"), (calc_psnr, rgb_l1)
    ):
        results[rgb_metric_name] = rgb_metric_fun(
            image_render,
            # pyre-fixme[6]: For 2nd argument expected `Tensor` but got
            #  `Optional[Tensor]`.
            image_rgb,
            mask=mask_crop,
        )

    if lpips_model is not None:
        for gt_image_type in ("_full_image", "_masked"):
            im1, im2 = [
                2.0 * im.clamp(0.0, 1.0) - 1.0  # pyre-ignore[16]
                for im in (
                    image_rgb_masked if gt_image_type == "_masked" else image_rgb,
                    cloned_render["image_render"],
                )
            ]
            results["lpips" + gt_image_type] = lpips_model.forward(im1, im2).item()

    # convert all metrics to floats
    results = {k: float(v) for k, v in results.items()}

    results["meta"] = {
        # store the size of the batch (corresponds to n_src_views+1)
        "batch_size": len(frame_type),
        # store the type of the target frame
        # pyre-fixme[16]: `None` has no attribute `__getitem__`.
        "frame_type": str(frame_data.frame_type[0]),
    }

    return results


def average_per_batch_results(
    results_per_batch: List[Dict[str, Any]],
    idx: Optional[torch.Tensor] = None,
) -> dict:
    """
    Average a list of per-batch metrics `results_per_batch`.
    Optionally, if `idx` is given, only a subset of the per-batch
    metrics, indexed by `idx`, is averaged.
    """
    result_keys = list(results_per_batch[0].keys())
    result_keys.remove("meta")
    if idx is not None:
        results_per_batch = [results_per_batch[i] for i in idx]
    if len(results_per_batch) == 0:
        return {k: float("NaN") for k in result_keys}
    return {
        k: float(np.array([r[k] for r in results_per_batch]).mean())
        for k in result_keys
    }


def _reduce_camera_iou_overlap(ious: torch.Tensor, topk: int = 2) -> torch.Tensor:
    """
    Calculate the final camera difficulty by computing the average of the
    ious of the two most similar cameras.

    Returns:
        single-element Tensor
    """
    return ious.topk(k=min(topk, len(ious) - 1)).values.mean()


def _get_camera_difficulty_bin_edges(camera_difficulty_bin_breaks: Tuple[float, float]):
    """
    Get the edges of camera difficulty bins.
    """
    _eps = 1e-5
    lower, upper = camera_difficulty_bin_breaks
    diff_bin_edges = torch.tensor([0.0 - _eps, lower, upper, 1.0 + _eps]).float()
    diff_bin_names = ["hard", "medium", "easy"]
    return diff_bin_edges, diff_bin_names


def summarize_nvs_eval_results(
    per_batch_eval_results: List[Dict[str, Any]],
    is_multisequence: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Compile the per-batch evaluation results `per_batch_eval_results` into
    a set of aggregate metrics. The produced metrics depend on is_multisequence.

    Args:
        per_batch_eval_results: Metrics of each per-batch evaluation.
        is_multisequence: Whether to evaluate as a multisequence task
        camera_difficulty_bin_breaks: edge hard-medium and medium-easy


    Returns:
        nvs_results_flat: A flattened dict of all aggregate metrics.
        aux_out: A dictionary holding a set of auxiliary results.
    """
    n_batches = len(per_batch_eval_results)
    eval_sets: List[Optional[str]] = []
    eval_sets = [None]
    if is_multisequence:
        eval_sets = ["train", "test"]
    batch_sizes = torch.tensor(
        [r["meta"]["batch_size"] for r in per_batch_eval_results]
    ).long()

    is_train = is_train_frame([r["meta"]["frame_type"] for r in per_batch_eval_results])

    # init the result database dict
    results = []

    # add per set averages
    for SET in eval_sets:
        if SET is None:
            ok_set = torch.ones(n_batches, dtype=torch.bool)
            set_name = "test"
        else:
            ok_set = is_train == int(SET == "train")
            set_name = SET

        # average over all results
        bin_results = average_per_batch_results(
            per_batch_eval_results, idx=torch.where(ok_set)[0]
        )
        results.append(
            {
                "subset": set_name,
                "subsubset": "diff=all",
                "metrics": bin_results,
            }
        )

        if is_multisequence:
            # split based on n_src_views
            n_src_views = batch_sizes - 1
            for n_src in EVAL_N_SRC_VIEWS:
                ok_src = ok_set & (n_src_views == n_src)
                n_src_results = average_per_batch_results(
                    per_batch_eval_results,
                    idx=torch.where(ok_src)[0],
                )
                results.append(
                    {
                        "subset": set_name,
                        "subsubset": f"n_src={int(n_src)}",
                        "metrics": n_src_results,
                    }
                )

    aux_out = {"results": results}
    return flatten_nvs_results(results), aux_out


def _get_flat_nvs_metric_key(result, metric_name) -> str:
    metric_key_postfix = f"|subset={result['subset']}|{result['subsubset']}"
    metric_key = f"{metric_name}{metric_key_postfix}"
    return metric_key


def flatten_nvs_results(results) -> Dict[str, Any]:
    """
    Takes input `results` list of dicts of the form::

        [
            {
                'subset':'train/test/...',
                'subsubset': 'src=1/src=2/...',
                'metrics': nvs_eval_metrics}
            },
            ...
        ]

    And converts to a flat dict as follows::

        {
            'subset=train/test/...|subsubset=src=1/src=2/...': nvs_eval_metrics,
            ...
        }
    """
    results_flat = {}
    for result in results:
        for metric_name, metric_val in result["metrics"].items():
            metric_key = _get_flat_nvs_metric_key(result, metric_name)
            assert metric_key not in results_flat
            results_flat[metric_key] = metric_val
    return results_flat


def pretty_print_nvs_metrics(results) -> None:
    subsets, subsubsets = [
        _ordered_set([r[k] for r in results]) for k in ("subset", "subsubset")
    ]
    metrics = _ordered_set([metric for r in results for metric in r["metrics"]])

    for subset in subsets:
        tab = {}
        for metric in metrics:
            tab[metric] = []
            header = ["metric"]
            for subsubset in subsubsets:
                metric_vals = [
                    r["metrics"][metric]
                    for r in results
                    if r["subsubset"] == subsubset and r["subset"] == subset
                ]
                if len(metric_vals) > 0:
                    tab[metric].extend(metric_vals)
                    header.extend(subsubsets)

        if any(len(v) > 0 for v in tab.values()):
            print(f"===== NVS results; subset={subset} =====")
            print(
                tabulate(
                    [[metric, *v] for metric, v in tab.items()],
                    # pyre-fixme[61]: `header` is undefined, or not always defined.
                    headers=header,
                )
            )


def _ordered_set(list_):
    return list(OrderedDict((i, 0) for i in list_).keys())


def aggregate_nvs_results(task_results):
    """
    Aggregate nvs results.
    For singlescene, this averages over all categories and scenes,
    for multiscene, the average is over all per-category results.
    """
    task_results_cat = [r_ for r in task_results for r_ in r]
    subsets, subsubsets = [
        _ordered_set([r[k] for r in task_results_cat]) for k in ("subset", "subsubset")
    ]
    metrics = _ordered_set(
        [metric for r in task_results_cat for metric in r["metrics"]]
    )
    average_results = []
    for subset in subsets:
        for subsubset in subsubsets:
            metrics_lists = [
                r["metrics"]
                for r in task_results_cat
                if r["subsubset"] == subsubset and r["subset"] == subset
            ]
            avg_metrics = {}
            for metric in metrics:
                avg_metrics[metric] = float(
                    np.nanmean(
                        np.array([metric_list[metric] for metric_list in metrics_lists])
                    )
                )
            average_results.append(
                {
                    "subset": subset,
                    "subsubset": subsubset,
                    "metrics": avg_metrics,
                }
            )
    return average_results
