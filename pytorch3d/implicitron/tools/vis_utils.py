# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from visdom import Visdom


logger = logging.getLogger(__name__)


def get_visdom_env(visdom_env: str, exp_dir: str) -> str:
    """
    Parse out visdom environment name from the input config.

    Args:
        visdom_env: Name of the wisdom environment, could be empty string.
        exp_dir: Root experiment directory.

    Returns:
        visdom_env: The name of the visdom environment. If the given visdom_env is
            empty, return the name of the bottom directory in exp_dir.
    """
    if len(visdom_env) == 0:
        visdom_env = exp_dir.split("/")[-1]
    else:
        visdom_env = visdom_env
    return visdom_env


# TODO: a proper singleton
_viz_singleton = None


def get_visdom_connection(
    server: str = "http://localhost",
    port: int = 8097,
) -> Optional["Visdom"]:
    """
    Obtain a connection to a visdom server if visdom is installed.

    Args:
        server: Server address.
        port: Server port.

    Returns:
        connection: The connection object.
    """
    try:
        from visdom import Visdom
    except ImportError:
        logger.debug("Cannot load visdom")
        return None

    if server == "None":
        return None

    global _viz_singleton
    if _viz_singleton is None:
        _viz_singleton = Visdom(server=server, port=port)
    return _viz_singleton


def visualize_basics(
    viz: "Visdom",
    preds: Dict[str, Any],
    visdom_env_imgs: str,
    title: str = "",
    visualize_preds_keys: Tuple[str, ...] = (
        "image_rgb",
        "images_render",
        "fg_probability",
        "masks_render",
        "depths_render",
        "depth_map",
    ),
    store_history: bool = False,
) -> None:
    """
    Visualize basic outputs of a `GenericModel` to visdom.

    Args:
        viz: The visdom object.
        preds: A dictionary containing `GenericModel` outputs.
        visdom_env_imgs: Target visdom environment name.
        title: The title of produced visdom window.
        visualize_preds_keys: The list of keys of `preds` for visualization.
        store_history: Store the history buffer in visdom windows.
    """
    imout = {}
    for k in visualize_preds_keys:
        if k not in preds or preds[k] is None:
            logger.info(f"cant show {k}")
            continue
        v = preds[k].cpu().detach().clone()
        if k.startswith("depth"):
            # divide by 95th percentile
            normfac = (
                v.view(v.shape[0], -1)
                .topk(k=int(0.05 * (v.numel() // v.shape[0])), dim=-1)
                .values[:, -1]
            )
            v = v / normfac[:, None, None, None].clamp(1e-4)
        if v.shape[1] == 1:
            v = v.repeat(1, 3, 1, 1)
        v = torch.nn.functional.interpolate(
            v,
            scale_factor=(
                600.0
                if (
                    "_eval" in visdom_env_imgs
                    and k in ("images_render", "depths_render")
                )
                else 200.0
            )
            / v.shape[2],
            mode="bilinear",
        )
        imout[k] = v

    # TODO: handle errors on the outside
    try:
        imout = {"all": torch.cat(list(imout.values()), dim=2)}
    except RuntimeError as e:
        print("cant cat!", e.args)

    for k, v in imout.items():
        viz.images(
            v.clamp(0.0, 1.0),
            win=k,
            env=visdom_env_imgs,
            opts={"title": title + "_" + k, "store_history": store_history},
        )


def make_depth_image(
    depths: torch.Tensor,
    masks: torch.Tensor,
    max_quantile: float = 0.98,
    min_quantile: float = 0.02,
    min_out_depth: float = 0.1,
    max_out_depth: float = 0.9,
) -> torch.Tensor:
    """
    Convert a batch of depth maps to a grayscale image.

    Args:
        depths: A tensor of shape `(B, 1, H, W)` containing a batch of depth maps.
        masks: A tensor of shape `(B, 1, H, W)` containing a batch of foreground masks.
        max_quantile: The quantile of the input depth values which will
            be mapped to `max_out_depth`.
        min_quantile: The quantile of the input depth values which will
            be mapped to `min_out_depth`.
        min_out_depth: The minimal value in each depth map will be assigned this color.
        max_out_depth: The maximal value in each depth map will be assigned this color.

    Returns:
        depth_image: A tensor of shape `(B, 1, H, W)` a batch of grayscale
            depth images.
    """
    normfacs = []
    for d, m in zip(depths, masks):
        ok = (d.view(-1) > 1e-6) * (m.view(-1) > 0.5)
        if ok.sum() <= 1:
            logger.info("empty depth!")
            normfacs.append(torch.zeros(2).type_as(depths))
            continue
        dok = d.view(-1)[ok].view(-1)
        _maxk = max(int(round((1 - max_quantile) * (dok.numel()))), 1)
        _mink = max(int(round(min_quantile * (dok.numel()))), 1)
        normfac_max = dok.topk(k=_maxk, dim=-1).values[-1]
        normfac_min = dok.topk(k=_mink, dim=-1, largest=False).values[-1]
        normfacs.append(torch.stack([normfac_min, normfac_max]))
    normfacs = torch.stack(normfacs)
    _min, _max = (normfacs[:, 0].view(-1, 1, 1, 1), normfacs[:, 1].view(-1, 1, 1, 1))
    depths = (depths - _min) / (_max - _min).clamp(1e-4)
    depths = (
        (depths * (max_out_depth - min_out_depth) + min_out_depth) * masks.float()
    ).clamp(0.0, 1.0)
    return depths
