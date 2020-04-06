# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
from typing import Optional, Tuple, Union

import torch


def wmean(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    dim: Union[int, Tuple[int]] = -2,
    keepdim: bool = True,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Finds the mean of the input tensor across the specified dimension.
    If the `weight` argument is provided, computes weighted mean.
    Args:
        x: tensor of shape `(*, D)`, where D is assumed to be spatial;
        weights: if given, non-negative tensor of shape `(*,)`. It must be
            broadcastable to `x.shape[:-1]`. Note that the weights for
            the last (spatial) dimension are assumed same;
        dim: dimension(s) in `x` to average over;
        keepdim: tells whether to keep the resulting singleton dimension.
        eps: minumum clamping value in the denominator.
    Returns:
        the mean tensor:
        * if `weights` is None => `mean(x, dim)`,
        * otherwise => `sum(x*w, dim) / max{sum(w, dim), eps}`.
    """
    args = {"dim": dim, "keepdim": keepdim}

    if weight is None:
        return x.mean(**args)

    if any(
        xd != wd and xd != 1 and wd != 1
        for xd, wd in zip(x.shape[-2::-1], weight.shape[::-1])
    ):
        raise ValueError("wmean: weights are not compatible with the tensor")

    return (x * weight[..., None]).sum(**args) / weight[..., None].sum(**args).clamp(
        eps
    )
