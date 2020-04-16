# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch


if TYPE_CHECKING:
    from pytorch3d.structures import Pointclouds


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


def eyes(
    dim: int,
    N: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generates a batch of `N` identity matrices of shape `(N, dim, dim)`.

    Args:
        **dim**: The dimensionality of the identity matrices.
        **N**: The number of identity matrices.
        **device**: The device to be used for allocating the matrices.
        **dtype**: The datatype of the matrices.

    Returns:
        **identities**: A batch of identity matrices of shape `(N, dim, dim)`.
    """
    identities = torch.eye(dim, device=device, dtype=dtype)
    return identities[None].repeat(N, 1, 1)


def convert_pointclouds_to_tensor(pcl: Union[torch.Tensor, "Pointclouds"]):
    """
    If `type(pcl)==Pointclouds`, converts a `pcl` object to a
    padded representation and returns it together with the number of points
    per batch. Otherwise, returns the input itself with the number of points
    set to the size of the second dimension of `pcl`.
    """
    if is_pointclouds(pcl):
        X = pcl.points_padded()  # type: ignore
        num_points = pcl.num_points_per_cloud()  # type: ignore
    elif torch.is_tensor(pcl):
        X = pcl
        num_points = X.shape[1] * torch.ones(  # type: ignore
            X.shape[0], device=X.device, dtype=torch.int64
        )
    else:
        raise ValueError(
            "The inputs X, Y should be either Pointclouds objects or tensors."
        )
    return X, num_points


def is_pointclouds(pcl: Union[torch.Tensor, "Pointclouds"]):
    """ Checks whether the input `pcl` is an instance `Pointclouds` of
    by checking the existence of `points_padded` and `num_points_per_cloud`
    functions.
    """
    return hasattr(pcl, "points_padded") and hasattr(pcl, "num_points_per_cloud")
