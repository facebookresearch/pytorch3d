# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Tuple, Union

import torch


def acos_linear_extrapolation(
    x: torch.Tensor,
    bound: Union[float, Tuple[float, float]] = 1.0 - 1e-4,
) -> torch.Tensor:
    """
    Implements `arccos(x)` which is linearly extrapolated outside `x`'s original
    domain of `(-1, 1)`. This allows for stable backpropagation in case `x`
    is not guaranteed to be strictly within `(-1, 1)`.

    More specifically:
    ```
    if -bound <= x <= bound:
        acos_linear_extrapolation(x) = acos(x)
    elif x <= -bound: # 1st order Taylor approximation
        acos_linear_extrapolation(x) = acos(-bound) + dacos/dx(-bound) * (x - (-bound))
    else:  # x >= bound
        acos_linear_extrapolation(x) = acos(bound) + dacos/dx(bound) * (x - bound)
    ```
    Note that `bound` can be made more specific with setting
    `bound=[lower_bound, upper_bound]` as detailed below.

    Args:
        x: Input `Tensor`.
        bound: A float constant or a float 2-tuple defining the region for the
            linear extrapolation of `acos`.
            If `bound` is a float scalar, linearly interpolates acos for
            `x <= -bound` or `bound <= x`.
            If `bound` is a 2-tuple, the first/second element of `bound`
            describes the lower/upper bound that defines the lower/upper
            extrapolation region, i.e. the region where
            `x <= bound[0]`/`bound[1] <= x`.
            Note that all elements of `bound` have to be within (-1, 1).
    Returns:
        acos_linear_extrapolation: `Tensor` containing the extrapolated `arccos(x)`.
    """

    if isinstance(bound, float):
        upper_bound = bound
        lower_bound = -bound
    else:
        lower_bound, upper_bound = bound

    if lower_bound > upper_bound:
        raise ValueError("lower bound has to be smaller or equal to upper bound.")

    if lower_bound <= -1.0 or upper_bound >= 1.0:
        raise ValueError("Both lower bound and upper bound have to be within (-1, 1).")

    # init an empty tensor and define the domain sets
    acos_extrap = torch.empty_like(x)
    x_upper = x >= upper_bound
    x_lower = x <= lower_bound
    x_mid = (~x_upper) & (~x_lower)

    # acos calculation for upper_bound < x < lower_bound
    acos_extrap[x_mid] = torch.acos(x[x_mid])
    # the linear extrapolation for x >= upper_bound
    acos_extrap[x_upper] = _acos_linear_approximation(x[x_upper], upper_bound)
    # the linear extrapolation for x <= lower_bound
    acos_extrap[x_lower] = _acos_linear_approximation(x[x_lower], lower_bound)

    return acos_extrap


def _acos_linear_approximation(x: torch.Tensor, x0: float) -> torch.Tensor:
    """
    Calculates the 1st order Taylor expansion of `arccos(x)` around `x0`.
    """
    return (x - x0) * _dacos_dx(x0) + math.acos(x0)


def _dacos_dx(x: float) -> float:
    """
    Calculates the derivative of `arccos(x)` w.r.t. `x`.
    """
    return (-1.0) / math.sqrt(1.0 - x * x)
