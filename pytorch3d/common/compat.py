# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Sequence, Tuple, Union

import torch


"""
Some functions which depend on PyTorch or Python versions.
"""


def meshgrid_ij(
    *A: Union[torch.Tensor, Sequence[torch.Tensor]]
) -> Tuple[torch.Tensor, ...]:  # pragma: no cover
    """
    Like torch.meshgrid was before PyTorch 1.10.0, i.e. with indexing set to ij
    """
    if (
        # pyre-fixme[16]: Callable `meshgrid` has no attribute `__kwdefaults__`.
        torch.meshgrid.__kwdefaults__ is not None
        and "indexing" in torch.meshgrid.__kwdefaults__
    ):
        # PyTorch >= 1.10.0
        # pyre-fixme[6]: For 1st param expected `Union[List[Tensor], Tensor]` but
        #  got `Union[Sequence[Tensor], Tensor]`.
        return torch.meshgrid(*A, indexing="ij")
    # pyre-fixme[6]: For 1st param expected `Union[List[Tensor], Tensor]` but got
    #  `Union[Sequence[Tensor], Tensor]`.
    return torch.meshgrid(*A)


def prod(iterable, *, start=1):
    """
    Like math.prod in Python 3.8 and later.
    """
    for i in iterable:
        start *= i
    return start
