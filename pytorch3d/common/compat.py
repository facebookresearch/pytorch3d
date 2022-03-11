# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Sequence, Tuple, Union

import torch


"""
Some functions which depend on PyTorch versions.
"""


def solve(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:  # pragma: no cover
    """
    Like torch.linalg.solve, tries to return X
    such that AX=B, with A square.
    """
    if hasattr(torch, "linalg") and hasattr(torch.linalg, "solve"):
        # PyTorch version >= 1.8.0
        return torch.linalg.solve(A, B)

    return torch.solve(B, A).solution


def lstsq(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:  # pragma: no cover
    """
    Like torch.linalg.lstsq, tries to return X
    such that AX=B.
    """
    if hasattr(torch, "linalg") and hasattr(torch.linalg, "lstsq"):
        # PyTorch version >= 1.9
        return torch.linalg.lstsq(A, B).solution

    solution = torch.lstsq(B, A).solution
    if A.shape[1] < A.shape[0]:
        return solution[: A.shape[1]]
    return solution


def qr(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover
    """
    Like torch.linalg.qr.
    """
    if hasattr(torch, "linalg") and hasattr(torch.linalg, "qr"):
        # PyTorch version >= 1.9
        return torch.linalg.qr(A)
    return torch.qr(A)


def eigh(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover
    """
    Like torch.linalg.eigh, assuming the argument is a symmetric real matrix.
    """
    if hasattr(torch, "linalg") and hasattr(torch.linalg, "eigh"):
        return torch.linalg.eigh(A)
    return torch.symeig(A, eigenvectors=True)


def meshgrid_ij(
    *A: Union[torch.Tensor, Sequence[torch.Tensor]]
) -> Tuple[torch.Tensor, ...]:  # pragma: no cover
    """
    Like torch.meshgrid was before PyTorch 1.10.0, i.e. with indexing set to ij
    """
    if (
        torch.meshgrid.__kwdefaults__ is not None
        and "indexing" in torch.meshgrid.__kwdefaults__
    ):
        # PyTorch >= 1.10.0
        return torch.meshgrid(*A, indexing="ij")
    return torch.meshgrid(*A)
