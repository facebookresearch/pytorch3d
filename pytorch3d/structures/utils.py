# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List, Sequence, Tuple, Union

import torch


"""
Util functions for points/verts/faces/volumes.
"""


def list_to_padded(
    x: Union[List[torch.Tensor], Tuple[torch.Tensor]],
    pad_size: Union[Sequence[int], None] = None,
    pad_value: float = 0.0,
    equisized: bool = False,
) -> torch.Tensor:
    r"""
    Transforms a list of N tensors each of shape (Si_0, Si_1, ... Si_D)
    into:
    - a single tensor of shape (N, pad_size(0), pad_size(1), ..., pad_size(D))
      if pad_size is provided
    - or a tensor of shape (N, max(Si_0), max(Si_1), ..., max(Si_D)) if pad_size is None.

    Args:
      x: list of Tensors
      pad_size: list(int) specifying the size of the padded tensor.
        If `None` (default), the largest size of each dimension
        is set as the `pad_size`.
      pad_value: float value to be used to fill the padded tensor
      equisized: bool indicating whether the items in x are of equal size
        (sometimes this is known and if provided saves computation)

    Returns:
      x_padded: tensor consisting of padded input tensors stored
        over the newly allocated memory.
    """
    if equisized:
        return torch.stack(x, 0)

    if not all(torch.is_tensor(y) for y in x):
        raise ValueError("All items have to be instances of a torch.Tensor.")

    # we set the common number of dimensions to the maximum
    # of the dimensionalities of the tensors in the list
    element_ndim = max(y.ndim for y in x)

    # replace empty 1D tensors with empty tensors with a correct number of dimensions
    x = [
        (y.new_zeros([0] * element_ndim) if (y.ndim == 1 and y.nelement() == 0) else y)
        for y in x
    ]

    if any(y.ndim != x[0].ndim for y in x):
        raise ValueError("All items have to have the same number of dimensions!")

    if pad_size is None:
        pad_dims = [
            max(y.shape[dim] for y in x if len(y) > 0) for dim in range(x[0].ndim)
        ]
    else:
        if any(len(pad_size) != y.ndim for y in x):
            raise ValueError("Pad size must contain target size for all dimensions.")
        pad_dims = pad_size

    N = len(x)
    x_padded = x[0].new_full((N, *pad_dims), pad_value)
    for i, y in enumerate(x):
        if len(y) > 0:
            slices = (i, *(slice(0, y.shape[dim]) for dim in range(y.ndim)))
            x_padded[slices] = y
    return x_padded


def padded_to_list(
    x: torch.Tensor,
    split_size: Union[Sequence[int], Sequence[Sequence[int]], None] = None,
):
    r"""
    Transforms a padded tensor of shape (N, S_1, S_2, ..., S_D) into a list
    of N tensors of shape:
    - (Si_1, Si_2, ..., Si_D) where (Si_1, Si_2, ..., Si_D) is specified in split_size(i)
    - or (S_1, S_2, ..., S_D) if split_size is None
    - or (Si_1, S_2, ..., S_D) if split_size(i) is an integer.

    Args:
      x: tensor
      split_size: optional 1D or 2D list/tuple of ints defining the number of
        items for each tensor.

    Returns:
      x_list: a list of tensors sharing the memory with the input.
    """
    x_list = list(x.unbind(0))

    if split_size is None:
        return x_list

    N = len(split_size)
    if x.shape[0] != N:
        raise ValueError("Split size must be of same length as inputs first dimension")

    for i in range(N):
        if isinstance(split_size[i], int):
            x_list[i] = x_list[i][: split_size[i]]
        else:
            slices = tuple(slice(0, s) for s in split_size[i])  # pyre-ignore
            x_list[i] = x_list[i][slices]
    return x_list


def list_to_packed(x: List[torch.Tensor]):
    r"""
    Transforms a list of N tensors each of shape (Mi, K, ...) into a single
    tensor of shape (sum(Mi), K, ...).

    Args:
      x: list of tensors.

    Returns:
        4-element tuple containing

        - **x_packed**: tensor consisting of packed input tensors along the
          1st dimension.
        - **num_items**: tensor of shape N containing Mi for each element in x.
        - **item_packed_first_idx**: tensor of shape N indicating the index of
          the first item belonging to the same element in the original list.
        - **item_packed_to_list_idx**: tensor of shape sum(Mi) containing the
          index of the element in the list the item belongs to.
    """
    if not x:
        raise ValueError("Input list is empty")
    device = x[0].device
    sizes = [xi.shape[0] for xi in x]
    sizes_total = sum(sizes)
    num_items = torch.tensor(sizes, dtype=torch.int64, device=device)
    item_packed_first_idx = torch.zeros_like(num_items)
    item_packed_first_idx[1:] = torch.cumsum(num_items[:-1], dim=0)
    item_packed_to_list_idx = torch.arange(
        sizes_total, dtype=torch.int64, device=device
    )
    item_packed_to_list_idx = (
        torch.bucketize(item_packed_to_list_idx, item_packed_first_idx, right=True) - 1
    )
    x_packed = torch.cat(x, dim=0)

    return x_packed, num_items, item_packed_first_idx, item_packed_to_list_idx


def packed_to_list(x: torch.Tensor, split_size: Union[list, int]):
    r"""
    Transforms a tensor of shape (sum(Mi), K, L, ...) to N set of tensors of
    shape (Mi, K, L, ...) where Mi's are defined in split_size

    Args:
      x: tensor
      split_size: list, tuple or int defining the number of items for each tensor
        in the output list.

    Returns:
      x_list: A list of Tensors
    """
    return x.split(split_size, dim=0)


def padded_to_packed(
    x: torch.Tensor,
    split_size: Union[list, tuple, None] = None,
    pad_value: Union[float, int, None] = None,
):
    r"""
    Transforms a padded tensor of shape (N, M, K) into a packed tensor
    of shape:
     - (sum(Mi), K) where (Mi, K) are the dimensions of
        each of the tensors in the batch and Mi is specified by split_size(i)
     - (N*M, K) if split_size is None

    Support only for 3-dimensional input tensor and 1-dimensional split size.

    Args:
      x: tensor
      split_size: list, tuple or int defining the number of items for each tensor
        in the output list.
      pad_value: optional value to use to filter the padded values in the input
        tensor.

    Only one of split_size or pad_value should be provided, or both can be None.

    Returns:
      x_packed: a packed tensor.
    """
    if x.ndim != 3:
        raise ValueError("Supports only 3-dimensional input tensors")

    N, M, D = x.shape

    if split_size is not None and pad_value is not None:
        raise ValueError("Only one of split_size or pad_value should be provided.")

    x_packed = x.reshape(-1, D)  # flatten padded

    if pad_value is None and split_size is None:
        return x_packed

    # Convert to packed using pad value
    if pad_value is not None:
        mask = x_packed.ne(pad_value).any(-1)
        x_packed = x_packed[mask]
        return x_packed

    # Convert to packed using split sizes
    # pyre-fixme[6]: Expected `Sized` for 1st param but got `Union[None,
    #  List[typing.Any], typing.Tuple[typing.Any, ...]]`.
    N = len(split_size)
    if x.shape[0] != N:
        raise ValueError("Split size must be of same length as inputs first dimension")

    # pyre-fixme[16]: `None` has no attribute `__iter__`.
    if not all(isinstance(i, int) for i in split_size):
        raise ValueError(
            "Support only 1-dimensional unbinded tensor. \
                Split size for more dimensions provided"
        )

    padded_to_packed_idx = torch.cat(
        [
            torch.arange(v, dtype=torch.int64, device=x.device) + i * M
            # pyre-fixme[6]: Expected `Iterable[Variable[_T]]` for 1st param but got
            #  `Union[None, List[typing.Any], typing.Tuple[typing.Any, ...]]`.
            for (i, v) in enumerate(split_size)
        ],
        dim=0,
    )

    return x_packed[padded_to_packed_idx]
