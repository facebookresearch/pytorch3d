# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import List, Union

import torch


"""
Util functions containing representation transforms for points/verts/faces.
"""


def list_to_padded(
    x: List[torch.Tensor],
    pad_size: Union[list, tuple, None] = None,
    pad_value: float = 0.0,
    equisized: bool = False,
) -> torch.Tensor:
    r"""
    Transforms a list of N tensors each of shape (Mi, Ki) into a single tensor
    of shape (N, pad_size(0), pad_size(1)), or (N, max(Mi), max(Ki))
    if pad_size is None.

    Args:
      x: list of Tensors
      pad_size: list(int) specifying the size of the padded tensor
      pad_value: float value to be used to fill the padded tensor
      equisized: bool indicating whether the items in x are of equal size
        (sometimes this is known and if provided saves computation)

    Returns:
      x_padded: tensor consisting of padded input tensors
    """
    if equisized:
        return torch.stack(x, 0)

    if pad_size is None:
        pad_dim0 = max(y.shape[0] for y in x if len(y) > 0)
        pad_dim1 = max(y.shape[1] for y in x if len(y) > 0)
    else:
        if len(pad_size) != 2:
            raise ValueError("Pad size must contain target size for 1st and 2nd dim")
        pad_dim0, pad_dim1 = pad_size

    N = len(x)
    x_padded = torch.full(
        (N, pad_dim0, pad_dim1), pad_value, dtype=x[0].dtype, device=x[0].device
    )
    for i, y in enumerate(x):
        if len(y) > 0:
            # pyre-fixme[16]: `Tensor` has no attribute `ndim`.
            if y.ndim != 2:
                raise ValueError("Supports only 2-dimensional tensor items")
            x_padded[i, : y.shape[0], : y.shape[1]] = y
    return x_padded


def padded_to_list(x: torch.Tensor, split_size: Union[list, tuple, None] = None):
    r"""
    Transforms a padded tensor of shape (N, M, K) into a list of N tensors
    of shape (Mi, Ki) where (Mi, Ki) is specified in split_size(i), or of shape
    (M, K) if split_size is None.
    Support only for 3-dimensional input tensor.

    Args:
      x: tensor
      split_size: list, tuple or int defining the number of items for each tensor
        in the output list.

    Returns:
      x_list: a list of tensors
    """
    # pyre-fixme[16]: `Tensor` has no attribute `ndim`.
    if x.ndim != 3:
        raise ValueError("Supports only 3-dimensional input tensors")
    x_list = list(x.unbind(0))

    if split_size is None:
        return x_list

    N = len(split_size)
    if x.shape[0] != N:
        raise ValueError("Split size must be of same length as inputs first dimension")

    for i in range(N):
        if isinstance(split_size[i], int):
            x_list[i] = x_list[i][: split_size[i]]
        elif len(split_size[i]) == 2:
            x_list[i] = x_list[i][: split_size[i][0], : split_size[i][1]]
        else:
            raise ValueError(
                "Support only for 2-dimensional unbinded tensor. \
                    Split size for more dimensions provided"
            )
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
    N = len(x)
    num_items = torch.zeros(N, dtype=torch.int64, device=x[0].device)
    item_packed_first_idx = torch.zeros(N, dtype=torch.int64, device=x[0].device)
    item_packed_to_list_idx = []
    cur = 0
    for i, y in enumerate(x):
        num = len(y)
        num_items[i] = num
        item_packed_first_idx[i] = cur
        item_packed_to_list_idx.append(
            torch.full((num,), i, dtype=torch.int64, device=y.device)
        )
        cur += num

    x_packed = torch.cat(x, dim=0)
    item_packed_to_list_idx = torch.cat(item_packed_to_list_idx, dim=0)

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
    # pyre-fixme[16]: `Tensor` has no attribute `split`.
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
    # pyre-fixme[16]: `Tensor` has no attribute `ndim`.
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
        # pyre-fixme[16]: `ByteTensor` has no attribute `any`.
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
