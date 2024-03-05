# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import collections
import dataclasses
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, Iterator

import torch


@contextmanager
def evaluating(net: torch.nn.Module):
    """Temporarily switch to evaluation mode."""
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()


def try_to_cuda(t: Any) -> Any:
    """
    Try to move the input variable `t` to a cuda device.

    Args:
        t: Input.

    Returns:
        t_cuda: `t` moved to a cuda device, if supported.
    """
    try:
        t = t.cuda()
    except AttributeError:
        pass
    return t


def try_to_cpu(t: Any) -> Any:
    """
    Try to move the input variable `t` to a cpu device.

    Args:
        t: Input.

    Returns:
        t_cpu: `t` moved to a cpu device, if supported.
    """
    try:
        t = t.cpu()
    except AttributeError:
        pass
    return t


def dict_to_cuda(batch: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Move all values in a dictionary to cuda if supported.

    Args:
        batch: Input dict.

    Returns:
        batch_cuda: `batch` moved to a cuda device, if supported.
    """
    return {k: try_to_cuda(v) for k, v in batch.items()}


def dict_to_cpu(batch):
    """
    Move all values in a dictionary to cpu if supported.

    Args:
        batch: Input dict.

    Returns:
        batch_cpu: `batch` moved to a cpu device, if supported.
    """
    return {k: try_to_cpu(v) for k, v in batch.items()}


def dataclass_to_cuda_(obj):
    """
    Move all contents of a dataclass to cuda inplace if supported.

    Args:
        batch: Input dataclass.

    Returns:
        batch_cuda: `batch` moved to a cuda device, if supported.
    """
    for f in dataclasses.fields(obj):
        setattr(obj, f.name, try_to_cuda(getattr(obj, f.name)))
    return obj


def dataclass_to_cpu_(obj):
    """
    Move all contents of a dataclass to cpu inplace if supported.

    Args:
        batch: Input dataclass.

    Returns:
        batch_cuda: `batch` moved to a cpu device, if supported.
    """
    for f in dataclasses.fields(obj):
        setattr(obj, f.name, try_to_cpu(getattr(obj, f.name)))
    return obj


# TODO: test it
def cat_dataclass(batch, tensor_collator: Callable):
    """
    Concatenate all fields of a list of dataclasses `batch` to a single
    dataclass object using `tensor_collator`.

    Args:
        batch: Input list of dataclasses.

    Returns:
        concatenated_batch: All elements of `batch` concatenated to a single
            dataclass object.
        tensor_collator: The function used to concatenate tensor fields.
    """

    elem = batch[0]
    collated = {}

    for f in dataclasses.fields(elem):
        elem_f = getattr(elem, f.name)
        if elem_f is None:
            collated[f.name] = None
        elif torch.is_tensor(elem_f):
            collated[f.name] = tensor_collator([getattr(e, f.name) for e in batch])
        elif dataclasses.is_dataclass(elem_f):
            collated[f.name] = cat_dataclass(
                [getattr(e, f.name) for e in batch], tensor_collator
            )
        elif isinstance(elem_f, collections.abc.Mapping):
            collated[f.name] = {
                k: (
                    tensor_collator([getattr(e, f.name)[k] for e in batch])
                    if elem_f[k] is not None
                    else None
                )
                for k in elem_f
            }
        else:
            raise ValueError("Unsupported field type for concatenation")

    return type(elem)(**collated)


def recursive_visitor(it: Iterable[Any]) -> Iterator[Any]:
    for x in it:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from recursive_visitor(x)
        else:
            yield x


def get_inlier_indicators(
    tensor: torch.Tensor, dim: int, outlier_rate: float
) -> torch.Tensor:
    remove_elements = int(min(outlier_rate, 1.0) * tensor.shape[dim] / 2)
    hi = torch.topk(tensor, remove_elements, dim=dim).indices.tolist()
    lo = torch.topk(-tensor, remove_elements, dim=dim).indices.tolist()
    remove_indices = set(recursive_visitor([hi, lo]))
    keep_indices = tensor.new_ones(tensor.shape[dim : dim + 1], dtype=torch.bool)
    keep_indices[list(remove_indices)] = False
    return keep_indices


class Timer:
    """
    A simple class for timing execution.

    Example::

        with Timer():
            print("This print statement is timed.")

    """

    def __init__(self, name="timer", quiet=False):
        self.name = name
        self.quiet = quiet

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        if not self.quiet:
            print("%20s: %1.6f sec" % (self.name, self.interval))
