# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Union

import torch
from pytorch3d.implicitron.tools.config import Configurable


# TODO: probabilistic embeddings?
class Autodecoder(Configurable, torch.nn.Module):
    """
    Autodecoder module

    Settings:
        encoding_dim: Embedding dimension for the decoder.
        n_instances: The maximum number of instances stored by the autodecoder.
        init_scale: Scale factor for the initial autodecoder weights.
        ignore_input: If `True`, optimizes a single code for any input.
    """

    encoding_dim: int = 0
    n_instances: int = 0
    init_scale: float = 1.0
    ignore_input: bool = False

    def __post_init__(self):
        super().__init__()
        if self.n_instances <= 0:
            # Do not init the codes at all in case we have 0 instances.
            return
        self._autodecoder_codes = torch.nn.Embedding(
            self.n_instances,
            self.encoding_dim,
            scale_grad_by_freq=True,
        )
        with torch.no_grad():
            # weight has been initialised from Normal(0, 1)
            self._autodecoder_codes.weight *= self.init_scale

        self._sequence_map = self._build_sequence_map()
        # Make sure to register hooks for correct handling of saving/loading
        # the module's _sequence_map.
        self._register_load_state_dict_pre_hook(self._load_sequence_map_hook)
        self._register_state_dict_hook(_save_sequence_map_hook)

    def _build_sequence_map(
        self, sequence_map_dict: Optional[Dict[str, int]] = None
    ) -> Dict[str, int]:
        """
        Args:
            sequence_map_dict: A dictionary used to initialize the sequence_map.

        Returns:
            sequence_map: a dictionary of key: id pairs.
        """
        # increments the counter when asked for a new value
        sequence_map = defaultdict(iter(range(self.n_instances)).__next__)
        if sequence_map_dict is not None:
            # Assign all keys from the loaded sequence_map_dict to self._sequence_map.
            # Since this is done in the original order, it should generate
            # the same set of key:id pairs. We check this with an assert to be sure.
            for x, x_id in sequence_map_dict.items():
                x_id_ = sequence_map[x]
                assert x_id == x_id_
        return sequence_map

    def calc_squared_encoding_norm(self):
        if self.n_instances <= 0:
            return None
        return (self._autodecoder_codes.weight ** 2).mean()

    def get_encoding_dim(self) -> int:
        if self.n_instances <= 0:
            return 0
        return self.encoding_dim

    def forward(self, x: Union[torch.LongTensor, List[str]]) -> Optional[torch.Tensor]:
        """
        Args:
            x: A batch of `N` sequence identifiers. Either a long tensor of size
            `(N,)` keys in [0, n_instances), or a list of `N` string keys that
            are hashed to codes (without collisions).

        Returns:
            codes: A tensor of shape `(N, self.encoding_dim)` containing the
                sequence-specific autodecoder codes.
        """
        if self.n_instances == 0:
            return None

        if self.ignore_input:
            x = ["singleton"]

        if isinstance(x[0], str):
            try:
                x = torch.tensor(
                    # pyre-ignore[29]
                    [self._sequence_map[elem] for elem in x],
                    dtype=torch.long,
                    device=next(self.parameters()).device,
                )
            except StopIteration:
                raise ValueError("Not enough n_instances in the autodecoder")

        # pyre-fixme[29]: `Union[torch.Tensor, torch.nn.Module]` is not a function.
        return self._autodecoder_codes(x)

    def _load_sequence_map_hook(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """
        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            prefix (str): the prefix for parameters and buffers used in this
                module
            local_metadata (dict): a dict containing the metadata for this module.
            strict (bool): whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this module
            missing_keys (list of str): if ``strict=True``, add missing keys to
                this list
            unexpected_keys (list of str): if ``strict=True``, add unexpected
                keys to this list
            error_msgs (list of str): error messages should be added to this
                list, and will be reported together in
                :meth:`~torch.nn.Module.load_state_dict`

        Returns:
            Constructed sequence_map if it exists in the state_dict
            else raises a warning only.
        """
        sequence_map_key = prefix + "_sequence_map"
        if sequence_map_key in state_dict:
            sequence_map_dict = state_dict.pop(sequence_map_key)
            self._sequence_map = self._build_sequence_map(
                sequence_map_dict=sequence_map_dict
            )
        else:
            warnings.warn("No sequence map in Autodecoder state dict!")


def _save_sequence_map_hook(
    self,
    state_dict,
    prefix,
    local_metadata,
) -> None:
    """
    Args:
        state_dict (dict): a dict containing parameters and
            persistent buffers.
        prefix (str): the prefix for parameters and buffers used in this
            module
        local_metadata (dict): a dict containing the metadata for this module.
    """
    sequence_map_key = prefix + "_sequence_map"
    sequence_map_dict = dict(self._sequence_map.items())
    state_dict[sequence_map_key] = sequence_map_dict
