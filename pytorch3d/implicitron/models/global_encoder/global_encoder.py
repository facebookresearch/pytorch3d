# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union

import torch
from pytorch3d.implicitron.tools.config import (
    registry,
    ReplaceableBase,
    run_auto_creation,
)
from pytorch3d.renderer.implicit import HarmonicEmbedding

from .autodecoder import Autodecoder


class GlobalEncoderBase(ReplaceableBase):
    """
    A base class for implementing encoders of global frame-specific quantities.

    The latter includes e.g. the harmonic encoding of a frame timestamp
    (`HarmonicTimeEncoder`), or an autodecoder encoding of the frame's sequence
    (`SequenceAutodecoder`).
    """

    def get_encoding_dim(self):
        """
        Returns the dimensionality of the returned encoding.
        """
        raise NotImplementedError()

    def calculate_squared_encoding_norm(self) -> Optional[torch.Tensor]:
        """
        Calculates the squared norm of the encoding to report as the
        `autodecoder_norm` loss of the model, as a zero dimensional tensor.
        """
        raise NotImplementedError()

    def forward(
        self,
        *,
        frame_timestamp: Optional[torch.Tensor] = None,
        sequence_name: Optional[Union[torch.LongTensor, List[str]]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Given a set of inputs to encode, generates a tensor containing the encoding.

        Returns:
            encoding: The tensor containing the global encoding.
        """
        raise NotImplementedError()


# TODO: probabilistic embeddings?
@registry.register
class SequenceAutodecoder(GlobalEncoderBase, torch.nn.Module):  # pyre-ignore: 13
    """
    A global encoder implementation which provides an autodecoder encoding
    of the frame's sequence identifier.
    """

    autodecoder: Autodecoder

    def __post_init__(self):
        run_auto_creation(self)

    def get_encoding_dim(self):
        return self.autodecoder.get_encoding_dim()

    def forward(
        self,
        *,
        frame_timestamp: Optional[torch.Tensor] = None,
        sequence_name: Optional[Union[torch.LongTensor, List[str]]] = None,
        **kwargs,
    ) -> torch.Tensor:
        if sequence_name is None:
            raise ValueError("sequence_name must be provided.")
        # run dtype checks and pass sequence_name to self.autodecoder
        return self.autodecoder(sequence_name)

    def calculate_squared_encoding_norm(self) -> Optional[torch.Tensor]:
        return self.autodecoder.calculate_squared_encoding_norm()


@registry.register
class HarmonicTimeEncoder(GlobalEncoderBase, torch.nn.Module):
    """
    A global encoder implementation which provides harmonic embeddings
    of each frame's timestamp.
    """

    n_harmonic_functions: int = 10
    append_input: bool = True
    time_divisor: float = 1.0

    def __post_init__(self):
        self._harmonic_embedding = HarmonicEmbedding(
            n_harmonic_functions=self.n_harmonic_functions,
            append_input=self.append_input,
        )

    def get_encoding_dim(self):
        return self._harmonic_embedding.get_output_dim(1)

    def forward(
        self,
        *,
        frame_timestamp: Optional[torch.Tensor] = None,
        sequence_name: Optional[Union[torch.LongTensor, List[str]]] = None,
        **kwargs,
    ) -> torch.Tensor:
        if frame_timestamp is None:
            raise ValueError("frame_timestamp must be provided.")
        if frame_timestamp.shape[-1] != 1:
            raise ValueError("Frame timestamp's last dimensions should be one.")
        time = frame_timestamp / self.time_divisor
        return self._harmonic_embedding(time)

    def calculate_squared_encoding_norm(self) -> Optional[torch.Tensor]:
        return None
