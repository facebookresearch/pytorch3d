# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from abc import ABC, abstractmethod
from typing import Optional

from pytorch3d.implicitron.models.renderer.base import ImplicitronRayBundle

from pytorch3d.implicitron.tools.config import ReplaceableBase
from pytorch3d.renderer.cameras import CamerasBase


class ImplicitFunctionBase(ABC, ReplaceableBase):
    @abstractmethod
    def forward(
        self,
        *,
        ray_bundle: ImplicitronRayBundle,
        fun_viewpool=None,
        camera: Optional[CamerasBase] = None,
        global_code=None,
        **kwargs,
    ):
        raise NotImplementedError()

    @staticmethod
    def allows_multiple_passes() -> bool:
        """
        Returns True if this implicit function allows
        multiple passes.
        """
        return False

    @staticmethod
    def requires_pooling_without_aggregation() -> bool:
        """
        Returns True if this implicit function needs
        pooling without aggregation.
        """
        return False

    def on_bind_args(self) -> None:
        """
        Called when the custom args are fixed in the main model forward pass.
        """
        pass
