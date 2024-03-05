# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from .shapenet_core import ShapeNetCore


__all__ = [k for k in globals().keys() if not k.startswith("_")]
