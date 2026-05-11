# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from pytorch3d import _C


# The pulsar native renderer is not yet ported to ROCm/HIP; the C++ extension
# omits PulsarRenderer when built for AMD. Importing the Python wrapper would
# crash at class-definition time on those builds (it references _C.MAX_UINT /
# _C.PulsarRenderer at module load). Gate the wrapper symbol on availability
# of the native class so the rest of pytorch3d.renderer remains importable.
if hasattr(_C, "PulsarRenderer"):
    from .renderer import Renderer  # noqa: F401
