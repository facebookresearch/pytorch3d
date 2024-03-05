# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import warnings


try:
    from .plotly_vis import get_camera_wireframe, plot_batch_individually, plot_scene
except ModuleNotFoundError as err:
    if "plotly" in str(err):
        warnings.warn(
            "Cannot import plotly-based visualization code."
            " Please install plotly to enable (pip install plotly)."
        )
    else:
        raise

from .texture_vis import texturesuv_image_matplotlib, texturesuv_image_PIL
