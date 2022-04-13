# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw
from pytorch3d.renderer.mesh import TexturesUV


def texturesuv_image_matplotlib(
    texture: TexturesUV,
    *,
    texture_index: int = 0,
    radius: float = 1,
    color=(1.0, 0.0, 0.0),
    subsample: Optional[int] = 10000,
    origin: str = "upper",
) -> None:  # pragma: no cover
    """
    Plot the texture image for one element of a TexturesUV with
    matplotlib together with verts_uvs positions circled.
    In particular a value in verts_uvs which is never referenced
    in faces_uvs will still be plotted.
    This is for debugging purposes, e.g. to align the map with
    the uv coordinates. In particular, matplotlib
    is used which is not an official dependency of PyTorch3D.

    Args:
        texture: a TexturesUV object with one mesh
        texture_index: index in the batch to plot
        radius: plotted circle radius in pixels
        color: any matplotlib-understood color for the circles.
        subsample: if not None, number of points to plot.
                Otherwise all points are plotted.
        origin: "upper" or "lower" like matplotlib.imshow .
            upper (the default) matches texturesuv_image_PIL.
    """

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    texture_image = texture.maps_padded()
    centers = texture.centers_for_image(index=texture_index).numpy()

    ax = plt.gca()
    ax.imshow(texture_image[texture_index].detach().cpu().numpy(), origin=origin)

    n_points = centers.shape[0]
    if subsample is None or n_points <= subsample:
        indices = range(n_points)
    else:
        indices = np.random.choice(n_points, subsample, replace=False)
    for i in indices:
        # setting clip_on=False makes it obvious when
        # we have UV coordinates outside the correct range
        ax.add_patch(Circle(centers[i], radius, color=color, clip_on=False))


def texturesuv_image_PIL(
    texture: TexturesUV,
    *,
    texture_index: int = 0,
    radius: float = 1,
    color: Any = "red",
    subsample: Optional[int] = 10000,
):  # pragma: no cover
    """
    Return a PIL image of the texture image of one element of the batch
    from a TexturesUV, together with the verts_uvs positions circled.
    In particular a value in verts_uvs which is never referenced
    in faces_uvs will still be plotted.
    This is for debugging purposes, e.g. to align the map with
    the uv coordinates. In particular, matplotlib
    is used which is not an official dependency of PyTorch3D.

    Args:
        texture: a TexturesUV object with one mesh
        texture_index: index in the batch to plot
        radius: plotted circle radius in pixels
        color: any PIL-understood color for the circles.
        subsample: if not None, number of points to plot.
                Otherwise all points are plotted.

    Returns:
        PIL Image object.
    """

    centers = texture.centers_for_image(index=texture_index).numpy()
    texture_image = texture.maps_padded()
    texture_array = (texture_image[texture_index] * 255).cpu().numpy().astype(np.uint8)

    image = Image.fromarray(texture_array)
    draw = ImageDraw.Draw(image)

    n_points = centers.shape[0]
    if subsample is None or n_points <= subsample:
        indices = range(n_points)
    else:
        indices = np.random.choice(n_points, subsample, replace=False)

    for i in indices:
        x = centers[i][0]
        y = centers[i][1]
        draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], fill=color)

    return image
