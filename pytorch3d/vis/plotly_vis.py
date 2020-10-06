# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import warnings
from typing import NamedTuple, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes, Pointclouds


class AxisArgs(NamedTuple):
    showgrid: bool = False
    zeroline: bool = False
    showline: bool = False
    ticks: str = ""
    showticklabels: bool = False
    backgroundcolor: str = "#fff"
    showaxeslabels: bool = False


class Lighting(NamedTuple):
    ambient: float = 0.8
    diffuse: float = 1.0
    fresnel: float = 0.0
    specular: float = 0.0
    roughness: float = 0.5
    facenormalsepsilon: float = 1e-6
    vertexnormalsepsilon: float = 1e-12


def plot_meshes(meshes: Meshes, *, in_subplots: bool = False, ncols: int = 1, **kwargs):
    """
    Takes a Meshes object and generates a plotly figure. If there is more than
    one mesh in the batch and in_subplots=True, each mesh will be
    visualized in an individual subplot with ncols number of subplots in the same row.
    Otherwise, each mesh in the batch will be visualized as an individual trace in the
    same plot. If the Meshes object has vertex colors defined as its texture, the vertex
    colors will be used for generating the plotly figure. Otherwise plotly's default
    colors will be used.

    Args:
        meshes: Meshes object to be visualized in a plotly figure.
        in_subplots: if each mesh in the batch should be visualized in an individual subplot
        ncols: number of subplots in the same row if in_subplots is set to be True. Otherwise
            ncols will be ignored.
        **kwargs: Accepts lighting (a Lighting object) and lightposition for Mesh3D and any of
            the args xaxis, yaxis and zaxis accept for scene. Accepts axis_args which is an
            AxisArgs object that is applied to the 3 axes. Also accepts subplot_titles, which
            should be a list of string titles matching the number of subplots.
            Example settings for axis_args and lighting are given above.

    Returns:
        Plotly figure of the mesh. If there is more than one mesh in the batch,
            the plotly figure will contain a series of vertically stacked subplots.
    """
    meshes = meshes.detach().cpu()
    subplot_titles = kwargs.get("subplot_titles", None)
    fig = _gen_fig_with_subplots(len(meshes), in_subplots, ncols, subplot_titles)
    for i in range(len(meshes)):
        verts = meshes[i].verts_packed()
        faces = meshes[i].faces_packed()
        # If mesh has vertex colors defined as texture, use vertex colors
        # for figure, otherwise use plotly's default colors.
        verts_rgb = None
        if isinstance(meshes[i].textures, TexturesVertex):
            verts_rgb = meshes[i].textures.verts_features_packed()
            verts_rgb.clamp_(min=0.0, max=1.0)
            verts_rgb = torch.tensor(255.0) * verts_rgb

        # Reposition the unused vertices to be "inside" the object
        # (i.e. they won't be visible in the plot).
        verts_used = torch.zeros((verts.shape[0],), dtype=torch.bool)
        verts_used[torch.unique(faces)] = True
        verts_center = verts[verts_used].mean(0)
        verts[~verts_used] = verts_center

        trace_row = i // ncols + 1 if in_subplots else 1
        trace_col = i % ncols + 1 if in_subplots else 1
        fig.add_trace(
            go.Mesh3d(  # pyre-ignore[16]
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                vertexcolor=verts_rgb,
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                lighting=kwargs.get("lighting", Lighting())._asdict(),
                lightposition=kwargs.get("lightposition", {}),
            ),
            row=trace_row,
            col=trace_col,
        )
        # Ensure update for every subplot.
        plot_scene = "scene" + str(i + 1) if in_subplots else "scene"
        current_layout = fig["layout"][plot_scene]

        axis_args = kwargs.get("axis_args", AxisArgs())

        xaxis, yaxis, zaxis = _gen_updated_axis_bounds(
            verts, verts_center, current_layout, axis_args
        )
        # Update the axis bounds with the axis settings passed in as kwargs.
        xaxis.update(**kwargs.get("xaxis", {}))
        yaxis.update(**kwargs.get("yaxis", {}))
        zaxis.update(**kwargs.get("zaxis", {}))

        current_layout.update(
            {"xaxis": xaxis, "yaxis": yaxis, "zaxis": zaxis, "aspectmode": "cube"}
        )
    return fig


def plot_pointclouds(
    pointclouds: Pointclouds,
    *,
    in_subplots: bool = False,
    ncols: int = 1,
    max_points: int = 20000,
    **kwargs,
):
    """
    Takes a Pointclouds object and generates a plotly figure. If there is more than
    one pointcloud in the batch, and in_subplots is set to be True, each pointcloud will be
    visualized in an individual subplot with ncols number of subplots in the same row.
    Otherwise, each pointcloud in the batch will be visualized as an individual trace in the
    same plot. If the Pointclouds object has features that are size (3) or (4) then those
    rgb/rgba values will be used for the plotly figure. Otherwise, plotly's default colors
    will be used. Assumes that all rgb/rgba feature values are in the range [0,1].

    Args:
        pointclouds: Pointclouds object which can contain a batch of pointclouds.
        in_subplots: if each pointcloud should be visualized in an individual subplot.
        ncols: number of subplots in the same row if in_subplots is set to be True. Otherwise
            ncols will be ignored.
        max_points: maximum number of points to plot. If the cloud has more, they are
            randomly subsampled.
        **kwargs: Accepts lighting (a Lighting object) and lightposition for Scatter3D
            and any of the args xaxis, yaxis and zaxis which scene accepts.
            Accepts axis_args which is an AxisArgs object that is applied to the 3 axes.
            Also accepts subplot_titles, whichshould be a list of string titles
            matching the number of subplots. Example settings for axis_args and lighting are
            given at the top of this file.

    Returns:
        Plotly figure of the pointcloud(s). If there is more than one pointcloud in the batch,
            the plotly figure will contain a plot with one trace per pointcloud, or with each
            pointcloud in a separate subplot if in_subplots is True.
    """
    pointclouds = pointclouds.detach().cpu()
    subplot_titles = kwargs.get("subplot_titles", None)
    fig = _gen_fig_with_subplots(len(pointclouds), in_subplots, ncols, subplot_titles)
    for i in range(len(pointclouds)):
        verts = pointclouds[i].points_packed()
        features = pointclouds[i].features_packed()

        indices = None
        if max_points is not None and verts.shape[0] > max_points:
            indices = np.random.choice(verts.shape[0], max_points, replace=False)
            verts = verts[indices]

        color = None
        if features is not None:
            features = features[indices]
            if features.shape[1] == 4:  # rgba
                template = "rgb(%d, %d, %d, %f)"
                rgb = (features[:, :3] * 255).int()
                color = [
                    template % (*rgb_, a_) for rgb_, a_ in zip(rgb, features[:, 3])
                ]

            if features.shape[1] == 3:
                template = "rgb(%d, %d, %d)"
                rgb = (features * 255).int()
                color = [template % (r, g, b) for r, g, b in rgb]

        trace_row = i // ncols + 1 if in_subplots else 1
        trace_col = i % ncols + 1 if in_subplots else 1
        fig.add_trace(
            go.Scatter3d(  # pyre-ignore[16]
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                marker={"color": color, "size": 1},
                mode="markers",
            ),
            row=trace_row,
            col=trace_col,
        )

        # Ensure update for every subplot.
        plot_scene = "scene" + str(i + 1) if in_subplots else "scene"
        current_layout = fig["layout"][plot_scene]

        verts_center = verts.mean(0)

        axis_args = kwargs.get("axis_args", AxisArgs())

        xaxis, yaxis, zaxis = _gen_updated_axis_bounds(
            verts, verts_center, current_layout, axis_args
        )
        xaxis.update(**kwargs.get("xaxis", {}))
        yaxis.update(**kwargs.get("yaxis", {}))
        zaxis.update(**kwargs.get("zaxis", {}))

        current_layout.update(
            {"xaxis": xaxis, "yaxis": yaxis, "zaxis": zaxis, "aspectmode": "cube"}
        )

    return fig


def _gen_fig_with_subplots(
    batch_size: int, in_subplots: bool, ncols: int, subplot_titles: Optional[list]
):
    """
    Takes in the number of objects to be plotted and generate a plotly figure
    with the appropriate number and orientation of subplots
    Args:
        batch_size: the number of elements in the batch of objects to be visualized.
        in_subplots: if each object should be visualized in an individual subplot.
        ncols: number of subplots in the same row if in_subplots is set to be True. Otherwise
            ncols will be ignored.
        subplot_titles: titles for the subplot(s). list of strings of length batch_size
            if in_subplots is True, otherwise length 1.

    Returns:
        Plotly figure with one plot if in_subplots is false. Otherwise, returns a plotly figure
        with ncols subplots per row.
    """
    if batch_size % ncols != 0:
        msg = "ncols is invalid for the given mesh batch size."
        warnings.warn(msg)
    fig_rows = batch_size // ncols if in_subplots else 1
    fig_cols = ncols if in_subplots else 1
    fig_type = [{"type": "scene"}]
    specs = [fig_type * fig_cols] * fig_rows
    # subplot_titles must have one title per subplot
    if subplot_titles is not None and len(subplot_titles) != fig_cols * fig_rows:
        subplot_titles = None
    fig = make_subplots(
        rows=fig_rows,
        cols=fig_cols,
        specs=specs,
        subplot_titles=subplot_titles,
        column_widths=[1.0] * fig_cols,
    )
    return fig


def _gen_updated_axis_bounds(
    verts: torch.Tensor,
    verts_center: torch.Tensor,
    current_layout: go.Scene,  # pyre-ignore[11]
    axis_args: AxisArgs,
) -> Tuple[dict, dict, dict]:
    """
    Takes in the vertices, center point of the vertices, and the current plotly figure and
    outputs axes with bounds that capture all points in the current subplot.
    Args:
        verts: tensor of size (N, 3) representing N points with xyz coordinates.
        verts_center: tensor of size (3) corresponding to the center point of verts.
        current_layout: the current plotly figure layout scene corresponding to verts' trace.
        axis_args: an AxisArgs object with default and/or user-set values for plotly's axes.

    Returns:
        a 3 item tuple of xaxis, yaxis, and zaxis, which are dictionaries with axis arguments
        for plotly including a range key with value the minimum and maximum value for that axis.
    """
    # Get ranges of vertices.
    max_expand = (verts.max(0)[0] - verts.min(0)[0]).max()
    verts_min = verts_center - max_expand
    verts_max = verts_center + max_expand
    bounds = torch.t(torch.stack((verts_min, verts_max)))

    # Ensure that within a subplot, the bounds capture all traces
    old_xrange, old_yrange, old_zrange = (
        current_layout["xaxis"]["range"],
        current_layout["yaxis"]["range"],
        current_layout["zaxis"]["range"],
    )
    x_range, y_range, z_range = bounds
    if old_xrange is not None:
        x_range[0] = min(x_range[0], old_xrange[0])
        x_range[1] = max(x_range[1], old_xrange[1])
    if old_yrange is not None:
        y_range[0] = min(y_range[0], old_yrange[0])
        y_range[1] = max(y_range[1], old_yrange[1])
    if old_zrange is not None:
        z_range[0] = min(z_range[0], old_zrange[0])
        z_range[1] = max(z_range[1], old_zrange[1])
    axis_args_dict = axis_args._asdict()
    xaxis = {"range": x_range, **axis_args_dict}
    yaxis = {"range": y_range, **axis_args_dict}
    zaxis = {"range": z_range, **axis_args_dict}
    return xaxis, yaxis, zaxis
