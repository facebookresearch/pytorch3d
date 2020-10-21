# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import warnings
from typing import Dict, List, NamedTuple, Union

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes, Pointclouds, join_meshes_as_scene


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


def plot_scene(
    plots: Dict[str, Dict[str, Union[Pointclouds, Meshes]]],
    *,
    ncols: int = 1,
    pointcloud_max_points: int = 20000,
    pointcloud_marker_size: int = 1,
    **kwargs,
):
    """
    Main function to visualize Meshes and Pointclouds.
    Plots input Pointclouds and Meshes data into named subplots,
    with named traces based on the dictionary keys.

    Args:
        plots: A dict containing subplot and trace names,
            as well as the Meshes and Pointclouds objects to be rendered.
            See below for examples of the format.
        ncols: the number of subplots per row
        pointcloud_max_points: the maximum number of points to plot from
            a pointcloud. If more are present, a random sample of size
            pointcloud_max_points is used.
        pointcloud_marker_size: the size of the points rendered by plotly
            when plotting a pointcloud.
        **kwargs: Accepts lighting (a Lighting object) and any of the args xaxis,
            yaxis and zaxis which Plotly's scene accepts. Accepts axis_args,
            which is an AxisArgs object that is applied to all 3 axes.
            Example settings for axis_args and lighting are given at the
            top of this file.

    Example:

    ..code-block::python

        mesh = ...
        point_cloud = ...
        fig = plot_scene({
            "subplot_title": {
                "mesh_trace_title": mesh,
                "pointcloud_trace_title": point_cloud
            }
        })
        fig.show()

    The above example will render one subplot which has both a mesh and pointcloud.

    If the Meshes or Pointclouds objects are batched, then every object in that batch
    will be plotted in a single trace.

    ..code-block::python
        mesh = ... # batch size 2
        point_cloud = ... # batch size 2
        fig = plot_scene({
            "subplot_title": {
                "mesh_trace_title": mesh,
                "pointcloud_trace_title": point_cloud
            }
        })
        fig.show()

    The above example renders one subplot with 2 traces, each of which renders
    both objects from their respective batched data.

    Multiple subplots follow the same pattern:
    ..code-block::python
        mesh = ... # batch size 2
        point_cloud = ... # batch size 2
        fig = plot_scene({
            "subplot1_title": {
                "mesh_trace_title": mesh[0],
                "pointcloud_trace_title": point_cloud[0]
            },
            "subplot2_title": {
                "mesh_trace_title": mesh[1],
                "pointcloud_trace_title": point_cloud[1]
            }
        },
        ncols=2)  # specify the number of subplots per row
        fig.show()

    The above example will render two subplots, each containing a mesh
    and a pointcloud. The ncols argument will render two subplots in one row
    instead of having them vertically stacked because the default is one subplot
    per row.

    For an example of using kwargs, see below:
    ..code-block::python
        mesh = ...
        point_cloud = ...
        fig = plot_scene({
            "subplot_title": {
                "mesh_trace_title": mesh,
                "pointcloud_trace_title": point_cloud
            }
        },
        axis_args=AxisArgs(backgroundcolor="rgb(200,230,200)")) # kwarg axis_args
        fig.show()

    The above example will render each axis with the input background color.

    See the tutorials in pytorch3d/docs/tutorials for more examples
    (namely rendered_color_points.ipynb and rendered_textured_meshes.ipynb).
    """
    subplots = list(plots.keys())
    fig = _gen_fig_with_subplots(len(subplots), ncols, subplots)
    lighting = kwargs.get("lighting", Lighting())._asdict()
    axis_args_dict = kwargs.get("axis_args", AxisArgs())._asdict()

    # Set axis arguments to defaults defined at the top of this file
    x_settings = {**axis_args_dict}
    y_settings = {**axis_args_dict}
    z_settings = {**axis_args_dict}

    # Update the axes with any axis settings passed in as kwargs.
    x_settings.update(**kwargs.get("xaxis", {}))
    y_settings.update(**kwargs.get("yaxis", {}))
    z_settings.update(**kwargs.get("zaxis", {}))

    camera = {
        "up": {
            "x": 0,
            "y": 1,
            "z": 0,
        }  # set the up vector to match PyTorch3D world coordinates conventions
    }

    for subplot_idx in range(len(subplots)):
        subplot_name = subplots[subplot_idx]
        traces = plots[subplot_name]
        for trace_name, struct in traces.items():
            if isinstance(struct, Meshes):
                _add_mesh_trace(fig, struct, trace_name, subplot_idx, ncols, lighting)
            elif isinstance(struct, Pointclouds):
                _add_pointcloud_trace(
                    fig,
                    struct,
                    trace_name,
                    subplot_idx,
                    ncols,
                    pointcloud_max_points,
                    pointcloud_marker_size,
                )
            else:
                raise ValueError(
                    "struct {} is not a Meshes or Pointclouds object".format(struct)
                )

        # Ensure update for every subplot.
        plot_scene = "scene" + str(subplot_idx + 1)
        current_layout = fig["layout"][plot_scene]
        xaxis = current_layout["xaxis"]
        yaxis = current_layout["yaxis"]
        zaxis = current_layout["zaxis"]

        # Update the axes with our above default and provided settings.
        xaxis.update(**x_settings)
        yaxis.update(**y_settings)
        zaxis.update(**z_settings)

        current_layout.update(
            {
                "xaxis": xaxis,
                "yaxis": yaxis,
                "zaxis": zaxis,
                "aspectmode": "cube",
                "camera": camera,
            }
        )

    return fig


def _add_mesh_trace(
    fig: go.Figure,  # pyre-ignore[11]
    meshes: Meshes,
    trace_name: str,
    subplot_idx: int,
    ncols: int,
    lighting: Lighting,
):
    """
    Adds a trace rendering a Meshes object to the passed in figure, with
    a given name and in a specific subplot.

    Args:
        fig: plotly figure to add the trace within.
        meshes: Meshes object to render. It can be batched.
        trace_name: name to label the trace with.
        subplot_idx: identifies the subplot, with 0 being the top left.
        ncols: the number of subplots per row.
        lighting: a Lighting object that specifies the Mesh3D lighting.
    """

    mesh = join_meshes_as_scene(meshes)
    mesh = mesh.detach().cpu()
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    # If mesh has vertex colors defined as texture, use vertex colors
    # for figure, otherwise use plotly's default colors.
    verts_rgb = None
    if isinstance(mesh.textures, TexturesVertex):
        verts_rgb = mesh.textures.verts_features_packed()
        verts_rgb.clamp_(min=0.0, max=1.0)
        verts_rgb = torch.tensor(255.0) * verts_rgb

    # Reposition the unused vertices to be "inside" the object
    # (i.e. they won't be visible in the plot).
    verts_used = torch.zeros((verts.shape[0],), dtype=torch.bool)
    verts_used[torch.unique(faces)] = True
    verts_center = verts[verts_used].mean(0)
    verts[~verts_used] = verts_center

    row, col = subplot_idx // ncols + 1, subplot_idx % ncols + 1
    fig.add_trace(
        go.Mesh3d(  # pyre-ignore[16]
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            vertexcolor=verts_rgb,
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            lighting=lighting,
            name=trace_name,
        ),
        row=row,
        col=col,
    )

    # Access the current subplot's scene configuration
    plot_scene = "scene" + str(subplot_idx + 1)
    current_layout = fig["layout"][plot_scene]

    # update the bounds of the axes for the current trace
    max_expand = (verts.max(0)[0] - verts.min(0)[0]).max()
    _update_axes_bounds(verts_center, max_expand, current_layout)


def _add_pointcloud_trace(
    fig: go.Figure,
    pointclouds: Pointclouds,
    trace_name: str,
    subplot_idx: int,
    ncols: int,
    max_points_per_pointcloud: int,
    marker_size: int,
):
    """
    Adds a trace rendering a Pointclouds object to the passed in figure, with
    a given name and in a specific subplot.

    Args:
        fig: plotly figure to add the trace within.
        pointclouds: Pointclouds object to render. It can be batched.
        trace_name: name to label the trace with.
        subplot_idx: identifies the subplot, with 0 being the top left.
        ncols: the number of sublpots per row.
        max_points_per_pointcloud: the number of points to render, which are randomly sampled.
        marker_size: the size of the rendered points
    """
    pointclouds = pointclouds.detach().cpu()
    verts = pointclouds.points_packed()
    features = pointclouds.features_packed()
    total_points_count = max_points_per_pointcloud * len(pointclouds)

    indices = None
    if verts.shape[0] > total_points_count:
        indices = np.random.choice(verts.shape[0], total_points_count, replace=False)
        verts = verts[indices]

    color = None
    if features is not None:
        features = features[indices]
        if features.shape[1] == 4:  # rgba
            template = "rgb(%d, %d, %d, %f)"
            rgb = (features[:, :3].clamp(0.0, 1.0) * 255).int()
            color = [template % (*rgb_, a_) for rgb_, a_ in zip(rgb, features[:, 3])]

        if features.shape[1] == 3:
            template = "rgb(%d, %d, %d)"
            rgb = (features.clamp(0.0, 1.0) * 255).int()
            color = [template % (r, g, b) for r, g, b in rgb]

    row = subplot_idx // ncols + 1
    col = subplot_idx % ncols + 1
    fig.add_trace(
        go.Scatter3d(  # pyre-ignore[16]
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            marker={"color": color, "size": marker_size},
            mode="markers",
            name=trace_name,
        ),
        row=row,
        col=col,
    )

    # Access the current subplot's scene configuration
    plot_scene = "scene" + str(subplot_idx + 1)
    current_layout = fig["layout"][plot_scene]

    # update the bounds of the axes for the current trace
    verts_center = verts.mean(0)
    max_expand = (verts.max(0)[0] - verts.min(0)[0]).max()
    _update_axes_bounds(verts_center, max_expand, current_layout)


def _gen_fig_with_subplots(batch_size: int, ncols: int, subplot_titles: List[str]):
    """
    Takes in the number of objects to be plotted and generate a plotly figure
    with the appropriate number and orientation of titled subplots.
    Args:
        batch_size: the number of elements in the batch of objects to be visualized.
        ncols: number of subplots in the same row.
        subplot_titles: titles for the subplot(s). list of strings of length batch_size.

    Returns:
        Plotly figure with ncols subplots per row, and batch_size subplots.
    """
    if batch_size % ncols != 0:
        msg = "ncols is invalid for the given mesh batch size."
        warnings.warn(msg)
    fig_rows = batch_size // ncols
    fig_cols = ncols
    fig_type = [{"type": "scene"}]
    specs = [fig_type * fig_cols] * fig_rows
    # subplot_titles must have one title per subplot
    fig = make_subplots(
        rows=fig_rows,
        cols=fig_cols,
        specs=specs,
        subplot_titles=subplot_titles,
        column_widths=[1.0] * fig_cols,
    )
    return fig


def _update_axes_bounds(
    verts_center: torch.Tensor,
    max_expand: float,
    current_layout: go.Scene,  # pyre-ignore[11]
):
    """
    Takes in the vertices' center point and max spread, and the current plotly figure
    layout and updates the layout to have bounds that include all traces for that subplot.
    Args:
        verts_center: tensor of size (3) corresponding to a trace's vertices' center point.
        max_expand: the maximum spread in any dimension of the trace's vertices.
        current_layout: the plotly figure layout scene corresponding to the referenced trace.
    """
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

    xaxis = {"range": x_range}
    yaxis = {"range": y_range}
    zaxis = {"range": z_range}
    current_layout.update({"xaxis": xaxis, "yaxis": yaxis, "zaxis": zaxis})
