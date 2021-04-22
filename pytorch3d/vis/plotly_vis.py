# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import warnings
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from pytorch3d.renderer import TexturesVertex
from pytorch3d.renderer.camera_utils import camera_to_eye_at_up
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Meshes, Pointclouds, join_meshes_as_scene


def get_camera_wireframe(scale: float = 0.3):
    """
    Returns a wireframe of a 3D line-plot of a camera symbol.
    """
    a = 0.5 * torch.tensor([-2, 1.5, 4])
    up1 = 0.5 * torch.tensor([0, 1.5, 4])
    up2 = 0.5 * torch.tensor([0, 2, 4])
    b = 0.5 * torch.tensor([2, 1.5, 4])
    c = 0.5 * torch.tensor([-2, -1.5, 4])
    d = 0.5 * torch.tensor([2, -1.5, 4])
    C = torch.zeros(3)
    F = torch.tensor([0, 0, 3])
    camera_points = [a, up1, up2, up1, b, d, c, a, C, b, d, C, c, C, F]
    lines = torch.stack([x.float() for x in camera_points]) * scale
    return lines


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
    plots: Dict[str, Dict[str, Union[Pointclouds, Meshes, CamerasBase]]],
    *,
    viewpoint_cameras: Optional[CamerasBase] = None,
    ncols: int = 1,
    camera_scale: float = 0.3,
    pointcloud_max_points: int = 20000,
    pointcloud_marker_size: int = 1,
    **kwargs,
):
    """
    Main function to visualize Meshes, Cameras and Pointclouds.
    Plots input Pointclouds, Meshes, and Cameras data into named subplots,
    with named traces based on the dictionary keys. Cameras are
    rendered at the camera center location using a wireframe.

    Args:
        plots: A dict containing subplot and trace names,
            as well as the Meshes, Cameras and Pointclouds objects to be rendered.
            See below for examples of the format.
        viewpoint_cameras: an instance of a Cameras object providing a location
            to view the plotly plot from. If the batch size is equal
            to the number of subplots, it is a one to one mapping.
            If the batch size is 1, then that viewpoint will be used
            for all the subplots will be viewed from that point.
            Otherwise, the viewpoint_cameras will not be used.
        ncols: the number of subplots per row
        camera_scale: determines the size of the wireframe used to render cameras.
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

    If the Meshes, Pointclouds, or Cameras objects are batched, then every object in that batch
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

    To view plotly plots from a PyTorch3D camera's point of view, we can use
    viewpoint_cameras:
    ..code-block::python
        mesh = ... # batch size 2
        R, T = look_at_view_transform(2.7, 0, [0, 180]) # 2 camera angles, front and back
        # Any instance of CamerasBase works, here we use FoVPerspectiveCameras
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        fig = plot_scene({
            "subplot1_title": {
                "mesh_trace_title": mesh[0]
            },
            "subplot2_title": {
                "mesh_trace_title": mesh[1]
            }
        },
        viewpoint_cameras=cameras)
        fig.show()

    The above example will render the first subplot seen from the camera on the +z axis,
    and the second subplot from the viewpoint of the camera on the -z axis.

    We can visualize these cameras as well:
    ..code-block::python
        mesh = ...
        R, T = look_at_view_transform(2.7, 0, [0, 180]) # 2 camera angles, front and back
        # Any instance of CamerasBase works, here we use FoVPerspectiveCameras
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        fig = plot_scene({
            "subplot1_title": {
                "mesh_trace_title": mesh,
                "cameras_trace_title": cameras,
            },
        })
        fig.show()

    The above example will render one subplot with the mesh object
    and two cameras.

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
    viewpoints_eye_at_up_world = None
    if viewpoint_cameras:
        n_viewpoint_cameras = len(viewpoint_cameras)
        if n_viewpoint_cameras == len(subplots) or n_viewpoint_cameras == 1:
            # Calculate the vectors eye, at, up in world space
            # to initialize the position of the camera in
            # the plotly figure
            viewpoints_eye_at_up_world = camera_to_eye_at_up(
                viewpoint_cameras.get_world_to_view_transform().cpu()
            )
        else:
            msg = "Invalid number {} of viewpoint cameras were provided. Either 1 \
            or {} cameras are required".format(
                len(viewpoint_cameras), len(subplots)
            )
            warnings.warn(msg)

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
            elif isinstance(struct, CamerasBase):
                _add_camera_trace(
                    fig, struct, trace_name, subplot_idx, ncols, camera_scale
                )
            else:
                raise ValueError(
                    "struct {} is not a Cameras, Meshes or Pointclouds object".format(
                        struct
                    )
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

        # update camera viewpoint if provided
        if viewpoints_eye_at_up_world is not None:
            # Use camera params for batch index or the first camera if only one provided.
            viewpoint_idx = min(n_viewpoint_cameras - 1, subplot_idx)

            eye, at, up = (i[viewpoint_idx] for i in viewpoints_eye_at_up_world)
            eye_x, eye_y, eye_z = eye.tolist()
            at_x, at_y, at_z = at.tolist()
            up_x, up_y, up_z = up.tolist()

            # scale camera eye to plotly [-1, 1] ranges
            x_range = xaxis["range"]
            y_range = yaxis["range"]
            z_range = zaxis["range"]

            eye_x = _scale_camera_to_bounds(eye_x, x_range, True)
            eye_y = _scale_camera_to_bounds(eye_y, y_range, True)
            eye_z = _scale_camera_to_bounds(eye_z, z_range, True)

            at_x = _scale_camera_to_bounds(at_x, x_range, True)
            at_y = _scale_camera_to_bounds(at_y, y_range, True)
            at_z = _scale_camera_to_bounds(at_z, z_range, True)

            up_x = _scale_camera_to_bounds(up_x, x_range, False)
            up_y = _scale_camera_to_bounds(up_y, y_range, False)
            up_z = _scale_camera_to_bounds(up_z, z_range, False)

            camera["eye"] = {"x": eye_x, "y": eye_y, "z": eye_z}
            camera["center"] = {"x": at_x, "y": at_y, "z": at_z}
            camera["up"] = {"x": up_x, "y": up_y, "z": up_z}

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


def plot_batch_individually(
    batched_structs: Union[
        List[Union[Meshes, Pointclouds, CamerasBase]], Meshes, Pointclouds, CamerasBase
    ],
    *,
    viewpoint_cameras: Optional[CamerasBase] = None,
    ncols: int = 1,
    extend_struct: bool = True,
    subplot_titles: Optional[List[str]] = None,
    **kwargs,
):
    """
    This is a higher level plotting function than plot_scene, for plotting
    Cameras, Meshes and Pointclouds in simple cases. The simplest use is to plot a
    single Cameras, Meshes or Pointclouds object, where you just pass it in as a
    one element list. This will plot each batch element in a separate subplot.

    More generally, you can supply multiple Cameras, Meshes or Pointclouds
    having the same batch size `n`. In this case, there will be `n` subplots,
    each depicting the corresponding batch element of all the inputs.

    In addition, you can include Cameras, Meshes and Pointclouds of size 1 in
    the input. These will either be rendered in the first subplot
    (if extend_struct is False), or in every subplot.

    Args:
        batched_structs: a list of Cameras, Meshes and/or Pointclouds to be rendered.
            Each structure's corresponding batch element will be plotted in
            a single subplot, resulting in n subplots for a batch of size n.
            Every struct should either have the same batch size or be of batch size 1.
            See extend_struct and the description above for how batch size 1 structs
            are handled. Also accepts a single Cameras, Meshes or Pointclouds object,
            which will have each individual element plotted in its own subplot.
        viewpoint_cameras: an instance of a Cameras object providing a location
            to view the plotly plot from. If the batch size is equal
            to the number of subplots, it is a one to one mapping.
            If the batch size is 1, then that viewpoint will be used
            for all the subplots will be viewed from that point.
            Otherwise, the viewpoint_cameras will not be used.
        ncols: the number of subplots per row
        extend_struct: if True, indicates that structs of batch size 1
            should be plotted in every subplot.
        subplot_titles: strings to name each subplot
        **kwargs: keyword arguments which are passed to plot_scene.
            See plot_scene documentation for details.

    Example:

    ..code-block::python

        mesh = ...  # mesh of batch size 2
        point_cloud = ... # point_cloud of batch size 2
        fig = plot_batch_individually([mesh, point_cloud], subplot_titles=["plot1", "plot2"])
        fig.show()

        # this is equivalent to the below figure
        fig = plot_scene({
            "plot1": {
                "trace1-1": mesh[0],
                "trace1-2": point_cloud[0]
            },
            "plot2":{
                "trace2-1": mesh[1],
                "trace2-2": point_cloud[1]
            }
        })
        fig.show()

    The above example will render two subplots which each have both a mesh and pointcloud.
    For more examples look at the pytorch3d tutorials at `pytorch3d/docs/tutorials`,
    in particular the files rendered_color_points.ipynb and rendered_textured_meshes.ipynb.
    """

    # check that every batch is the same size or is size 1
    if len(batched_structs) == 0:
        msg = "No structs to plot"
        warnings.warn(msg)
        return
    max_size = 0
    if isinstance(batched_structs, list):
        max_size = max(len(s) for s in batched_structs)
        for struct in batched_structs:
            if len(struct) not in (1, max_size):
                msg = "invalid batch size {} provided: {}".format(len(struct), struct)
                raise ValueError(msg)
    else:
        max_size = len(batched_structs)

    if max_size == 0:
        msg = "No data is provided with at least one element"
        raise ValueError(msg)

    if subplot_titles:
        if len(subplot_titles) != max_size:
            msg = "invalid number of subplot titles"
            raise ValueError(msg)

    scene_dictionary = {}
    # construct the scene dictionary
    for scene_num in range(max_size):
        subplot_title = (
            subplot_titles[scene_num]
            if subplot_titles
            else "subplot " + str(scene_num + 1)
        )
        scene_dictionary[subplot_title] = {}

        if isinstance(batched_structs, list):
            for i, batched_struct in enumerate(batched_structs):
                # check for whether this struct needs to be extended
                if i >= len(batched_struct) and not extend_struct:
                    continue
                _add_struct_from_batch(
                    batched_struct, scene_num, subplot_title, scene_dictionary, i + 1
                )
        else:  # batched_structs is a single struct
            _add_struct_from_batch(
                batched_structs, scene_num, subplot_title, scene_dictionary
            )

    return plot_scene(
        scene_dictionary, viewpoint_cameras=viewpoint_cameras, ncols=ncols, **kwargs
    )


def _add_struct_from_batch(
    batched_struct: Union[CamerasBase, Meshes, Pointclouds],
    scene_num: int,
    subplot_title: str,
    scene_dictionary: Dict[str, Dict[str, Union[CamerasBase, Meshes, Pointclouds]]],
    trace_idx: int = 1,
):
    """
    Adds the struct corresponding to the given scene_num index to
    a provided scene_dictionary to be passed in to plot_scene

    Args:
        batched_struct: the batched data structure to add to the dict
        scene_num: the subplot from plot_batch_individually which this struct
            should be added to
        subplot_title: the title of the subplot
        scene_dictionary: the dictionary to add the indexed struct to
        trace_idx: the trace number, starting at 1 for this struct's trace
    """
    struct = None
    if isinstance(batched_struct, CamerasBase):
        # we can't index directly into camera batches
        R, T = batched_struct.R, batched_struct.T
        # pyre-fixme[6]: Expected `Sized` for 1st param but got `Union[torch.Tensor,
        #  torch.nn.Module]`.
        r_idx = min(scene_num, len(R) - 1)
        # pyre-fixme[6]: Expected `Sized` for 1st param but got `Union[torch.Tensor,
        #  torch.nn.Module]`.
        t_idx = min(scene_num, len(T) - 1)
        # pyre-fixme[29]:
        #  `Union[BoundMethod[typing.Callable(torch.Tensor.__getitem__)[[Named(self,
        #  torch.Tensor), Named(item, typing.Any)], typing.Any], torch.Tensor],
        #  torch.Tensor, torch.nn.Module]` is not a function.
        R = R[r_idx].unsqueeze(0)
        # pyre-fixme[29]:
        #  `Union[BoundMethod[typing.Callable(torch.Tensor.__getitem__)[[Named(self,
        #  torch.Tensor), Named(item, typing.Any)], typing.Any], torch.Tensor],
        #  torch.Tensor, torch.nn.Module]` is not a function.
        T = T[t_idx].unsqueeze(0)
        struct = CamerasBase(device=batched_struct.device, R=R, T=T)
    else:  # batched meshes and pointclouds are indexable
        struct_idx = min(scene_num, len(batched_struct) - 1)
        struct = batched_struct[struct_idx]
    trace_name = "trace{}-{}".format(scene_num + 1, trace_idx)
    scene_dictionary[subplot_title][trace_name] = struct


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
        ncols: the number of subplots per row.
        max_points_per_pointcloud: the number of points to render, which are randomly sampled.
        marker_size: the size of the rendered points
    """
    pointclouds = pointclouds.detach().cpu()
    verts = pointclouds.points_packed()
    features = pointclouds.features_packed()

    indices = None
    if pointclouds.num_points_per_cloud().max() > max_points_per_pointcloud:
        start_index = 0
        index_list = []
        for num_points in pointclouds.num_points_per_cloud():
            if num_points > max_points_per_pointcloud:
                indices_cloud = np.random.choice(
                    num_points, max_points_per_pointcloud, replace=False
                )
                index_list.append(start_index + indices_cloud)
            else:
                index_list.append(start_index + np.arange(num_points))
            start_index += num_points
        indices = np.concatenate(index_list)
        verts = verts[indices]

    color = None
    if features is not None:
        if indices is not None:
            # Only select features if we selected vertices above
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


def _add_camera_trace(
    fig: go.Figure,
    cameras: CamerasBase,
    trace_name: str,
    subplot_idx: int,
    ncols: int,
    camera_scale: float,
):
    """
    Adds a trace rendering a Cameras object to the passed in figure, with
    a given name and in a specific subplot.

    Args:
        fig: plotly figure to add the trace within.
        cameras: the Cameras object to render. It can be batched.
        trace_name: name to label the trace with.
        subplot_idx: identifies the subplot, with 0 being the top left.
        ncols: the number of subplots per row.
        camera_scale: the size of the wireframe used to render the Cameras object.
    """
    cam_wires = get_camera_wireframe(camera_scale).to(cameras.device)
    cam_trans = cameras.get_world_to_view_transform().inverse()
    cam_wires_trans = cam_trans.transform_points(cam_wires).detach().cpu()
    # if batch size is 1, unsqueeze to add dimension
    if len(cam_wires_trans.shape) < 3:
        cam_wires_trans = cam_wires_trans.unsqueeze(0)

    nan_tensor = torch.Tensor([[float("NaN")] * 3])
    all_cam_wires = cam_wires_trans[0]
    for wire in cam_wires_trans[1:]:
        # We combine camera points into a single tensor to plot them in a
        # single trace. The NaNs are inserted between sets of camera
        # points so that the lines drawn by Plotly are not drawn between
        # points that belong to different cameras.
        all_cam_wires = torch.cat((all_cam_wires, nan_tensor, wire))
    x, y, z = all_cam_wires.detach().cpu().numpy().T.astype(float)

    row, col = subplot_idx // ncols + 1, subplot_idx % ncols + 1
    fig.add_trace(
        go.Scatter3d(  # pyre-ignore [16]
            x=x, y=y, z=z, marker={"size": 1}, name=trace_name
        ),
        row=row,
        col=col,
    )

    # Access the current subplot's scene configuration
    plot_scene = "scene" + str(subplot_idx + 1)
    current_layout = fig["layout"][plot_scene]

    # flatten for bounds calculations
    flattened_wires = cam_wires_trans.flatten(0, 1)
    verts_center = flattened_wires.mean(0)
    max_expand = (flattened_wires.max(0)[0] - flattened_wires.min(0)[0]).max()
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
    fig_rows = batch_size // ncols
    if batch_size % ncols != 0:
        fig_rows += 1  # allow for non-uniform rows
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


def _scale_camera_to_bounds(
    coordinate: float, axis_bounds: Tuple[float, float], is_position: bool
):
    """
    We set our plotly plot's axes' bounding box to [-1,1]x[-1,1]x[-1,1]. As such,
    the plotly camera location has to be scaled accordingly to have its world coordinates
    correspond to its relative plotted coordinates for viewing the plotly plot.
    This function does the scaling and offset to transform the coordinates.

    Args:
        coordinate: the float value to be transformed
        axis_bounds: the bounds of the plotly plot for the axis which
            the coordinate argument refers to
        is_position: If true, the float value is the coordinate of a position, and so must
            be moved in to [-1,1]. Otherwise it is a component of a direction, and so needs only
            to be scaled.
    """
    scale = (axis_bounds[1] - axis_bounds[0]) / 2
    if not is_position:
        return coordinate / scale
    offset = (axis_bounds[1] / scale) - 1
    return coordinate / scale - offset
