# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import warnings
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from pytorch3d.renderer import (
    HeterogeneousRayBundle,
    ray_bundle_to_ray_points,
    RayBundle,
    TexturesAtlas,
    TexturesVertex,
)
from pytorch3d.renderer.camera_utils import camera_to_eye_at_up
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import join_meshes_as_scene, Meshes, Pointclouds


Struct = Union[CamerasBase, Meshes, Pointclouds, RayBundle, HeterogeneousRayBundle]


def _get_len(struct: Union[Struct, List[Struct]]) -> int:  # pragma: no cover
    """
    Returns the length (usually corresponds to the batch size) of the input structure.
    """
    # pyre-ignore[6]
    if not _is_ray_bundle(struct):
        # pyre-ignore[6]
        return len(struct)
    if _is_heterogeneous_ray_bundle(struct):
        # pyre-ignore[16]
        return len(struct.camera_counts)
    # pyre-ignore[16]
    return len(struct.directions)


def _is_ray_bundle(struct: Struct) -> bool:
    """
    Args:
        struct: Struct object to test
    Returns:
        True if something is a RayBundle, HeterogeneousRayBundle or
        ImplicitronRayBundle, else False
    """
    return hasattr(struct, "directions")


def _is_heterogeneous_ray_bundle(struct: Union[List[Struct], Struct]) -> bool:
    """
    Args:
        struct :object to test
    Returns:
        True if something is a HeterogeneousRayBundle or ImplicitronRayBundle
        and cant be reduced to RayBundle else False
    """
    # pyre-ignore[16]
    return hasattr(struct, "camera_counts") and struct.camera_counts is not None


def get_camera_wireframe(scale: float = 0.3):  # pragma: no cover
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


class AxisArgs(NamedTuple):  # pragma: no cover
    showgrid: bool = False
    zeroline: bool = False
    showline: bool = False
    ticks: str = ""
    showticklabels: bool = False
    backgroundcolor: str = "#fff"
    showaxeslabels: bool = False


class Lighting(NamedTuple):  # pragma: no cover
    ambient: float = 0.8
    diffuse: float = 1.0
    fresnel: float = 0.0
    specular: float = 0.0
    roughness: float = 0.5
    facenormalsepsilon: float = 1e-6
    vertexnormalsepsilon: float = 1e-12


@torch.no_grad()
def plot_scene(
    plots: Dict[str, Dict[str, Struct]],
    *,
    viewpoint_cameras: Optional[CamerasBase] = None,
    ncols: int = 1,
    camera_scale: float = 0.3,
    pointcloud_max_points: int = 20000,
    pointcloud_marker_size: int = 1,
    raybundle_max_rays: int = 20000,
    raybundle_max_points_per_ray: int = 1000,
    raybundle_ray_point_marker_size: int = 1,
    raybundle_ray_line_width: int = 1,
    **kwargs,
):  # pragma: no cover
    """
    Main function to visualize Cameras, Meshes, Pointclouds, and RayBundle.
    Plots input Cameras, Meshes, Pointclouds, and RayBundle data into named subplots,
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
        raybundle_max_rays: maximum number of rays of a RayBundle to visualize. Randomly
            subsamples without replacement in case the number of rays is bigger than max_rays.
        raybundle_max_points_per_ray: the maximum number of points per ray in RayBundle
            to visualize. If more are present, a random sample of size
            max_points_per_ray is used.
        raybundle_ray_point_marker_size: the size of the ray points of a plotted RayBundle
        raybundle_ray_line_width: the width of the plotted rays of a RayBundle
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

    RayBundle visualization is also supproted:
    ..code-block::python
        cameras = PerspectiveCameras(...)
        ray_bundle = RayBundle(origins=..., lengths=..., directions=..., xys=...)
        fig = plot_scene({
            "subplot1_title": {
                "ray_bundle_trace_title": ray_bundle,
                "cameras_trace_title": cameras,
            },
        })
        fig.show()

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
            "x": 0.0,
            "y": 1.0,
            "z": 0.0,
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
            or {} cameras are required".format(len(viewpoint_cameras), len(subplots))
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
            elif _is_ray_bundle(struct):
                _add_ray_bundle_trace(
                    fig,
                    struct,
                    trace_name,
                    subplot_idx,
                    ncols,
                    raybundle_max_rays,
                    raybundle_max_points_per_ray,
                    raybundle_ray_point_marker_size,
                    raybundle_ray_line_width,
                )
            else:
                raise ValueError(
                    "struct {} is not a Cameras, Meshes, Pointclouds,".format(struct)
                    + " , RayBundle or HeterogeneousRayBundle object."
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
            # pyre-fixme[61]: `n_viewpoint_cameras` is undefined, or not always defined.
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


@torch.no_grad()
def plot_batch_individually(
    batched_structs: Union[
        List[Struct],
        Struct,
    ],
    *,
    viewpoint_cameras: Optional[CamerasBase] = None,
    ncols: int = 1,
    extend_struct: bool = True,
    subplot_titles: Optional[List[str]] = None,
    **kwargs,
):  # pragma: no cover
    """
    This is a higher level plotting function than plot_scene, for plotting
    Cameras, Meshes, Pointclouds, and RayBundle in simple cases. The simplest use
    is to plot a single Cameras, Meshes, Pointclouds, or a RayBundle object,
    where you just pass it in as a one element list. This will plot each batch
    element in a separate subplot.

    More generally, you can supply multiple Cameras, Meshes, Pointclouds, or RayBundle
    having the same batch size `n`. In this case, there will be `n` subplots,
    each depicting the corresponding batch element of all the inputs.

    In addition, you can include Cameras, Meshes, Pointclouds, or RayBundle of size 1 in
    the input. These will either be rendered in the first subplot
    (if extend_struct is False), or in every subplot.
    RayBundle includes ImplicitronRayBundle and HeterogeneousRaybundle.

    Args:
        batched_structs: a list of Cameras, Meshes, Pointclouds and RayBundle to be
            rendered. Each structure's corresponding batch element will be plotted in a
            single subplot, resulting in n subplots for a batch of size n. Every struct
            should either have the same batch size or be of batch size 1. See extend_struct
            and the description above for how batch size 1 structs are handled. Also accepts
            a single Cameras, Meshes, Pointclouds, and RayBundle object, which will have
            each individual element plotted in its own subplot.
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
    if _get_len(batched_structs) == 0:
        msg = "No structs to plot"
        warnings.warn(msg)
        return
    max_size = 0
    if isinstance(batched_structs, list):
        max_size = max(_get_len(s) for s in batched_structs)
        for struct in batched_structs:
            struct_len = _get_len(struct)
            if struct_len not in (1, max_size):
                msg = "invalid batch size {} provided: {}".format(struct_len, struct)
                raise ValueError(msg)
    else:
        max_size = _get_len(batched_structs)

    if max_size == 0:
        msg = "No data is provided with at least one element"
        raise ValueError(msg)

    if subplot_titles:
        if len(subplot_titles) != max_size:
            msg = "invalid number of subplot titles"
            raise ValueError(msg)

    # if we are dealing with HeterogeneousRayBundle of ImplicitronRayBundle create
    # first indexes for faster
    first_idxs = None
    if _is_heterogeneous_ray_bundle(batched_structs):
        # pyre-ignore[16]
        cumsum = batched_structs.camera_counts.cumsum(dim=0)
        first_idxs = torch.cat((cumsum.new_zeros((1,)), cumsum))

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
                first_idxs = None
                if _is_heterogeneous_ray_bundle(batched_structs[i]):
                    # pyre-ignore[16]
                    cumsum = batched_struct.camera_counts.cumsum(dim=0)
                    first_idxs = torch.cat((cumsum.new_zeros((1,)), cumsum))
                # check for whether this struct needs to be extended
                batched_struct_len = _get_len(batched_struct)
                if i >= batched_struct_len and not extend_struct:
                    continue
                _add_struct_from_batch(
                    batched_struct,
                    scene_num,
                    subplot_title,
                    scene_dictionary,
                    i + 1,
                    first_idxs=first_idxs,
                )
        else:  # batched_structs is a single struct
            _add_struct_from_batch(
                batched_structs,
                scene_num,
                subplot_title,
                scene_dictionary,
                first_idxs=first_idxs,
            )

    return plot_scene(
        scene_dictionary, viewpoint_cameras=viewpoint_cameras, ncols=ncols, **kwargs
    )


def _add_struct_from_batch(
    batched_struct: Struct,
    scene_num: int,
    subplot_title: str,
    scene_dictionary: Dict[str, Dict[str, Struct]],
    trace_idx: int = 1,
    first_idxs: Optional[torch.Tensor] = None,
) -> None:  # pragma: no cover
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
        # pyre-fixme[6]: For 1st argument expected
        #  `pyre_extensions.PyreReadOnly[Sized]` but got `Union[Tensor, Module]`.
        r_idx = min(scene_num, len(R) - 1)
        # pyre-fixme[6]: For 1st argument expected
        #  `pyre_extensions.PyreReadOnly[Sized]` but got `Union[Tensor, Module]`.
        t_idx = min(scene_num, len(T) - 1)
        # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, slice[Any, A...
        R = R[r_idx].unsqueeze(0)
        # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, slice[Any, A...
        T = T[t_idx].unsqueeze(0)
        struct = CamerasBase(device=batched_struct.device, R=R, T=T)
    elif _is_ray_bundle(batched_struct) and not _is_heterogeneous_ray_bundle(
        batched_struct
    ):
        # for RayBundle we treat the camera count as the batch index
        struct_idx = min(scene_num, _get_len(batched_struct) - 1)

        struct = RayBundle(
            **{
                attr: getattr(batched_struct, attr)[struct_idx]
                for attr in ["origins", "directions", "lengths", "xys"]
            }
        )
    elif _is_heterogeneous_ray_bundle(batched_struct):
        # for RayBundle we treat the camera count as the batch index
        struct_idx = min(scene_num, _get_len(batched_struct) - 1)

        struct = RayBundle(
            **{
                attr: getattr(batched_struct, attr)[
                    # pyre-ignore[16]
                    first_idxs[struct_idx] : first_idxs[struct_idx + 1]
                ]
                for attr in ["origins", "directions", "lengths", "xys"]
            }
        )

    else:  # batched meshes and pointclouds are indexable
        struct_idx = min(scene_num, _get_len(batched_struct) - 1)
        # pyre-ignore[16]
        struct = batched_struct[struct_idx]
    trace_name = "trace{}-{}".format(scene_num + 1, trace_idx)
    scene_dictionary[subplot_title][trace_name] = struct


def _add_mesh_trace(
    fig: go.Figure,
    meshes: Meshes,
    trace_name: str,
    subplot_idx: int,
    ncols: int,
    lighting: Lighting,
) -> None:  # pragma: no cover
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
    # If mesh has vertex colors or face colors, use them
    # for figure, otherwise use plotly's default colors.
    verts_rgb = None
    faces_rgb = None
    if isinstance(mesh.textures, TexturesVertex):
        verts_rgb = mesh.textures.verts_features_packed()
        verts_rgb.clamp_(min=0.0, max=1.0)
        verts_rgb = torch.tensor(255.0) * verts_rgb
    if isinstance(mesh.textures, TexturesAtlas):
        atlas = mesh.textures.atlas_packed()
        # If K==1
        if atlas.shape[1] == 1 and atlas.shape[3] == 3:
            faces_rgb = atlas[:, 0, 0]

    # Reposition the unused vertices to be "inside" the object
    # (i.e. they won't be visible in the plot).
    verts_used = torch.zeros((verts.shape[0],), dtype=torch.bool)
    verts_used[torch.unique(faces)] = True
    verts_center = verts[verts_used].mean(0)
    verts[~verts_used] = verts_center

    row, col = subplot_idx // ncols + 1, subplot_idx % ncols + 1
    fig.add_trace(
        go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            vertexcolor=verts_rgb,
            facecolor=faces_rgb,
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
) -> None:  # pragma: no cover
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
    pointclouds = pointclouds.detach().cpu().subsample(max_points_per_pointcloud)
    verts = pointclouds.points_packed()
    features = pointclouds.features_packed()

    color = None
    if features is not None:
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
        go.Scatter3d(
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
) -> None:  # pragma: no cover
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
        go.Scatter3d(x=x, y=y, z=z, marker={"size": 1}, name=trace_name),
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


def _add_ray_bundle_trace(
    fig: go.Figure,
    ray_bundle: Union[RayBundle, HeterogeneousRayBundle],
    trace_name: str,
    subplot_idx: int,
    ncols: int,
    max_rays: int,
    max_points_per_ray: int,
    marker_size: int,
    line_width: int,
) -> None:  # pragma: no cover
    """
    Adds a trace rendering a ray bundle object
    to the passed in figure, with a given name and in a specific subplot.

    Args:
        fig: plotly figure to add the trace within.
        ray_bundle: the RayBundle, ImplicitronRayBundle or HeterogeneousRaybundle to render.
            It can be batched.
        trace_name: name to label the trace with.
        subplot_idx: identifies the subplot, with 0 being the top left.
        ncols: the number of subplots per row.
        max_rays: maximum number of plotted rays in total. Randomly subsamples
            without replacement in case the number of rays is bigger than max_rays.
        max_points_per_ray: maximum number of points plotted per ray.
        marker_size: the size of the ray point markers.
        line_width: the width of the ray lines.
    """

    n_pts_per_ray = ray_bundle.lengths.shape[-1]
    n_rays = ray_bundle.lengths.shape[:-1].numel()

    # flatten all batches of rays into a single big bundle
    ray_bundle_flat = RayBundle(
        **{
            attr: torch.flatten(getattr(ray_bundle, attr), start_dim=0, end_dim=-2)
            for attr in ["origins", "directions", "lengths", "xys"]
        }
    )

    # subsample the rays (if needed)
    if n_rays > max_rays:
        indices_rays = torch.randperm(n_rays)[:max_rays]
        ray_bundle_flat = RayBundle(
            **{
                attr: getattr(ray_bundle_flat, attr)[indices_rays]
                for attr in ["origins", "directions", "lengths", "xys"]
            }
        )

    # make ray line endpoints
    min_max_ray_depth = torch.stack(
        [
            ray_bundle_flat.lengths.min(dim=1).values,
            ray_bundle_flat.lengths.max(dim=1).values,
        ],
        dim=-1,
    )
    ray_lines_endpoints = ray_bundle_to_ray_points(
        ray_bundle_flat._replace(lengths=min_max_ray_depth)
    )

    # make the ray lines for plotly plotting
    nan_tensor = torch.tensor(
        [[float("NaN")] * 3],
        device=ray_lines_endpoints.device,
        dtype=ray_lines_endpoints.dtype,
    )
    ray_lines = torch.empty(size=(1, 3), device=ray_lines_endpoints.device)
    for ray_line in ray_lines_endpoints:
        # We combine the ray lines into a single tensor to plot them in a
        # single trace. The NaNs are inserted between sets of ray lines
        # so that the lines drawn by Plotly are not drawn between
        # lines that belong to different rays.
        ray_lines = torch.cat((ray_lines, nan_tensor, ray_line))
    x, y, z = ray_lines.detach().cpu().numpy().T.astype(float)
    row, col = subplot_idx // ncols + 1, subplot_idx % ncols + 1
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            marker={"size": 0.1},
            line={"width": line_width},
            name=trace_name,
        ),
        row=row,
        col=col,
    )

    # subsample the ray points (if needed)
    if n_pts_per_ray > max_points_per_ray:
        indices_ray_pts = torch.cat(
            [
                torch.randperm(n_pts_per_ray)[:max_points_per_ray] + ri * n_pts_per_ray
                for ri in range(ray_bundle_flat.lengths.shape[0])
            ]
        )
        ray_bundle_flat = ray_bundle_flat._replace(
            lengths=ray_bundle_flat.lengths.reshape(-1)[indices_ray_pts].reshape(
                ray_bundle_flat.lengths.shape[0], -1
            )
        )

    # plot the ray points
    ray_points = (
        ray_bundle_to_ray_points(ray_bundle_flat)
        .view(-1, 3)
        .detach()
        .cpu()
        .numpy()
        .astype(float)
    )
    fig.add_trace(
        go.Scatter3d(
            x=ray_points[:, 0],
            y=ray_points[:, 1],
            z=ray_points[:, 2],
            mode="markers",
            name=trace_name + "_points",
            marker={"size": marker_size},
        ),
        row=row,
        col=col,
    )

    # Access the current subplot's scene configuration
    plot_scene = "scene" + str(subplot_idx + 1)
    current_layout = fig["layout"][plot_scene]

    # update the bounds of the axes for the current trace
    all_ray_points = ray_bundle_to_ray_points(ray_bundle).reshape(-1, 3)
    ray_points_center = all_ray_points.mean(dim=0)
    max_expand = (all_ray_points.max(0)[0] - all_ray_points.min(0)[0]).max().item()
    _update_axes_bounds(ray_points_center, float(max_expand), current_layout)


def _gen_fig_with_subplots(
    batch_size: int, ncols: int, subplot_titles: List[str]
):  # pragma: no cover
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
    current_layout: go.Scene,
) -> None:  # pragma: no cover
    """
    Takes in the vertices' center point and max spread, and the current plotly figure
    layout and updates the layout to have bounds that include all traces for that subplot.
    Args:
        verts_center: tensor of size (3) corresponding to a trace's vertices' center point.
        max_expand: the maximum spread in any dimension of the trace's vertices.
        current_layout: the plotly figure layout scene corresponding to the referenced trace.
    """
    verts_center = verts_center.detach().cpu()
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
) -> float:  # pragma: no cover
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
