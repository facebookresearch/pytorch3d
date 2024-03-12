# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import math
import warnings
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from ...camera_conversions import _pulsar_from_cameras_projection
from ...cameras import (
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    OrthographicCameras,
    PerspectiveCameras,
)
from ..compositor import AlphaCompositor, NormWeightedCompositor
from ..rasterizer import PointsRasterizer
from .renderer import Renderer as PulsarRenderer


def _ensure_float_tensor(val_in, device):
    """Make sure that the value provided is wrapped a PyTorch float tensor."""
    if not isinstance(val_in, torch.Tensor):
        val_out = torch.tensor(val_in, dtype=torch.float32, device=device).reshape((1,))
    else:
        val_out = val_in.to(torch.float32).to(device).reshape((1,))
    return val_out


class PulsarPointsRenderer(nn.Module):
    """
    This renderer is a PyTorch3D interface wrapper around the pulsar renderer.

    It provides an interface consistent with PyTorch3D Pointcloud rendering.
    It will extract all necessary information from the rasterizer and compositor
    objects and convert them to the pulsar required format, then invoke rendering
    in the pulsar renderer. All gradients are handled appropriately through the
    wrapper and the wrapper should provide equivalent results to using the pulsar
    renderer directly.
    """

    def __init__(
        self,
        rasterizer: PointsRasterizer,
        compositor: Optional[Union[NormWeightedCompositor, AlphaCompositor]] = None,
        n_channels: int = 3,
        max_num_spheres: int = int(1e6),  # noqa: B008
        **kwargs,
    ) -> None:
        """
        rasterizer (PointsRasterizer): An object encapsulating rasterization parameters.
        compositor (ignored): Only keeping this for interface consistency. Default: None.
        n_channels (int): The number of channels of the resulting image. Default: 3.
        max_num_spheres (int): The maximum number of spheres intended to render with
            this renderer. Default: 1e6.
        kwargs (Any): kwargs to pass on to the pulsar renderer.
            See `pytorch3d.renderer.points.pulsar.renderer.Renderer` for all options.
        """
        super().__init__()
        self.rasterizer = rasterizer
        if compositor is not None:
            warnings.warn(
                "Creating a `PulsarPointsRenderer` with a compositor object! "
                "This object is ignored and just allowed as an argument for interface "
                "compatibility."
            )
        # Initialize the pulsar renderers.
        if not isinstance(
            rasterizer.cameras,
            (
                FoVOrthographicCameras,
                FoVPerspectiveCameras,
                PerspectiveCameras,
                OrthographicCameras,
            ),
        ):
            raise ValueError(
                "Only FoVPerspectiveCameras, PerspectiveCameras, "
                "FoVOrthographicCameras and OrthographicCameras are supported "
                "by the pulsar backend."
            )
        if isinstance(rasterizer.raster_settings.image_size, tuple):
            height, width = rasterizer.raster_settings.image_size
        else:
            width = rasterizer.raster_settings.image_size
            height = rasterizer.raster_settings.image_size
        # Making sure about integer types.
        width = int(width)
        height = int(height)
        max_num_spheres = int(max_num_spheres)
        orthogonal_projection = isinstance(
            rasterizer.cameras, (FoVOrthographicCameras, OrthographicCameras)
        )
        n_channels = int(n_channels)
        self.renderer = PulsarRenderer(
            width=width,
            height=height,
            max_num_balls=max_num_spheres,
            orthogonal_projection=orthogonal_projection,
            right_handed_system=False,
            n_channels=n_channels,
            **kwargs,
        )

    def _conf_check(self, point_clouds, kwargs: Dict[str, Any]) -> bool:
        """
        Verify internal configuration state with kwargs and pointclouds.

        This method will raise ValueError's for any inconsistencies found. It
        returns whether an orthogonal projection will be used.
        """
        if "gamma" not in kwargs.keys():
            raise ValueError(
                "gamma is a required keyword argument for the PulsarPointsRenderer!"
            )
        if (
            len(point_clouds) != len(self.rasterizer.cameras)
            and len(self.rasterizer.cameras) != 1
        ):
            raise ValueError(
                (
                    "The len(point_clouds) must either be equal to len(rasterizer.cameras) or "
                    "only one camera must be used. len(point_clouds): %d, "
                    "len(rasterizer.cameras): %d."
                )
                % (
                    len(point_clouds),
                    len(self.rasterizer.cameras),
                )
            )
        # Make sure the rasterizer and cameras objects have no
        # changes that can't be matched.
        orthogonal_projection = isinstance(
            self.rasterizer.cameras, (FoVOrthographicCameras, OrthographicCameras)
        )
        if orthogonal_projection != self.renderer._renderer.orthogonal:
            raise ValueError(
                "The camera type can not be changed after renderer initialization! "
                "Current camera orthogonal: %r. Original orthogonal: %r."
            ) % (orthogonal_projection, self.renderer._renderer.orthogonal)
        image_size = self.rasterizer.raster_settings.image_size
        if isinstance(image_size, tuple):
            expected_height, expected_width = image_size
        else:
            expected_height = expected_width = image_size
        if expected_width != self.renderer._renderer.width:
            raise ValueError(
                (
                    "The rasterizer width can not be changed after renderer "
                    "initialization! Current width: %s. Original width: %d."
                )
                % (
                    expected_width,
                    self.renderer._renderer.width,
                )
            )
        if expected_height != self.renderer._renderer.height:
            raise ValueError(
                (
                    "The rasterizer height can not be changed after renderer "
                    "initialization! Current height: %s. Original height: %d."
                )
                % (
                    expected_height,
                    self.renderer._renderer.height,
                )
            )
        return orthogonal_projection

    def _extract_intrinsics(  # noqa: C901
        self, orthogonal_projection, kwargs, cloud_idx, device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
        """
        Translate the camera intrinsics from PyTorch3D format to pulsar format.
        """
        # Shorthand:
        cameras = self.rasterizer.cameras
        if orthogonal_projection:
            focal_length = torch.zeros((1,), dtype=torch.float32)
            if isinstance(cameras, FoVOrthographicCameras):
                znear = kwargs.get("znear", cameras.znear)[cloud_idx]
                zfar = kwargs.get("zfar", cameras.zfar)[cloud_idx]
                max_y = kwargs.get("max_y", cameras.max_y)[cloud_idx]
                min_y = kwargs.get("min_y", cameras.min_y)[cloud_idx]
                max_x = kwargs.get("max_x", cameras.max_x)[cloud_idx]
                min_x = kwargs.get("min_x", cameras.min_x)[cloud_idx]
                if max_y != -min_y:
                    raise ValueError(
                        "The orthographic camera must be centered around 0. "
                        f"Max is {max_y} and min is {min_y}."
                    )
                if max_x != -min_x:
                    raise ValueError(
                        "The orthographic camera must be centered around 0. "
                        f"Max is {max_x} and min is {min_x}."
                    )
                if not torch.all(
                    kwargs.get("scale_xyz", cameras.scale_xyz)[cloud_idx] == 1.0
                ):
                    raise ValueError(
                        "The orthographic camera scale must be ((1.0, 1.0, 1.0),). "
                        f"{kwargs.get('scale_xyz', cameras.scale_xyz)[cloud_idx]}."
                    )
                sensor_width = max_x - min_x
                if not sensor_width > 0.0:
                    raise ValueError(
                        f"The orthographic camera must have positive size! Is: {sensor_width}."  # noqa: B950
                    )
                principal_point_x, principal_point_y = (
                    torch.zeros((1,), dtype=torch.float32),
                    torch.zeros((1,), dtype=torch.float32),
                )
            else:
                # Currently, this means it must be an 'OrthographicCameras' object.
                focal_length_conf = kwargs.get("focal_length", cameras.focal_length)[
                    cloud_idx
                ]
                if (
                    focal_length_conf.numel() == 2
                    and focal_length_conf[0] * self.renderer._renderer.width
                    - focal_length_conf[1] * self.renderer._renderer.height
                    > 1e-5
                ):
                    raise ValueError(
                        "Pulsar only supports a single focal length! "
                        "Provided: %s." % (str(focal_length_conf))
                    )
                if focal_length_conf.numel() == 2:
                    sensor_width = 2.0 / focal_length_conf[0]
                else:
                    if focal_length_conf.numel() != 1:
                        raise ValueError(
                            "Focal length not parsable: %s." % (str(focal_length_conf))
                        )
                    sensor_width = 2.0 / focal_length_conf
                if "znear" not in kwargs.keys() or "zfar" not in kwargs.keys():
                    raise ValueError(
                        "pulsar needs znear and zfar values for "
                        "the OrthographicCameras. Please provide them as keyword "
                        "argument to the forward method."
                    )
                znear = kwargs["znear"][cloud_idx]
                zfar = kwargs["zfar"][cloud_idx]
                principal_point_x = (
                    kwargs.get("principal_point", cameras.principal_point)[cloud_idx][0]
                    * 0.5
                    * self.renderer._renderer.width
                )
                principal_point_y = (
                    kwargs.get("principal_point", cameras.principal_point)[cloud_idx][1]
                    * 0.5
                    * self.renderer._renderer.height
                )
        else:
            if not isinstance(cameras, PerspectiveCameras):
                # Create a virtual focal length that is closer than znear.
                znear = kwargs.get("znear", cameras.znear)[cloud_idx]
                zfar = kwargs.get("zfar", cameras.zfar)[cloud_idx]
                focal_length = znear - 1e-6
                # Create a sensor size that matches the expected fov assuming this f.
                afov = kwargs.get("fov", cameras.fov)[cloud_idx]
                if kwargs.get("degrees", cameras.degrees):
                    afov *= math.pi / 180.0
                sensor_width = math.tan(afov / 2.0) * 2.0 * focal_length
                if not (
                    kwargs.get("aspect_ratio", cameras.aspect_ratio)[cloud_idx]
                    - self.renderer._renderer.width / self.renderer._renderer.height
                    < 1e-6
                ):
                    raise ValueError(
                        "The aspect ratio ("
                        f"{kwargs.get('aspect_ratio', cameras.aspect_ratio)[cloud_idx]}) "
                        "must agree with the resolution width / height ("
                        f"{self.renderer._renderer.width / self.renderer._renderer.height})."  # noqa: B950
                    )
                principal_point_x, principal_point_y = (
                    torch.zeros((1,), dtype=torch.float32),
                    torch.zeros((1,), dtype=torch.float32),
                )
            else:
                focal_length_conf = kwargs.get("focal_length", cameras.focal_length)[
                    cloud_idx
                ]
                if (
                    focal_length_conf.numel() == 2
                    and focal_length_conf[0] * self.renderer._renderer.width
                    - focal_length_conf[1] * self.renderer._renderer.height
                    > 1e-5
                ):
                    raise ValueError(
                        "Pulsar only supports a single focal length! "
                        "Provided: %s." % (str(focal_length_conf))
                    )
                if "znear" not in kwargs.keys() or "zfar" not in kwargs.keys():
                    raise ValueError(
                        "pulsar needs znear and zfar values for "
                        "the PerspectiveCameras. Please provide them as keyword "
                        "argument to the forward method."
                    )
                znear = kwargs["znear"][cloud_idx]
                zfar = kwargs["zfar"][cloud_idx]
                if focal_length_conf.numel() == 2:
                    focal_length_px = focal_length_conf[0]
                else:
                    if focal_length_conf.numel() != 1:
                        raise ValueError(
                            "Focal length not parsable: %s." % (str(focal_length_conf))
                        )
                    focal_length_px = focal_length_conf
                focal_length = torch.tensor(
                    [
                        znear - 1e-6,
                    ],
                    dtype=torch.float32,
                    device=focal_length_px.device,
                )
                sensor_width = focal_length / focal_length_px * 2.0
                principal_point_x = (
                    kwargs.get("principal_point", cameras.principal_point)[cloud_idx][0]
                    * 0.5
                    * self.renderer._renderer.width
                )
                principal_point_y = (
                    kwargs.get("principal_point", cameras.principal_point)[cloud_idx][1]
                    * 0.5
                    * self.renderer._renderer.height
                )
        focal_length = _ensure_float_tensor(focal_length, device)
        sensor_width = _ensure_float_tensor(sensor_width, device)
        principal_point_x = _ensure_float_tensor(principal_point_x, device)
        principal_point_y = _ensure_float_tensor(principal_point_y, device)
        znear = _ensure_float_tensor(znear, device)
        zfar = _ensure_float_tensor(zfar, device)
        return (
            focal_length,
            sensor_width,
            principal_point_x,
            principal_point_y,
            znear,
            zfar,
        )

    def _extract_extrinsics(
        self, kwargs, cloud_idx
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract the extrinsic information from the kwargs for a specific point cloud.

        Instead of implementing a direct translation from the PyTorch3D to the Pulsar
        camera model, we chain the two conversions of PyTorch3D->OpenCV and
        OpenCV->Pulsar for better maintainability (PyTorch3D->OpenCV is maintained and
        tested by the core PyTorch3D team, whereas OpenCV->Pulsar is maintained and
        tested by the Pulsar team).
        """
        # Shorthand:
        cameras = self.rasterizer.cameras
        R = kwargs.get("R", cameras.R)[cloud_idx]
        T = kwargs.get("T", cameras.T)[cloud_idx]
        tmp_cams = PerspectiveCameras(
            R=R.unsqueeze(0), T=T.unsqueeze(0), device=R.device
        )
        size_tensor = torch.tensor(
            [[self.renderer._renderer.height, self.renderer._renderer.width]]
        )
        pulsar_cam = _pulsar_from_cameras_projection(tmp_cams, size_tensor)
        cam_pos = pulsar_cam[0, :3]
        cam_rot = pulsar_cam[0, 3:9]
        return cam_pos, cam_rot

    def _get_vert_rad(
        self, vert_pos, cam_pos, orthogonal_projection, focal_length, kwargs, cloud_idx
    ) -> torch.Tensor:
        """
        Get point radiuses.

        These can be depending on the camera position in case of a perspective
        transform.
        """
        # Normalize point radiuses.
        # `self.rasterizer.raster_settings.radius` can either be a float
        # or itself a tensor.
        raster_rad = self.rasterizer.raster_settings.radius
        if kwargs.get("radius_world", False):
            return raster_rad
        if (
            isinstance(raster_rad, torch.Tensor)
            and raster_rad.numel() > 1
            and raster_rad.ndim > 1
        ):
            # In this case it must be a batched torch tensor.
            raster_rad = raster_rad[cloud_idx]
        if orthogonal_projection:
            vert_rad = (
                torch.ones(
                    (vert_pos.shape[0],), dtype=torch.float32, device=vert_pos.device
                )
                * raster_rad
            )
        else:
            point_dists = torch.norm((vert_pos - cam_pos), p=2, dim=1, keepdim=False)
            vert_rad = raster_rad / focal_length.to(vert_pos.device) * point_dists
            if isinstance(self.rasterizer.cameras, PerspectiveCameras):
                # NDC normalization happens through adjusted focal length.
                pass
            else:
                vert_rad = vert_rad / 2.0  # NDC normalization.
        return vert_rad

    # point_clouds is not typed to avoid a cyclic dependency.
    def forward(self, point_clouds, **kwargs) -> torch.Tensor:
        """
        Get the rendering of the provided `Pointclouds`.

        The number of point clouds in the `Pointclouds` object determines the
        number of resulting images. The provided cameras can be either 1 or equal
        to the number of pointclouds (in the first case, the same camera will be
        used for all clouds, in the latter case each point cloud will be rendered
        with the corresponding camera).

        The following kwargs are support from PyTorch3D (depending on the selected
        camera model potentially overriding camera parameters):
            radius_world (bool): use the provided radiuses from the raster_settings
              plain as radiuses in world space. Default: False.
            znear (Iterable[float]): near geometry cutoff. Is required for
              OrthographicCameras and PerspectiveCameras.
            zfar (Iterable[float]): far geometry cutoff. Is required for
              OrthographicCameras and PerspectiveCameras.
            R (torch.Tensor): [Bx3x3] camera rotation matrices.
            T (torch.Tensor): [Bx3] camera translation vectors.
            principal_point (torch.Tensor): [Bx2] camera intrinsic principal
              point offset vectors.
            focal_length (torch.Tensor): [Bx1] camera intrinsic focal lengths.
            aspect_ratio (Iterable[float]): camera aspect ratios.
            fov (Iterable[float]): camera FOVs.
            degrees (bool): whether FOVs are specified in degrees or
              radians.
            min_x (Iterable[float]): minimum x for the FoVOrthographicCameras.
            max_x (Iterable[float]): maximum x for the FoVOrthographicCameras.
            min_y (Iterable[float]): minimum y for the FoVOrthographicCameras.
            max_y (Iterable[float]): maximum y for the FoVOrthographicCameras.

        The following kwargs are supported from pulsar:
            gamma (float): The gamma value to use. This defines the transparency for
                differentiability (see pulsar paper for details). Must be in [1., 1e-5]
                with 1.0 being mostly transparent. This keyword argument is *required*!
            bg_col (torch.Tensor): The background color. Must be a tensor on the same
                device as the point clouds, with as many channels as features (no batch
                dimension - it is the same for all images in the batch).
                Default: 0.0 for all channels.
            percent_allowed_difference (float): a value in [0., 1.[ with the maximum
                allowed difference in channel space. This is used to speed up the
                computation. Default: 0.01.
            max_n_hits (int): a hard limit on the number of sphere hits per ray.
                Default: max int.
            mode (int): render mode in {0, 1}. 0: render image; 1: render hit map.
        """
        orthogonal_projection: bool = self._conf_check(point_clouds, kwargs)
        # Get access to inputs. We're using the list accessor and process
        # them sequentially.
        position_list = point_clouds.points_list()
        features_list = point_clouds.features_list()
        # Result list.
        images = []
        for cloud_idx, (vert_pos, vert_col) in enumerate(
            zip(position_list, features_list)
        ):
            # Get extrinsics.
            cam_pos, cam_rot = self._extract_extrinsics(kwargs, cloud_idx)
            # Get intrinsics.
            (
                focal_length,
                sensor_width,
                principal_point_x,
                principal_point_y,
                znear,
                zfar,
            ) = self._extract_intrinsics(
                orthogonal_projection, kwargs, cloud_idx, cam_pos.device
            )
            # Put everything together.
            cam_params = torch.cat(
                (
                    cam_pos,
                    cam_rot.to(cam_pos.device),
                    torch.cat(
                        [
                            focal_length,
                            sensor_width,
                            principal_point_x,
                            principal_point_y,
                        ],
                    ),
                )
            )
            # Get point radiuses (can depend on camera position).
            vert_rad = self._get_vert_rad(
                vert_pos,
                cam_pos,
                orthogonal_projection,
                focal_length,
                kwargs,
                cloud_idx,
            )
            # Clean kwargs for passing on.
            gamma = kwargs["gamma"][cloud_idx]
            if "first_R_then_T" in kwargs.keys():
                raise ValueError("`first_R_then_T` is not supported in this interface.")
            otherargs = {
                argn: argv
                for argn, argv in kwargs.items()
                if argn
                not in [
                    "radius_world",
                    "gamma",
                    "znear",
                    "zfar",
                    "R",
                    "T",
                    "principal_point",
                    "focal_length",
                    "aspect_ratio",
                    "fov",
                    "degrees",
                    "min_x",
                    "max_x",
                    "min_y",
                    "max_y",
                ]
            }
            # background color
            if "bg_col" not in otherargs:
                bg_col = torch.zeros(
                    vert_col.shape[1], device=cam_params.device, dtype=torch.float32
                )
                otherargs["bg_col"] = bg_col
            # Go!
            images.append(
                self.renderer(
                    vert_pos=vert_pos,
                    vert_col=vert_col,
                    vert_rad=vert_rad,
                    cam_params=cam_params,
                    gamma=gamma,
                    max_depth=zfar,
                    min_depth=znear,
                    **otherargs,
                ).flip(dims=[0])
            )
        return torch.stack(images, dim=0)
