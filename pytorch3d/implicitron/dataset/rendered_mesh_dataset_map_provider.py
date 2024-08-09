# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from os.path import dirname, join, realpath
from typing import Optional, Tuple

import torch
from pytorch3d.implicitron.tools.config import registry, run_auto_creation
from pytorch3d.io import IO
from pytorch3d.renderer import (
    AmbientLights,
    BlendParams,
    CamerasBase,
    FoVPerspectiveCameras,
    HardPhongShader,
    look_at_view_transform,
    MeshRasterizer,
    MeshRendererWithFragments,
    PointLights,
    RasterizationSettings,
)
from pytorch3d.structures.meshes import Meshes

from .dataset_map_provider import DatasetMap, DatasetMapProviderBase, PathManagerFactory
from .single_sequence_dataset import SingleSceneDataset
from .utils import DATASET_TYPE_KNOWN


@registry.register
class RenderedMeshDatasetMapProvider(DatasetMapProviderBase):
    """
    A simple single-scene dataset based on PyTorch3D renders of a mesh.
    Provides `num_views` renders of the mesh as train, with no val
    and test. The renders are generated from viewpoints sampled at uniformly
    distributed azimuth intervals. The elevation is kept constant so that the
    camera's vertical position coincides with the equator.

    By default, uses Keenan Crane's cow model, and the camera locations are
    set to make sense for that.

    Although the rendering used to generate this dataset will use a GPU
    if one is available, the data it produces is on the CPU just like
    the data returned by implicitron's other dataset map providers.
    This is because both datasets and models can be large, so implicitron's
    training loop expects data on the CPU and only moves
    what it needs to the device.

    For a more detailed explanation of this code, please refer to the
    docs/tutorials/fit_textured_mesh.ipynb notebook.

    Members:
        num_views: The number of generated renders.
        data_file: The folder that contains the mesh file. By default, finds
            the cow mesh in the same repo as this code.
        azimuth_range: number of degrees on each side of the start position to
            take samples
        distance: distance from camera centres to the origin.
        resolution: the common height and width of the output images.
        use_point_light: whether to use a particular point light as opposed
            to ambient white.
        gpu_idx: which gpu to use for rendering the mesh.
        path_manager_factory: (Optional) An object that generates an instance of
            PathManager that can translate provided file paths.
        path_manager_factory_class_type: The class type of `path_manager_factory`.
    """

    num_views: int = 40
    data_file: Optional[str] = None
    azimuth_range: float = 180
    distance: float = 2.7
    resolution: int = 128
    use_point_light: bool = True
    gpu_idx: Optional[int] = 0
    # pyre-fixme[13]: Attribute `path_manager_factory` is never initialized.
    path_manager_factory: PathManagerFactory
    path_manager_factory_class_type: str = "PathManagerFactory"

    def get_dataset_map(self) -> DatasetMap:
        # pyre-ignore[16]
        return DatasetMap(train=self.train_dataset, val=None, test=None)

    def get_all_train_cameras(self) -> CamerasBase:
        # pyre-ignore[16]
        return self.poses

    def __post_init__(self) -> None:
        super().__init__()
        run_auto_creation(self)
        if torch.cuda.is_available() and self.gpu_idx is not None:
            device = torch.device(f"cuda:{self.gpu_idx}")
        else:
            device = torch.device("cpu")
        if self.data_file is None:
            data_file = join(
                dirname(dirname(dirname(dirname(realpath(__file__))))),
                "docs",
                "tutorials",
                "data",
                "cow_mesh",
                "cow.obj",
            )
        else:
            data_file = self.data_file
        io = IO(path_manager=self.path_manager_factory.get())
        mesh = io.load_mesh(data_file, device=device)
        poses, images, masks = _generate_cow_renders(
            num_views=self.num_views,
            mesh=mesh,
            azimuth_range=self.azimuth_range,
            distance=self.distance,
            resolution=self.resolution,
            device=device,
            use_point_light=self.use_point_light,
        )
        # pyre-ignore[16]
        self.poses = poses.cpu()
        # pyre-ignore[16]
        self.train_dataset = SingleSceneDataset(  # pyre-ignore[28]
            object_name="cow",
            images=list(images.permute(0, 3, 1, 2).cpu()),
            fg_probabilities=list(masks[:, None].cpu()),
            poses=[self.poses[i] for i in range(len(poses))],
            frame_types=[DATASET_TYPE_KNOWN] * len(poses),
            eval_batches=None,
        )


@torch.no_grad()
def _generate_cow_renders(
    *,
    num_views: int,
    mesh: Meshes,
    azimuth_range: float,
    distance: float,
    resolution: int,
    device: torch.device,
    use_point_light: bool,
) -> Tuple[CamerasBase, torch.Tensor, torch.Tensor]:
    """
    Returns:
        cameras: A batch of `num_views` `FoVPerspectiveCameras` from which the
            images are rendered.
        images: A tensor of shape `(num_views, height, width, 3)` containing
            the rendered images.
        silhouettes: A tensor of shape `(num_views, height, width)` containing
            the rendered silhouettes.
    """

    # Load obj file

    # We scale normalize and center the target mesh to fit in a sphere of radius 1
    # centered at (0,0,0). (scale, center) will be used to bring the predicted mesh
    # to its original center and scale.  Note that normalizing the target mesh,
    # speeds up the optimization but is not necessary!
    verts = mesh.verts_packed()
    N = verts.shape[0]
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-(center.expand(N, 3)))
    mesh.scale_verts_((1.0 / float(scale)))

    # Get a batch of viewing angles.
    elev = torch.linspace(0, 0, num_views)  # keep constant
    azim = torch.linspace(-azimuth_range, azimuth_range, num_views) + 180.0

    # Place a point light in front of the object. As mentioned above, the front of
    # the cow is facing the -z direction.
    if use_point_light:
        lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    else:
        lights = AmbientLights(device=device)

    # Initialize a perspective camera that represents a batch of different
    # viewing angles. All the cameras helper methods support mixed type inputs and
    # broadcasting. So we can view the camera from a fixed distance, and
    # then specify elevation and azimuth angles for each viewpoint as tensors.
    R, T = look_at_view_transform(dist=distance, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # Define the settings for rasterization and shading.
    # As we are rendering images for visualization
    # purposes only we will set faces_per_pixel=1 and blur_radius=0.0. Refer to
    # rasterize_meshes.py for explanations of these parameters.  We also leave
    # bin_size and max_faces_per_bin to their default values of None, which sets
    # their values using heuristics and ensures that the faster coarse-to-fine
    # rasterization method is used.  Refer to docs/notes/renderer.md for an
    # explanation of the difference between naive and coarse-to-fine rasterization.
    raster_settings = RasterizationSettings(
        image_size=resolution, blur_radius=0.0, faces_per_pixel=1
    )

    # Create a Phong renderer by composing a rasterizer and a shader. The textured
    # Phong shader will interpolate the texture uv coordinates for each vertex,
    # sample from a texture image and apply the Phong lighting model
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0, 0.0, 0.0))
    rasterizer_type = MeshRasterizer
    renderer = MeshRendererWithFragments(
        rasterizer=rasterizer_type(cameras=cameras, raster_settings=raster_settings),
        shader=HardPhongShader(
            device=device, cameras=cameras, lights=lights, blend_params=blend_params
        ),
    )

    # Create a batch of meshes by repeating the cow mesh and associated textures.
    # Meshes has a useful `extend` method which allows us do this very easily.
    # This also extends the textures.
    meshes = mesh.extend(num_views)

    # Render the cow mesh from each viewing angle
    target_images, fragments = renderer(meshes, cameras=cameras, lights=lights)
    silhouette_binary = (fragments.pix_to_face[..., 0] >= 0).float()

    return cameras, target_images[..., :3], silhouette_binary
