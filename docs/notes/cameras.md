---
hide_title: true
sidebar_label: Cameras
---

# Cameras

## Camera Coordinate Systems

When working with 3D data, there are 4 coordinate systems users need to know
* **World coordinate system**
This is the system the object/scene lives - the world.
* **Camera view coordinate system**
This is the system that has its origin on the image plane and the `Z`-axis perpendicular to the image plane. In PyTorch3D, we assume that `+X` points left, and `+Y` points up and `+Z` points out from the image plane. The transformation from world to view happens after applying a rotation (`R`) and translation (`T`).
* **NDC coordinate system**
This is the normalized coordinate system that confines in a volume the rendered part of the object/scene. Also known as view volume. For square images, under the PyTorch3D convention, `(+1, +1, znear)` is the top left near corner, and `(-1, -1, zfar)` is the bottom right far corner of the volume. For non-square images, the side of the volume in `XY` with the smallest length ranges from `[-1, 1]` while the larger side from `[-s, s]`, where `s` is the aspect ratio and `s > 1` (larger divided by smaller side).
The transformation from view to NDC happens after applying the camera projection matrix (`P`).
* **Screen coordinate system**
This is another representation of the view volume with the `XY` coordinates defined in pixel space instead of a normalized space. (0,0) is the top left corner of the top left pixel
and (W,H) is the bottom right corner of the bottom right pixel.

An illustration of the 4 coordinate systems is shown below
![cameras](https://user-images.githubusercontent.com/669761/145090051-67b506d7-6d73-4826-a677-5873b7cb92ba.png)

## Defining Cameras in PyTorch3D

Cameras in PyTorch3D transform an object/scene from world to view by first transforming the object/scene to view (via transforms `R` and `T`) and then projecting the 3D object/scene to a normalized space via the projection matrix `P = K[R | T]`, where `K` is the intrinsic matrix. The camera parameters in `K` define the normalized space. If users define the camera parameters in NDC space, then the transform projects points to NDC. If the camera parameters are defined in screen space, the transformed points are in screen space.

Note that the base `CamerasBase` class makes no assumptions about the coordinate systems. All the above transforms are geometric transforms defined purely by `R`, `T` and `K`. This means that users can define cameras  in any coordinate system and for any transforms. The method `transform_points` will apply `K` , `R` and `T` to the input points as a simple matrix transformation. However, if users wish to use cameras with the PyTorch3D renderer, they need to abide to PyTorch3D's coordinate system assumptions (read below).

We provide instantiations of common camera types in PyTorch3D and how users can flexibly define the projection space below.

## Interfacing with the PyTorch3D Renderer

The PyTorch3D renderer for both meshes and point clouds assumes that the camera transformed points, meaning the points passed as input to the rasterizer, are in PyTorch3D's NDC space. So to get the expected rendering outcome, users need to make sure that their 3D input data and cameras abide by these PyTorch3D coordinate system assumptions. The PyTorch3D coordinate system assumes `+X:left`, `+Y: up` and `+Z: from us to scene` (right-handed) . Confusions regarding coordinate systems are common so we advise that you spend some time understanding your data and the coordinate system they live in and transform them accordingly before using the PyTorch3D renderer.

Examples of cameras and how they interface with the PyTorch3D renderer can be found in our tutorials.

### Camera Types

All cameras inherit from `CamerasBase` which is a base class for all cameras. PyTorch3D provides four different camera types. The `CamerasBase` defines methods that are common to all camera models:
* `get_camera_center` that returns the optical center of the camera in world coordinates
* `get_world_to_view_transform` which returns a 3D transform from world coordinates to the camera view coordinates `(R, T)`
* `get_full_projection_transform` which composes the projection transform (`K`) with the world-to-view transform `(R, T)`
* `transform_points` which takes a set of input points in world coordinates and projects to NDC coordinates ranging from [-1, -1, znear] to  [+1, +1, zfar].
* `get_ndc_camera_transform` which defines the conversion to PyTorch3D's NDC space and is called when interfacing with the PyTorch3D renderer. If the camera is defined in NDC space, then the identity transform is returned. If the cameras is defined in screen space, the conversion from screen to NDC is returned. If users define their own camera in screen space, they need to think of the screen to NDC conversion. We provide examples for the `PerspectiveCameras` and `OrthographicCameras`.
* `transform_points_ndc` which takes a set of points in world coordinates and projects them to PyTorch3D's NDC space
* `transform_points_screen` which takes a set of input points in world coordinates and projects them to the screen coordinates ranging from [0, 0, znear] to [W, H, zfar]

Users can easily customize their own cameras. For each new camera, users should implement the `get_projection_transform` routine that returns the mapping `P` from camera view coordinates to NDC coordinates.

#### FoVPerspectiveCameras, FoVOrthographicCameras
These two cameras follow the OpenGL convention for perspective and orthographic cameras respectively. The user provides the near `znear` and far `zfar` field which confines the view volume in the `Z` axis. The view volume in the `XY` plane is defined by field of view angle (`fov`) in the case of `FoVPerspectiveCameras` and by `min_x, min_y, max_x, max_y` in the case of `FoVOrthographicCameras`.
These cameras are by default in NDC space.

#### PerspectiveCameras, OrthographicCameras
These two cameras follow the Multi-View Geometry convention for cameras. The user provides the focal length (`fx`, `fy`) and the principal point (`px`, `py`). For example, `camera = PerspectiveCameras(focal_length=((fx, fy),), principal_point=((px, py),))`

The camera projection of a 3D point `(X, Y, Z)` in view coordinates to a point `(x, y, z)` in projection space (either NDC or screen) is

```
# for perspective camera
x = fx * X / Z + px
y = fy * Y / Z + py
z = 1 / Z

# for orthographic camera
x = fx * X + px
y = fy * Y + py
z = Z
```

The user can define the camera parameters in NDC or in screen space. Screen space camera parameters are common and for that case the user needs to set `in_ndc` to `False` and also provide the `image_size=(height, width)` of the screen, aka the image.

The `get_ndc_camera_transform` provides the transform from screen to NDC space in PyTorch3D. Note that the screen space assumes that the principal point is provided in the space with `+X left`, `+Y down` and origin at the top left corner of the image. To convert to NDC we need to account for the scaling of the normalized space as well as the change in `XY` direction.

Below are example of equivalent `PerspectiveCameras` instantiations in NDC and screen space, respectively.

```python
# NDC space camera
fcl_ndc = (1.2,)
prp_ndc = ((0.2, 0.5),)
cameras_ndc = PerspectiveCameras(focal_length=fcl_ndc, principal_point=prp_ndc)

# Screen space camera
image_size = ((128, 256),)    # (h, w)
fcl_screen = (76.8,)          # fcl_ndc * min(image_size) / 2
prp_screen = ((115.2, 32), )  # w / 2 - px_ndc * min(image_size) / 2, h / 2 - py_ndc * min(image_size) / 2
cameras_screen = PerspectiveCameras(focal_length=fcl_screen, principal_point=prp_screen, in_ndc=False, image_size=image_size)
```

The relationship between screen and NDC specifications of a camera's `focal_length` and `principal_point` is given by the following equations, where `s = min(image_width, image_height)`.
The transformation of x and y coordinates between screen and NDC is exactly the same as for px and py.

```
fx_ndc = fx_screen * 2.0 / s
fy_ndc = fy_screen * 2.0 / s

px_ndc = - (px_screen - image_width / 2.0) * 2.0 / s
py_ndc = - (py_screen - image_height / 2.0) * 2.0 / s
```
