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
This is the normalized coordinate system that confines in a volume the rendered part of the object/scene. Also known as view volume. Under the PyTorch3D convention, `(+1, +1, znear)` is the top left near corner, and `(-1, -1, zfar)` is the bottom right far corner of the volume. The transformation from view to NDC happens after applying the camera projection matrix (`P`).
* **Screen coordinate system**
This is another representation of the view volume with the `XY` coordinates defined in pixel space instead of a normalized space.

An illustration of the 4 coordinate systems is shown below
![cameras](https://user-images.githubusercontent.com/4369065/90317960-d9b8db80-dee1-11ea-8088-39c414b1e2fa.png)

## Defining Cameras in PyTorch3D

Cameras in PyTorch3D transform an object/scene from world to NDC by first transforming the object/scene to view (via transforms `R` and `T`) and then projecting the 3D object/scene to NDC (via the projection matrix `P`, else known as camera matrix). Thus, the camera parameters in `P` are assumed to be in NDC space. If the user has camera parameters in screen space, which is a common use case, the parameters should transformed to NDC (see below for an example)

We describe the camera types in PyTorch3D and the convention for the camera parameters provided at construction time.

### Camera Types

All cameras inherit from `CamerasBase` which is a base class for all cameras. PyTorch3D provides four different camera types. The `CamerasBase` defines methods that are common to all camera models:
* `get_camera_center` that returns the optical center of the camera in world coordinates
* `get_world_to_view_transform` which returns a 3D transform from world coordinates to the camera view coordinates (R, T)
* `get_full_projection_transform` which composes the projection transform (P) with the world-to-view transform (R, T)
* `transform_points` which takes a set of input points in world coordinates and projects to NDC coordinates ranging from [-1, -1, znear] to  [+1, +1, zfar].
* `transform_points_screen` which takes a set of input points in world coordinates and projects them to the screen coordinates ranging from [0, 0, znear] to [W-1, H-1, zfar]

Users can easily customize their own cameras. For each new camera, users should implement the `get_projection_transform` routine that returns the mapping `P` from camera view coordinates to NDC coordinates.

#### FoVPerspectiveCameras, FoVOrthographicCameras
These two cameras follow the OpenGL convention for perspective and orthographic cameras respectively. The user provides the near `znear` and far `zfar` field which confines the view volume in the `Z` axis. The view volume in the `XY` plane is defined by field of view angle (`fov`) in the case of `FoVPerspectiveCameras` and by `min_x, min_y, max_x, max_y` in the case of `FoVOrthographicCameras`.

#### PerspectiveCameras, OrthographicCameras
These two cameras follow the Multi-View Geometry convention for cameras. The user provides the focal length (`fx`, `fy`) and the principal point (`px`, `py`). For example, `camera = PerspectiveCameras(focal_length=((fx, fy),), principal_point=((px, py),))`

As mentioned above, the focal length and principal point are used to convert a point `(X, Y, Z)` from view coordinates to NDC coordinates, as follows

```
# for perspective
x_ndc = fx * X / Z + px
y_ndc = fy * Y / Z + py
z_ndc = 1 / Z

# for orthographic
x_ndc = fx * X + px
y_ndc = fy * Y + py
z_ndc = Z
```

Commonly, users have access to the focal length (`fx_screen`, `fy_screen`) and the principal point (`px_screen`, `py_screen`) in screen space. In that case, to construct the camera the user needs to additionally provide the `image_size = ((image_width, image_height),)`. More precisely, `camera = PerspectiveCameras(focal_length=((fx_screen, fy_screen),), principal_point=((px_screen, py_screen),), image_size = ((image_width, image_height),))`. Internally, the camera parameters are converted from screen to NDC as follows:

```
fx = fx_screen * 2.0 / image_width
fy = fy_screen * 2.0 / image_height

px = - (px_screen - image_width / 2.0) * 2.0 / image_width
py = - (py_screen - image_height / 2.0) * 2.0/ image_height
```
