---
hide_title: true
sidebar_label: Cubify
---

# Cubify

The [cubify operator](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/ops/cubify.py) converts an 3D occupancy grid of shape `BxDxHxW`, where `B` is the batch size, into a mesh instantiated as a [Meshes](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/structures/meshes.py) data structure of `B` elements. The operator replaces every occupied voxel (if its occupancy probability is greater than a user defined threshold) with a cuboid of 12 faces and 8 vertices. Shared vertices are merged, and internal faces are removed resulting in a **watertight** mesh.

The operator provides three alignment modes {*topleft*, *corner*, *center*} which define the span of the mesh vertices with respect to the voxel grid. The alignment modes are described in the figure below for a 2D grid.

![input](https://user-images.githubusercontent.com/4369065/81032959-af697380-8e46-11ea-91a8-fae89597f988.png)
