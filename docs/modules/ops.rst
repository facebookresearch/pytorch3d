pytorch3d.ops 
===========================

``voxel_down_sample`` reduces each cloud in a ``Pointclouds`` batch to one
averaged point per occupied voxel. Optional features and normals are averaged
along with positions, preserving gradients within each voxel assignment.
The grid has a fixed world origin at zero; its size does not depend on the
point cloud's bounding box. Empty clouds and heterogeneous batches are supported.

.. automodule:: pytorch3d.ops
    :members:
    :undoc-members:
