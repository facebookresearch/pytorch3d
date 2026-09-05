import torch


def volume_centroid(mesh):
    """
    Compute the volumetric centroid of this mesh, which is distinct from the center of mass.
    The center of mass (average of all vertices) will be closer to where there are a
    higher density of points in a mesh are, but the centroid, which is based on volume,
    will be closer to a perceived center of the mesh, as opposed to based on the density
    of vertices. This function assumes that the mesh is watertight, and that the faces are
    all oriented in the same direction.
    Returns:
      The position of the centroid as a tensor of shape (3).
    """
    v_idxs = mesh.faces_padded().split([1, 1, 1], dim=-1)
    verts = mesh.verts_padded()
    valid = (mesh.faces_padded() != -1).all(dim=-1, keepdim=True)

    v0, v1, v2 = [
        torch.gather(
            verts,
            1,
            idx.where(valid, torch.zeros_like(idx)).expand(-1, -1, 3),
        ).where(valid, torch.zeros_like(idx, dtype=verts.dtype))
        for idx in v_idxs
    ]

    tetra_center = (v0 + v1 + v2) / 4
    signed_tetra_vol = (v0 * torch.cross(v1, v2, dim=-1)).sum(dim=-1, keepdim=True) / 6
    denom = signed_tetra_vol.sum(dim=-2)
    # clamp the denominator to prevent instability for degenerate meshes.
    denom = torch.where(denom < 0, denom.clamp(max=-1e-5), denom.clamp(min=1e-5))
    return (tetra_center * signed_tetra_vol).sum(dim=-2) / denom
