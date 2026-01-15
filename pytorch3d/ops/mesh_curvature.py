import torch
import pytorch3d 
# from pytorch3d.ops import cot_laplacian
# from pytorch3d.structures import Meshes

def one_hot_sparse(A,num_classes,value=None):
    A = A.int()
    B = torch.arange(A.shape[0]).to(A.device)
    if value==None:
        C = torch.ones_like(B)
    else:
        C = value
    return torch.sparse_coo_tensor(torch.stack([B,A]),C,size=(A.shape[0],num_classes))

def faces_angle(meshs: pytorch3d.structures.Meshes)->torch.Tensor:
    """
    Compute the angle of each face in a mesh
    Args:
        meshs: Meshes object
    Returns:
        angles: Tensor of shape (N,3) where N is the number of faces
    """
    Face_coord = meshs.verts_packed()[meshs.faces_packed()]
    A = Face_coord[:,1,:] - Face_coord[:,0,:]
    B = Face_coord[:,2,:] - Face_coord[:,1,:]
    C = Face_coord[:,0,:] - Face_coord[:,2,:]
    angle_0 = torch.arccos(-torch.sum(A*C,dim=1)/torch.norm(A,dim=1)/torch.norm(C,dim=1))
    angle_1 = torch.arccos(-torch.sum(A*B,dim=1)/torch.norm(A,dim=1)/torch.norm(B,dim=1))
    angle_2 = torch.arccos(-torch.sum(B*C,dim=1)/torch.norm(B,dim=1)/torch.norm(C,dim=1))
    angles = torch.stack([angle_0,angle_1,angle_2],dim=1)
    return angles

def dual_area_weights_on_faces(Surfaces: pytorch3d.structures.Meshes)->torch.Tensor:
    """
    Compute the dual area weights of 3 vertices of each triangles in a mesh
    Args:
        Surfaces: Meshes object
    Returns:
        dual_area_weight: Tensor of shape (N,3) where N is the number of triangles
        the dual area of a vertices in a triangles is defined as the area of the sub-quadrilateral divided by three perpendicular bisectors
    """
    angles = faces_angle(Surfaces)
    sin2angle = torch.sin(2*angles)
    dual_area_weight = torch.ones_like(Surfaces.faces_packed())*(torch.sum(sin2angle,dim=1).view(-1,1).repeat(1,3))
    for i in range(3):
        j,k = (i+1)%3, (i+2)%3
        dual_area_weight[:,i] = 0.5*(sin2angle[:,j]+sin2angle[:,k])/dual_area_weight[:,i]
    return dual_area_weight


def Dual_area_for_vertices(Surfaces: pytorch3d.structures.Meshes)->torch.Tensor:
    """
    Compute the dual area of each vertices in a mesh
    Args:
        Surfaces: Meshes object
    Returns:
        dual_area_vertex: Tensor of shape (N,1) where N is the number of vertices
        the dual area of a vertices is defined as the sum of the dual area of the triangles that contains this vertices
    """

    dual_area_weight = dual_area_weights_on_faces(Surfaces)
    dual_area_faces = Surfaces.faces_areas_packed().view(-1,1).repeat(1,3)*dual_area_weight
    face_vertices_to_idx = one_hot_sparse(Surfaces.faces_packed().view(-1),num_classes=Surfaces.num_verts_per_mesh().sum())
    dual_area_vertex = torch.sparse.mm(face_vertices_to_idx.float().T,dual_area_faces.view(-1,1)).T
    return dual_area_vertex


def Gaussian_curvature(Surfaces: pytorch3d.structures.Meshes,return_topology=False)->torch.Tensor:
    """
    Compute the gaussian curvature of each vertices in a mesh by local Gauss-Bonnet theorem
    Args:
        Surfaces: Meshes object
        return_topology: bool, if True, return the Euler characteristic and genus of the mesh
    Returns:
        gaussian_curvature: Tensor of shape (N,1) where N is the number of vertices
        the gaussian curvature of a vertices is defined as the sum of the angles of the triangles that contains this vertices minus 2*pi and divided by the dual area of this vertices
    """

    face_vertices_to_idx = one_hot_sparse(Surfaces.faces_packed().view(-1),num_classes=Surfaces.num_verts_per_mesh().sum())
    vertices_to_meshid = one_hot_sparse(Surfaces.verts_packed_to_mesh_idx(),num_classes=Surfaces.num_verts_per_mesh().shape[0])
    sum_angle_for_vertices = torch.sparse.mm(face_vertices_to_idx.float().T,faces_angle(Surfaces).view(-1,1)).T
    # Euler_chara = torch.sparse.mm(vertices_to_meshid.float().T,(2*torch.pi - sum_angle_for_vertices).T).T/torch.pi/2
    # Euler_chara = torch.round(Euler_chara)
    # print('Euler_characteristic:',Euler_chara)
    # Genus = (2-Euler_chara)/2
    #print('Genus:',Genus)
    gaussian_curvature = (2*torch.pi - sum_angle_for_vertices)/Dual_area_for_vertices(Surfaces)
    if return_topology:
        Euler_chara = torch.sparse.mm(vertices_to_meshid.float().T,(2*torch.pi - sum_angle_for_vertices).T).T/torch.pi/2
        Euler_chara = torch.round(Euler_chara)
        return gaussian_curvature, Euler_chara, Genus
    return gaussian_curvature

def Average_from_verts_to_face(Surfaces: pytorch3d.structures.Meshes, vect_verts: torch.Tensor)->torch.Tensor:
    """
    Compute the average of feature vectors defined on vertices to faces by dual area weights
    Args:
        Surfaces: Meshes object
        vect_verts: Tensor of shape (N,C) where N is the number of vertices, C is the number of feature channels
    Returns:
        vect_faces: Tensor of shape (F,C) where F is the number of faces
    """
    assert vect_verts.shape[0] == Surfaces.verts_packed().shape[0]
    dual_weight = dual_area_weights_on_faces(Surfaces).view(-1)
    wg = one_hot_sparse(Surfaces.faces_packed().view(-1),num_classes=Surfaces.num_verts_per_mesh().sum(),value=dual_weight).float()
    return torch.sparse.mm(wg,vect_verts).view(-1,3).sum(dim=1)
