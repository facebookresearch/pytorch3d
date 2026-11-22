import torch
import pytorch3d 
from pytorch3d.ops import knn_points,knn_gather
# from pytorch3d.structures import Meshes

def Weingarten_maps(pointscloud:torch.Tensor, k=50)->torch.Tensor:
    """
    Compute the Weingarten maps of a point cloud
    Args:
        pointscloud: Tensor of shape (B,N,3) where N is the number of points in each batch
        k: int, number of neighbors
    Returns:
        Weingarten_fields: Tensor of shape (B,N,2,2) where N is the number of points
        normals_field: Tensor of shape (B,N,3) where N is the number of points
        tangent1_field: Tensor of shape (B,N,3) 
        tangent2_field: Tensor of shape (B,N,3) 
    """


    pointscloud_shape = pointscloud.shape
    batch_size = pointscloud_shape[0]
    num_points = torch.LongTensor([pointscloud_shape[1]]*batch_size)

    # undo global mean for stability
    pointscloud_centered = pointscloud - pointscloud.mean(-2).view(batch_size,1,3)
    knn_info = knn_points(pointscloud_centered,pointscloud_centered,lengths1=num_points,lengths2=num_points,K=k,return_nn=True)
    
    # compute knn & covariance matrix 
    knn_point_centered = knn_info.knn - knn_info.knn.mean(-2).view(batch_size,-1,1,3)
    covs_field = torch.matmul(knn_point_centered.transpose(-1,-2),knn_point_centered) /(knn_point_centered.shape[-1]-1)
    frames_field = torch.linalg.eigh(covs_field).eigenvectors

    normals_field = frames_field[:,:,:,0]
    tangent1_field = frames_field[:,:,:,1]
    tangent2_field = frames_field[:,:,:,2]


    local_pt_difference = knn_info.knn[:,:,1:k,:]- pointscloud_centered[:,:,None,:] # B x N x K x 3

    # Disambiguates normals by checking the sign of the projection of the
    proj = (normals_field[:, :, None] * local_pt_difference).sum(-1)
    # check how many projections are positive
    n_pos = (proj > 0).type_as(knn_info.knn).sum(-1, keepdim=True)
    # flip the principal directions where number of positive correlations for 
    flip = (n_pos < (0.5 * (k-1))).type_as(knn_info.knn)

    normals_field = (1.0 - 2.0 * flip) * normals_field

    # local normals difference
    local_normals_difference = knn_gather(normals_field,knn_info.idx,lengths=num_points)[:,:,1:k,:] - normals_field[:,:,None,:]

    # project the difference onto the tangent plane, getting the differential of the gaussian map
    local_dpt_tangent1 = (local_pt_difference * tangent1_field[:,:,None,:]).sum(-1,keepdim=True)
    local_dpt_tangent2 = (local_pt_difference * tangent2_field[:,:,None,:]).sum(-1,keepdim=True)
    local_dpt_tangent = torch.cat((local_dpt_tangent1,local_dpt_tangent2),dim=-1)
    local_dnormals_tangent1 = (local_normals_difference * tangent1_field[:,:,None,:]).sum(-1,keepdim=True)
    local_dnormals_tangent2 = (local_normals_difference * tangent2_field[:,:,None,:]).sum(-1,keepdim=True)
    local_dnormals_tangent = torch.cat((local_dnormals_tangent1,local_dnormals_tangent2),dim=-1)


    # estimate the weingarten map by solving a least squares problem: W = Dn^T Dp (Dp^T Dp)^-1
    XXT = torch.matmul(local_dpt_tangent.transpose(-1,-2),local_dpt_tangent)
    YXT = torch.matmul(local_dnormals_tangent.transpose(-1,-2),local_dpt_tangent)
    XYT = torch.matmul(local_dpt_tangent.transpose(-1,-2),local_dnormals_tangent)
    #Weingarten_fields_0 = torch.matmul(YXT,torch.inverse(XXT+1e-8*torch.eye(2).type_as(XXT))) ## the unsymetric version


    # solve the sylvester equation to get the shape operator (symmetric version)
    S = YXT + XYT

    XXT_eig = torch.linalg.eigh(XXT)
    Q = XXT_eig.eigenvectors
    #D = torch.diag_embed(XXT_eig.eigenvalues)
    # XX^T = Q^T D Q
    Q_TSQ = torch.matmul(Q.transpose(-1,-2),torch.matmul(S,Q))

    a = XXT_eig.eigenvalues[:,:,0]
    b = XXT_eig.eigenvalues[:,:,1]
    a_b = a+b
    a2_a_b = torch.stack((2*a,a_b),dim=-1).view(batch_size,-1,1,2)
    a_b_b2 = torch.stack((a_b,2*b),dim=-1).view(batch_size,-1,1,2)
    c = torch.stack((a2_a_b,a_b_b2),dim=-2).view(batch_size,-1,2,2)

    E = (1/c+1e-6) * Q_TSQ
    Weingarten_fields = torch.matmul(Q,torch.matmul(E,Q.transpose(-1,-2)))


    return Weingarten_fields, normals_field, tangent1_field, tangent2_field

def Curvature_pcl(pointscloud, k=50, return_princpals=False):
    """
    Compute the gaussian curvature of point clouds
    pointscloud: B x N x 3
    k: int, number of neighbors
    return_princpals: bool,if True, return principal curvature and principal directions
    if False, return gaussian curvature, mean curvature only
    """

    pointscloud_shape = pointscloud.shape
    batch_size = pointscloud_shape[0]
    num_points = torch.LongTensor([pointscloud_shape[1]]*batch_size)
    Weingarten_fields, normals_field, tangent1_field, tangent2_field = Weingarten_maps(pointscloud, k=k)
    tangent_space = tangent_space = torch.cat((tangent1_field.view(batch_size,-1,1,3),tangent2_field.view(batch_size,-1,1,3)),dim=-2)
    if return_princpals:
        principal_curvature , principal_direction_local = torch.linalg.eigh(Weingarten_fields)
        principal_direction_global = torch.matmul(principal_direction_local.transpose(-1,-2),tangent_space)
        return principal_curvature, principal_direction_global, normals_field
    else:
        gaussian_curvature_pcl = torch.det(Weingarten_fields)
        mean_curvature_pcl = Weingarten_fields.diagonal(offset=0, dim1=-1, dim2=-2).mean(-1)
        return gaussian_curvature_pcl, mean_curvature_pcl
    

