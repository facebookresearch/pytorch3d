import torch
from pytorch3d import _C
from pytorch3d.ops import knn_points

class P2F_dist(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,points, vertices, faces):
        #points:    N,3
        #vertices:  M,3
        #faces:     M,3
        f1_idx,f2_idx,f3_idx=faces[:,0],faces[:,1],faces[:,2]
        f1=vertices[f1_idx]
        f2=vertices[f2_idx]
        f3=vertices[f3_idx]
        indexes,w1,w2,w3=_C.closest_point_on_surface_forward(points,f1,f2,f3)
        return indexes,w1,w2,w3
    

def face_area_normals(faces,vs):
    #faces  [M,3]
    #vs     [B,N,3]
    face_normals=torch.cross(vs[:,faces[:,1],:]-vs[:,faces[:,0],:],
                             vs[:,faces[:,2],:]-vs[:,faces[:,0],:],dim=2)
    face_areas=torch.norm(face_normals,dim=2)
    face_normals=face_normals/face_areas[:,:,None]
    face_areas=0.5*face_areas
    return face_areas,face_normals

def sampl_surface(faces,vs,count):
    #faces:     [M,3]
    #vs:        [N,3]
    vs=vs.unsqueeze(0)
    bsize,nvs,_=vs.shape
    weights,normal=face_area_normals(faces,vs)
    weights_sum = torch.sum(weights, dim=1)
    dist = torch.distributions.categorical.Categorical(probs=weights / weights_sum[:, None])
    face_index = dist.sample((count,))

    tri_origins = vs[:, faces[:, 0], :]
    tri_vectors = vs[:, faces[:, 1:], :].clone()
    tri_vectors -= tri_origins.repeat(1, 1, 2).reshape((bsize, len(faces), 2, 3))

    # pull the vectors for the faces we are going to sample from
    face_index = face_index.transpose(0, 1)
    face_index = face_index[:, :, None].expand((bsize, count, 3))
    tri_origins = torch.gather(tri_origins, dim=1, index=face_index)
    face_index2 = face_index[:, :, None, :].expand((bsize, count, 2, 3))
    tri_vectors = torch.gather(tri_vectors, dim=1, index=face_index2)

    # randomly generate two 0-1 scalar components to multiply edge vectors by
    random_lengths = torch.rand(count, 2, 1, device=vs.device, dtype=tri_vectors.dtype)

    # points will be distributed on a quadrilateral if we use 2x [0-1] samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    random_test = random_lengths.sum(dim=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = torch.abs(random_lengths)

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * random_lengths[None, :]).sum(dim=2)

    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    samples = sample_vector + tri_origins

    normals = torch.gather(normal, dim=1, index=face_index)

    return samples.squeeze(0), normals.squeeze(0)


def get_face_center(vertices,faces):
    return (vertices[faces[:,0]]+vertices[faces[:,1]]+vertices[faces[:,2]])/3


class DirDist_M2M(torch.nn.Module):
    def __init__(self,num_query=20000,std=0.05):
        super().__init__()        
        self.num_query=num_query
        self.std=std


    def forward(self,src_v,src_f,tgt_v,tgt_f):
        #src_v, tgt_v   [N,3]
        #src_f, tgt_g   [M,3], long

        '''
        Note: You could also choose 'pytorch3d.structures.Meshes' to represent the two meshes
        src_v=src_mesh.verts_packed()
        src_f=src_mesh.faces_packed()

        tgt_v=tgt_mesh.verts_packed()
        tgt_f=tgt_mesh.faces_packed()'''

        query_points,_=sampl_surface(tgt_f,tgt_v,self.num_query)
        noise_offset=torch.randn_like(query_points)*self.std
        query_points=query_points+noise_offset

        src_f1=src_v[src_f[:,0]]
        src_f2=src_v[src_f[:,1]]
        src_f3=src_v[src_f[:,2]]

        tgt_f1=tgt_v[tgt_f[:,0]]
        tgt_f2=tgt_v[tgt_f[:,1]]
        tgt_f3=tgt_v[tgt_f[:,2]]

        src_center=(src_f1+src_f2+src_f3)/3
        tgt_center=(tgt_f1+tgt_f2+tgt_f3)/3

        query_points=torch.cat([query_points.detach(),src_center.detach()],dim=0)

        src_indexes,src_w1,src_w2,src_w3=_C.closest_point_on_surface_forward(query_points,src_f1,src_f2,src_f3)
        tgt_indexes,tgt_w1,tgt_w2,tgt_w3=_C.closest_point_on_surface_forward(query_points,tgt_f1,tgt_f2,tgt_f3)

        sel_src_f1=src_f1[src_indexes.long()]
        sel_src_f2=src_f2[src_indexes.long()]
        sel_src_f3=src_f3[src_indexes.long()]

        sel_tgt_f1=tgt_f1[tgt_indexes.long()]
        sel_tgt_f2=tgt_f2[tgt_indexes.long()]
        sel_tgt_f3=tgt_f3[tgt_indexes.long()]

        closest_src=src_w1[:,None]*sel_src_f1+src_w2[:,None]*sel_src_f2+src_w3[:,None]*sel_src_f3
        closest_tgt=tgt_w1[:,None]*sel_tgt_f1+tgt_w2[:,None]*sel_tgt_f2+tgt_w3[:,None]*sel_tgt_f3

        dir_src=query_points-closest_src
        udf_src=torch.norm(dir_src+1e-10,dim=-1,keepdim=True)
        geo_src=torch.cat([dir_src,udf_src],dim=1)


        dir_tgt=query_points-closest_tgt
        udf_tgt=torch.norm(dir_tgt+1e-10,dim=-1,keepdim=True)
        geo_tgt=torch.cat([dir_tgt,udf_tgt],dim=1)

        return torch.mean(torch.abs(geo_src-geo_tgt))*4


class DirDist_P2P(torch.nn.Module):
    def __init__(self,up_ratio=10,K=5,std=0.05,weighted_query=True,beta=3):
        super().__init__()
        self.K=K
        self.up_ratio=up_ratio
        self.std=std
        self.weighted_query=weighted_query
        self.beta=beta

    def cal_udf_weights(self,x,query):
        #x: (B,N,3)
        #query=self.grid_flatten.to(x).unsqueeze(0).repeat(x.size(0),1,1)
        
        dists,idx,knn_pc=knn_points(query,x,K=self.K,return_nn=True,return_sorted=True)   #(B,N,K) (B,N,K) (B,N,K,3)

        dir=query.unsqueeze(2)-knn_pc   #(B,N,K,3)

        #weights=torch.softmax(-dists.sqrt(),dim=2)   #(B,N,K) weight more, dist small
        #weights=torch.softmax(-dists,dim=2)   #(B,N,K) weight more, dist small
        #weights=torch.softmax(-dists/torch.min(dists,dim=2,keepdim=True)[0],dim=2)   #(B,N,K) weight more, dist small

        norm = torch.sum(1.0 / (dists + 1e-8), dim = 2, keepdim = True)
        weights = (1.0 / (dists.detach() + 1e-8)) / norm.detach()


        #print(weights)
        #assert False

        #udf=torch.sum((dists+1e-10).sqrt()*weights,dim=2)  #(B,N)
        #udf=torch.sum(dists*weights,dim=2)  #(B,N)

        udf_grad=torch.sum(dir*weights.unsqueeze(-1),dim=2) #(B,N,3)
        udf=torch.norm(udf_grad+1e-10,dim=-1)

        return udf,udf_grad,weights

    def cal_udf(self,x,weights,query):
        #query=self.grid_flatten.to(x).unsqueeze(0).repeat(x.size(0),1,1)

        dists,idx,knn_pc=knn_points(query,x,K=self.K,return_nn=True,return_sorted=True)   #(B,N,K) (B,N,K) (B,N,K,3)
        dir=query.unsqueeze(2)-knn_pc   #(B,N,K,3)
        #udf=torch.sum((dists+1e-10).sqrt()*weights,dim=2)  #(B,N)
        #udf=torch.sum(dists*weights,dim=2)  #(B,N)

        udf_grad=torch.sum(dir*weights.unsqueeze(-1),dim=2) #(B,N,3)
        udf=torch.norm(udf_grad+1e-10,dim=-1)
        return udf,udf_grad

    def forward(self,src,tgt):
        #src: target (B,N,3)
        #tgt: source (B,N,3)

        with torch.no_grad():

            std=self.std
            noise_offset=torch.randn(tgt.size(0),tgt.size(1),self.up_ratio,3).to(tgt).float() * std

            
            query=tgt.unsqueeze(2)+noise_offset
            query=query.reshape(tgt.size(0),-1,3).detach()

        
        query=torch.cat((query,src.detach()),dim=1)

        udf_tgt,udf_grad_tgt,_=self.cal_udf_weights(tgt,query)
        udf_src,udf_grad_src,_=self.cal_udf_weights(src,query)
        

        udf_error=torch.abs(udf_tgt-udf_src)    #(B,M)

        udf_grad_error=torch.sum(torch.abs(udf_grad_src-udf_grad_tgt),axis=-1)  #(B,M)
        #udf_grad_loss=torch.mean(torch.square(udf_grad_src-udf_grad_tgt))

        if self.weighted_query:

            with torch.no_grad():
                query_weights=torch.exp(-udf_error.detach()*self.beta)*torch.exp(-udf_grad_error.detach()*self.beta)
            return torch.sum((udf_error+udf_grad_error)*query_weights.detach())/query.size(0)/query.size(1)
        
        else:
            query_weights=1
            return torch.sum((udf_error+udf_grad_error)*query_weights)/query.size(0)/query.size(1)


class DirDist_M2P(torch.nn.Module):
    def __init__(self,up_ratio=3,beta=0,K=5,std=0.05):
        super().__init__()
        self.up_ratio=up_ratio
        self.beta=beta
        self.K=K
        self.std=std

    def forward(self,src_v,src_f,tgt_points):
        #src_v  [N,3]
        #src_f  [F,3]   long
        #tgt_points [M,3]

        src_f1=src_v[src_f[:,0]]
        src_f2=src_v[src_f[:,1]]
        src_f3=src_v[src_f[:,2]]

        src_center=(src_f1+src_f2+src_f3)/3

        query_points=tgt_points.unsqueeze(1)+self.std*torch.randn(tgt_points.size(0),self.up_ratio,tgt_points.size(1)).to(tgt_points)
        query_points=query_points.reshape(-1,3)

        query_points=torch.cat([query_points.detach(),src_center.detach()],dim=0)

        src_indexes,src_w1,src_w2,src_w3=_C.closest_point_on_surface_forward(query_points,src_f1,src_f2,src_f3)

        sel_src_f1=src_f1[src_indexes.long()]
        sel_src_f2=src_f2[src_indexes.long()]
        sel_src_f3=src_f3[src_indexes.long()]

        closest_src=src_w1[:,None]*sel_src_f1+src_w2[:,None]*sel_src_f2+src_w3[:,None]*sel_src_f3

        dir_src=query_points-closest_src
        udf_src=torch.norm(dir_src+1e-10,dim=-1,keepdim=True)
        geo_src=torch.cat([dir_src,udf_src],dim=1)

        dists,_,knn_pc=knn_points(query_points.unsqueeze(0),tgt_points.unsqueeze(0),K=self.K,return_nn=True,return_sorted=True)   #(B,N,K) (B,N,K) (B,N,K,3)

        dir=query_points.unsqueeze(0).unsqueeze(2)-knn_pc   #(B,N,K,3)

        norm = torch.sum(1.0 / (dists + 1e-8), dim = 2, keepdim = True)
        weights = (1.0 / (dists.detach() + 1e-8)) / norm.detach()

        
        dir_tgt=torch.sum(dir*weights.unsqueeze(-1),dim=2).squeeze(0)   #(N,3)
        udf_tgt=torch.norm(dir_tgt+1e-10,dim=1)

        geo_tgt=torch.cat([dir_tgt,udf_tgt.unsqueeze(1)],dim=-1)    

        errors=torch.sum(torch.abs(geo_src-geo_tgt),dim=-1)   

        query_weights=torch.exp(-errors*self.beta).detach()     

        return torch.mean(errors*query_weights)
    



