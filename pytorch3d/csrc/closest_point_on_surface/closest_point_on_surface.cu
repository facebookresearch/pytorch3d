#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <vector>


__device__ float clamp(const float input, const float min_value, const float max_value)
{
    float output=input;
    if (input>max_value)
    {
        output=max_value;
    }
    else if(input<min_value)
    {
        output=min_value;
    }
    return output;
}




__device__ void closestPointOnTriangle(
    const float p_x,
    const float p_y,
    const float p_z,
    const float f1_x,
    const float f1_y,
    const float f1_z,
    const float f2_x,
    const float f2_y,
    const float f2_z,
    const float f3_x,
    const float f3_y,
    const float f3_z,
    float& result_dist,
    float& result_s,
    float& result_t
    )
{
    const float a=(f2_x-f1_x)*(f2_x-f1_x)+(f2_y-f1_y)*(f2_y-f1_y)+(f2_z-f1_z)*(f2_z-f1_z);
    const float b=(f2_x-f1_x)*(f3_x-f1_x)+(f2_y-f1_y)*(f3_y-f1_y)+(f2_z-f1_z)*(f3_z-f1_z);
    const float c=(f3_x-f1_x)*(f3_x-f1_x)+(f3_y-f1_y)*(f3_y-f1_y)+(f3_z-f1_z)*(f3_z-f1_z);
    const float d=(f2_x-f1_x)*(f1_x-p_x)+(f2_y-f1_y)*(f1_y-p_y)+(f2_z-f1_z)*(f1_z-p_z);
    const float e=(f3_x-f1_x)*(f1_x-p_x)+(f3_y-f1_y)*(f1_y-p_y)+(f3_z-f1_z)*(f1_z-p_z);
    const float f=(f1_x-p_x)*(f1_x-p_x)+(f1_y-p_y)*(f1_y-p_y)+(f1_z-p_z)*(f1_z-p_z);

    float det=a*c-b*b;
    float s=b*e-c*d;
    float t=b*d-a*e;

    if ( s + t < det )
    {
        if ( s < 0.f )
        {
            if ( t < 0.f )
            {
                if ( d < 0.f )
                {
                    s = clamp( -d/a, 0.f, 1.f );
                    t = 0.f;
                }
                else
                {
                    s = 0.f;
                    t = clamp( -e/c, 0.f, 1.f );
                }
            }
            else
            {
                s = 0.f;
                t = clamp( -e/c, 0.f, 1.f );
            }
        }
        else if ( t < 0.f )
        {
            s = clamp( -d/a, 0.f, 1.f );
            t = 0.f;
        }
        else
        {
            float invDet = 1.f / det;
            s *= invDet;
            t *= invDet;
        }
    }
    else
    {
        if ( s < 0.f )
        {
            float tmp0 = b+d;
            float tmp1 = c+e;
            if ( tmp1 > tmp0 )
            {
                float numer = tmp1 - tmp0;
                float denom = a-2*b+c;
                s = clamp( numer/denom, 0.f, 1.f );
                t = 1-s;
            }
            else
            {
                t = clamp( -e/c, 0.f, 1.f );
                s = 0.f;
            }
        }
        else if ( t < 0.f )
        {
            if ( a+d > b+e )
            {
                float numer = c+e-b-d;
                float denom = a-2*b+c;
                s = clamp( numer/denom, 0.f, 1.f );
                t = 1-s;
            }
            else
            {
                s = clamp( -e/c, 0.f, 1.f );
                t = 0.f;
            }
        }
        else
        {
            float numer = c+e-b-d;
            float denom = a-2*b+c;
            s = clamp( numer/denom, 0.f, 1.f );
            t = 1.f - s;
        }
    }
    result_s=s;
    result_t=t;
    result_dist=a*s*s+2*b*s*t+c*t*t+2*d*s+2*e*t+f;
    
    //return 1;
}



__global__ void closestPointonSurface_kernel(
    int n,      //number of points
    const float* points,
    int m,      //number of triangles
    const float* f1,
    const float* f2,
    const float* f3,
    float* w1,
    float* w2,
    float* w3,
    int* indexes
)
{
    const int batch=1024;
    float dist_temp;
    float s_temp;
    float t_temp;

    __shared__ float f1_buff[batch*3];
    __shared__ float f2_buff[batch*3];
    __shared__ float f3_buff[batch*3];

    float f1_x;
    float f1_y;
    float f1_z;
    float f2_x;
    float f2_y;
    float f2_z;
    float f3_x;
    float f3_y;
    float f3_z;

    int  tid=threadIdx.x+blockIdx.x*blockDim.x;
    for (int i =tid;i<n;i+=gridDim.x)
    {
        float best_dist=1e10;
        int best_idx;
        float best_s;
        float best_t;
        const float p_x=points[i*3+0];
        const float p_y=points[i*3+1];
        const float p_z=points[i*3+2];

        for (int start=0;start<m;start+=batch)
        {
            int end = min(start+batch,m);
            //copy the batch data to the shared memory
            for (int j=threadIdx.x;j<(end-start);j+=blockDim.x)
            {
                f1_buff[j*3+0]=f1[(start+j)*3+0];
                f1_buff[j*3+1]=f1[(start+j)*3+1];
                f1_buff[j*3+2]=f1[(start+j)*3+2];
                f2_buff[j*3+0]=f2[(start+j)*3+0];
                f2_buff[j*3+1]=f2[(start+j)*3+1];
                f2_buff[j*3+2]=f2[(start+j)*3+2];
                f3_buff[j*3+0]=f3[(start+j)*3+0];
                f3_buff[j*3+1]=f3[(start+j)*3+1];
                f3_buff[j*3+2]=f3[(start+j)*3+2];
            }
            __syncthreads();

            for (int j=0;j<(end-start);j++)
            {
                f1_x=f1_buff[j*3+0];
                f1_y=f1_buff[j*3+1];
                f1_z=f1_buff[j*3+2];
                f2_x=f2_buff[j*3+0];
                f2_y=f2_buff[j*3+1];
                f2_z=f2_buff[j*3+2];
                f3_x=f3_buff[j*3+0];
                f3_y=f3_buff[j*3+1];
                f3_z=f3_buff[j*3+2];
                
                closestPointOnTriangle(p_x,p_y,p_z,
                f1_x,f1_y,f1_z,
                f2_x,f2_y,f2_z,
                f3_x,f3_y,f3_z,
                dist_temp,s_temp,t_temp);
                if (best_dist>=dist_temp)
                {
                    best_dist=dist_temp;
                    best_idx=j+start;
                    best_s=s_temp;
                    best_t=t_temp;
                }
            }
            __syncthreads();
        }

        w1[i]=1-best_s-best_t;
        w2[i]=best_s;
        w3[i]=best_t;
        indexes[i]=best_idx;
    }
    
}




std::vector<at::Tensor> closestPointonSurface_cuda_forward(
    at::Tensor points,
    at::Tensor f1,
    at::Tensor f2,
    at::Tensor f3
)
{
    const int n=points.size(0);
    const int m=f1.size(0);
    //printf("%d",m);
    at::Tensor w1=torch::zeros({n},torch::CUDA(torch::kFloat));
    at::Tensor w2=torch::zeros({n},torch::CUDA(torch::kFloat));
    at::Tensor w3=torch::zeros({n},torch::CUDA(torch::kFloat));
    at::Tensor indexes=torch::zeros({n},torch::CUDA(torch::kInt));

    closestPointonSurface_kernel<<<32768,1024>>>(
        n,points.data_ptr<float>(),m,
        f1.data_ptr<float>(),f2.data_ptr<float>(),f3.data_ptr<float>(),
        w1.data_ptr<float>(),w2.data_ptr<float>(),w3.data_ptr<float>(),
        indexes.data_ptr<int>()
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
        printf("Error in closestPointonSurface_cuda_forward: %s\n", cudaGetErrorString(err));
    }

    return {indexes,w1,w2,w3};

}