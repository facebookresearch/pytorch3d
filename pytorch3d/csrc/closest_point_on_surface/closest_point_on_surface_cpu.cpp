#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor> closestPointonSurface_cuda_forward(
    at::Tensor points,
    at::Tensor f1,
    at::Tensor f2,
    at::Tensor f3
);

std::vector<at::Tensor> closestPointonSurface_forward(
    at::Tensor points,
    at::Tensor f1,
    at::Tensor f2,
    at::Tensor f3
    ) 
{
    return closestPointonSurface_cuda_forward(points,f1,f2,f3);
}



//PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
//{
//  m.def("forward", &closestPointonSurface_forward, "forward (CUDA)");
  //m.def("backward", &closest_point_on_surface_backward, "backward (CUDA)");
//}