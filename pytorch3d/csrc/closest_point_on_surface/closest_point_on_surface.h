#pragma once
#include <torch/extension.h>
#include <vector>

#include "utils/pytorch3d_cutils.h"


//std::vector<at::Tensor> closestPointonSurface_cuda_forward(
 //   at::Tensor points,
 //   at::Tensor f1,
 //   at::Tensor f2,
 //   at::Tensor f3
//);

std::vector<at::Tensor> closestPointonSurface_forward(
    at::Tensor points,
    at::Tensor f1,
    at::Tensor f2,
    at::Tensor f3
    );