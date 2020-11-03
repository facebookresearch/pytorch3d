// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#ifndef PULSAR_NATIVE_PYTORCH_TENSOR_UTIL_H_
#define PULSAR_NATIVE_PYTORCH_TENSOR_UTIL_H_

#include <ATen/ATen.h>

namespace pulsar {
namespace pytorch {

torch::Tensor sphere_ids_from_result_info_nograd(
    const torch::Tensor& forw_info);

}
} // namespace pulsar

#endif
