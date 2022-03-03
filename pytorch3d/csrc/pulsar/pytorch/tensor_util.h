/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
