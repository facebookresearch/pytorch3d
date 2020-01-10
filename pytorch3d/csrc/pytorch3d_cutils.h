// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <torch/extension.h>

#define CHECK_CUDA(x) \
  AT_ASSERTM(x.type().is_cuda(), #x "must be a CUDA tensor.")
#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM(x.is_contiguous(), #x "must be contiguous.")
#define CHECK_CONTIGUOUS_CUDA(x) \
  CHECK_CUDA(x);                 \
  CHECK_CONTIGUOUS(x)
