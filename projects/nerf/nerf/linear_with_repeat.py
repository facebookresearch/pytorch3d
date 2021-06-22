# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
import torch.nn.functional as F


class LinearWithRepeat(torch.nn.Linear):
    """
    if x has shape (..., k, n1)
    and y has shape (..., n2)
    then
    LinearWithRepeat(n1 + n2, out_features).forward((x,y))
    is equivalent to
    Linear(n1 + n2, out_features).forward(
        torch.cat([x, y.unsqueeze(-2).expand(..., k, n2)], dim=-1)
    )

    Or visually:
    Given the following, for each ray,

                feature   ->

    ray         xxxxxxxx
    position    xxxxxxxx
      |         xxxxxxxx
      v         xxxxxxxx


    and
                            yyyyyyyy

    where the y's do not depend on the position
    but only on the ray,
    we want to evaluate a Linear layer on both
    types of data at every position.

    It's as if we constructed

                xxxxxxxxyyyyyyyy
                xxxxxxxxyyyyyyyy
                xxxxxxxxyyyyyyyy
                xxxxxxxxyyyyyyyy

    and sent that through the Linear.
    """

    def forward(self, input: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        n1 = input[0].shape[-1]
        output1 = F.linear(input[0], self.weight[:, :n1], self.bias)
        output2 = F.linear(input[1], self.weight[:, n1:], None)
        return output1 + output2.unsqueeze(-2)
