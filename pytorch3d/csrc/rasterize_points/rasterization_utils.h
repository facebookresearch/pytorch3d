/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// The default value of the NDC range is [-1, 1], however in the case that
// H != W, the NDC range is set such that the shorter side has range [-1, 1] and
// the longer side is scaled by the ratio of H:W. S1 is the dimension for which
// the NDC range is calculated and S2 is the other image dimension.
// e.g. to get the NDC x range S1 = W and S2 = H
inline float NonSquareNdcRange(int S1, int S2) {
  float range = 2.0f;
  if (S1 > S2) {
    range = (S1 * range) / S2;
  }
  return range;
}

// Given a pixel coordinate 0 <= i < S1, convert it to a normalized device
// coordinates. We divide the NDC range into S1 evenly-sized
// pixels, and assume that each pixel falls in the *center* of its range.
// The default value of the NDC range is [-1, 1], however in the case that
// H != W, the NDC range is set such that the shorter side has range [-1, 1] and
// the longer side is scaled by the ratio of H:W. The dimension of i should be
// S1 and the other image dimension is S2 For example, to get the x and y NDC
// coordinates or a given pixel i:
//     x = PixToNonSquareNdc(i, W, H)
//     y = PixToNonSquareNdc(i, H, W)
inline float PixToNonSquareNdc(int i, int S1, int S2) {
  float range = NonSquareNdcRange(S1, S2);
  // NDC: offset + (i * pixel_width + half_pixel_width)
  // The NDC range is [-range/2, range/2].
  const float offset = (range / 2.0f);
  return -offset + (range * i + offset) / S1;
}
