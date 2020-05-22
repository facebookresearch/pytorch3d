#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


from typing import NamedTuple

import torch
from pytorch3d import _C


# Example functions for blending the top K features per pixel using the outputs
# from rasterization.
# NOTE: All blending function should return a (N, H, W, C) tensor per batch element.
# This can be an image (C=3) or a set of features.


# Data class to store blending params with defaults
class CompositeParams(NamedTuple):
    radius: float = 4.0 / 256.0


class _CompositeAlphaPoints(torch.autograd.Function):
    """
    Composite features within a z-buffer using alpha compositing. Given a z-buffer
    with corresponding features and weights, these values are accumulated according
    to their weights such that features nearer in depth contribute more to the final
    feature than ones further away.

    Concretely this means:
        weighted_fs[b,c,i,j] = sum_k cum_alpha_k * features[c,pointsidx[b,k,i,j]]
        cum_alpha_k = alphas[b,k,i,j] * prod_l=0..k-1 (1 - alphas[b,l,i,j])

    Args:
        features: Packed Tensor of shape (C, P) giving the features of each point.
        alphas: float32 Tensor of shape (N, points_per_pixel, image_size,
            image_size) giving the weight of each point in the z-buffer.
            Values should be in the interval [0, 1].
        pointsidx: int32 Tensor of shape (N, points_per_pixel, image_size, image_size)
            giving the indices of the nearest points at each pixel, sorted in z-order.
            Concretely pointsidx[n, k, y, x] = p means that features[:, p] is the
            feature of the kth closest point (along the z-direction) to pixel (y, x) in
            batch element n. This is weighted by alphas[n, k, y, x].

    Returns:
        weighted_fs: Tensor of shape (N, C, image_size, image_size)
            giving the accumulated features at each point.
    """

    @staticmethod
    def forward(ctx, features, alphas, points_idx):
        pt_cld = _C.accum_alphacomposite(features, alphas, points_idx)

        ctx.save_for_backward(features.clone(), alphas.clone(), points_idx.clone())
        return pt_cld

    @staticmethod
    def backward(ctx, grad_output):
        grad_features = None
        grad_alphas = None
        grad_points_idx = None
        features, alphas, points_idx = ctx.saved_tensors

        grad_features, grad_alphas = _C.accum_alphacomposite_backward(
            grad_output, features, alphas, points_idx
        )

        return grad_features, grad_alphas, grad_points_idx, None


def alpha_composite(pointsidx, alphas, pt_clds, blend_params=None) -> torch.Tensor:
    """
    Composite features within a z-buffer using alpha compositing. Given a z-buffer
    with corresponding features and weights, these values are accumulated according
    to their weights such that features nearer in depth contribute more to the final
    feature than ones further away.

    Concretely this means:
        weighted_fs[b,c,i,j] = sum_k cum_alpha_k * features[c,pointsidx[b,k,i,j]]
        cum_alpha_k = alphas[b,k,i,j] * prod_l=0..k-1 (1 - alphas[b,l,i,j])


    Args:
        pt_clds: Tensor of shape (N, C, P) giving the features of each point (can use
            RGB for example).
        alphas: float32 Tensor of shape (N, points_per_pixel, image_size,
            image_size) giving the weight of each point in the z-buffer.
            Values should be in the interval [0, 1].
        pointsidx: int32 Tensor of shape (N, points_per_pixel, image_size, image_size)
            giving the indices of the nearest points at each pixel, sorted in z-order.
            Concretely pointsidx[n, k, y, x] = p means that features[n, :, p] is the
            feature of the kth closest point (along the z-direction) to pixel (y, x) in
            batch element n. This is weighted by alphas[n, k, y, x].

    Returns:
        Combined features: Tensor of shape (N, C, image_size, image_size)
            giving the accumulated features at each point.
    """
    # pyre-fixme[16]: `_CompositeAlphaPoints` has no attribute `apply`.
    return _CompositeAlphaPoints.apply(pt_clds, alphas, pointsidx)


class _CompositeNormWeightedSumPoints(torch.autograd.Function):
    """
    Composite features within a z-buffer using normalized weighted sum. Given a z-buffer
    with corresponding features and weights, these values are accumulated
    according to their weights such that depth is ignored; the weights are used to
    perform a weighted sum.

    Concretely this means:
        weighted_fs[b,c,i,j] =
         sum_k alphas[b,k,i,j] * features[c,pointsidx[b,k,i,j]] / sum_k alphas[b,k,i,j]

    Args:
        features: Packed Tensor of shape (C, P) giving the features of each point.
        alphas: float32 Tensor of shape (N, points_per_pixel, image_size,
            image_size) giving the weight of each point in the z-buffer.
            Values should be in the interval [0, 1].
        pointsidx: int32 Tensor of shape (N, points_per_pixel, image_size, image_size)
            giving the indices of the nearest points at each pixel, sorted in z-order.
            Concretely pointsidx[n, k, y, x] = p means that features[:, p] is the
            feature of the kth closest point (along the z-direction) to pixel (y, x) in
            batch element n. This is weighted by alphas[n, k, y, x].

    Returns:
        weighted_fs: Tensor of shape (N, C, image_size, image_size)
            giving the accumulated features at each point.
    """

    @staticmethod
    def forward(ctx, features, alphas, points_idx):
        pt_cld = _C.accum_weightedsumnorm(features, alphas, points_idx)

        ctx.save_for_backward(features.clone(), alphas.clone(), points_idx.clone())
        return pt_cld

    @staticmethod
    def backward(ctx, grad_output):
        grad_features = None
        grad_alphas = None
        grad_points_idx = None
        features, alphas, points_idx = ctx.saved_tensors

        grad_features, grad_alphas = _C.accum_weightedsumnorm_backward(
            grad_output, features, alphas, points_idx
        )

        return grad_features, grad_alphas, grad_points_idx, None


def norm_weighted_sum(pointsidx, alphas, pt_clds, blend_params=None) -> torch.Tensor:
    """
    Composite features within a z-buffer using normalized weighted sum. Given a z-buffer
    with corresponding features and weights, these values are accumulated
    according to their weights such that depth is ignored; the weights are used to
    perform a weighted sum.

    Concretely this means:
        weighted_fs[b,c,i,j] =
         sum_k alphas[b,k,i,j] * features[c,pointsidx[b,k,i,j]] / sum_k alphas[b,k,i,j]

    Args:
        pt_clds: Packed feature tensor of shape (C, P) giving the features of each point
            (can use RGB for example).
        alphas: float32 Tensor of shape (N, points_per_pixel, image_size,
            image_size) giving the weight of each point in the z-buffer.
            Values should be in the interval [0, 1].
        pointsidx: int32 Tensor of shape (N, points_per_pixel, image_size, image_size)
            giving the indices of the nearest points at each pixel, sorted in z-order.
            Concretely pointsidx[n, k, y, x] = p means that features[:, p] is the
            feature of the kth closest point (along the z-direction) to pixel (y, x) in
            batch element n. This is weighted by alphas[n, k, y, x].

    Returns:
        Combined features: Tensor of shape (N, C, image_size, image_size)
            giving the accumulated features at each point.
    """
    # pyre-fixme[16]: `_CompositeNormWeightedSumPoints` has no attribute `apply`.
    return _CompositeNormWeightedSumPoints.apply(pt_clds, alphas, pointsidx)


class _CompositeWeightedSumPoints(torch.autograd.Function):
    """
    Composite features within a z-buffer using normalized weighted sum. Given a z-buffer
    with corresponding features and weights, these values are accumulated
    according to their weights such that depth is ignored; the weights are used to
    perform a weighted sum. As opposed to norm weighted sum, the weights are not
    normalized to sum to 1.

    Concretely this means:
        weighted_fs[b,c,i,j] = sum_k alphas[b,k,i,j] * features[c,pointsidx[b,k,i,j]]

    Args:
        features: Packed Tensor of shape (C, P) giving the features of each point.
        alphas: float32 Tensor of shape (N, points_per_pixel, image_size,
            image_size) giving the weight of each point in the z-buffer.
            Values should be in the interval [0, 1].
        pointsidx: int32 Tensor of shape (N, points_per_pixel, image_size, image_size)
            giving the indices of the nearest points at each pixel, sorted in z-order.
            Concretely pointsidx[n, k, y, x] = p means that features[:, p] is the
            feature of the kth closest point (along the z-direction) to pixel (y, x) in
            batch element n. This is weighted by alphas[n, k, y, x].

    Returns:
        weighted_fs: Tensor of shape (N, C, image_size, image_size)
            giving the accumulated features at each point.
    """

    @staticmethod
    def forward(ctx, features, alphas, points_idx):
        pt_cld = _C.accum_weightedsum(features, alphas, points_idx)

        ctx.save_for_backward(features.clone(), alphas.clone(), points_idx.clone())
        return pt_cld

    @staticmethod
    def backward(ctx, grad_output):
        grad_features = None
        grad_alphas = None
        grad_points_idx = None
        features, alphas, points_idx = ctx.saved_tensors

        grad_features, grad_alphas = _C.accum_weightedsum_backward(
            grad_output, features, alphas, points_idx
        )

        return grad_features, grad_alphas, grad_points_idx, None


def weighted_sum(pointsidx, alphas, pt_clds, blend_params=None) -> torch.Tensor:
    """
    Composite features within a z-buffer using normalized weighted sum.

    Args:
        pt_clds: Packed Tensor of shape (C, P) giving the features of each point
            (can use RGB for example).
        alphas: float32 Tensor of shape (N, points_per_pixel, image_size,
            image_size) giving the weight of each point in the z-buffer.
            Values should be in the interval [0, 1].
        pointsidx: int32 Tensor of shape (N, points_per_pixel, image_size, image_size)
            giving the indices of the nearest points at each pixel, sorted in z-order.
            Concretely pointsidx[n, k, y, x] = p means that features[:, p] is the
            feature of the kth closest point (along the z-direction) to pixel (y, x) in
            batch element n. This is weighted by alphas[n, k, y, x].

    Returns:
        Combined features: Tensor of shape (N, C, image_size, image_size)
            giving the accumulated features at each point.
    """
    # pyre-fixme[16]: `_CompositeWeightedSumPoints` has no attribute `apply`.
    return _CompositeWeightedSumPoints.apply(pt_clds, alphas, pointsidx)
