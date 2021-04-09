# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import torch


class HarmonicEmbedding(torch.nn.Module):
    def __init__(
        self,
        n_harmonic_functions: int = 6,
        omega0: float = 1.0,
        logspace: bool = True,
    ):
        """
        Given an input tensor `x` of shape [minibatch, ... , dim],
        the harmonic embedding layer converts each feature
        in `x` into a series of harmonic features `embedding`
        as follows:
            ```
            embedding[..., i*dim:(i+1)*dim] = [
                sin(x[..., i]),
                sin(f_1*x[..., i]),
                sin(f_2*x[..., i]),
                ...
                sin(f_N * x[..., i]),
                cos(x[..., i]),
                cos(f_1*x[..., i]),
                cos(f_2*x[..., i]),
                ...
                cos(f_N * x[..., i])
            ]
            ```
        where N corresponds to `n_harmonic_functions`, and f_i is a scalar
        denoting the i-th frequency of the harmonic embedding.

        If `logspace==True`, the frequencies `[f_1, ..., f_N]` are
        either powers of 2:
            `f_1, ..., f_N = 2**torch.arange(n_harmonic_functions)`

        If `logspace==False`, frequencies are linearly spaced  between
        `1.0` and `2**(n_harmonic_functions-1)`:
            `f_1, ..., f_N = torch.linspace(
                1.0, 2**(n_harmonic_functions-1), n_harmonic_functions
            )`

        Note that `x` is also premultiplied by the base frequency `omega0`
        before evaluating the harmonic functions.
        """
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                n_harmonic_functions,
                dtype=torch.float32,
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        self.register_buffer("_frequencies", omega0 * frequencies)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: tensor of shape [..., dim]
        Returns:
            embedding: a harmonic embedding of `x`
                of shape [..., n_harmonic_functions * dim * 2]
        """
        embed = (x[..., None] * self._frequencies).view(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)
