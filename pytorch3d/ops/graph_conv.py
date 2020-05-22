# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import torch
import torch.nn as nn
from pytorch3d import _C
from torch.autograd import Function
from torch.autograd.function import once_differentiable


class GraphConv(nn.Module):
    """A single graph convolution layer."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        init: str = "normal",
        directed: bool = False,
    ):
        """
        Args:
            input_dim: Number of input features per vertex.
            output_dim: Number of output features per vertex.
            init: Weight initialization method. Can be one of ['zero', 'normal'].
            directed: Bool indicating if edges in the graph are directed.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.directed = directed
        self.w0 = nn.Linear(input_dim, output_dim)
        self.w1 = nn.Linear(input_dim, output_dim)

        if init == "normal":
            nn.init.normal_(self.w0.weight, mean=0, std=0.01)
            nn.init.normal_(self.w1.weight, mean=0, std=0.01)
            # pyre-fixme[16]: Optional type has no attribute `data`.
            self.w0.bias.data.zero_()
            self.w1.bias.data.zero_()
        elif init == "zero":
            self.w0.weight.data.zero_()
            self.w1.weight.data.zero_()
        else:
            raise ValueError('Invalid GraphConv initialization "%s"' % init)

    def forward(self, verts, edges):
        """
        Args:
            verts: FloatTensor of shape (V, input_dim) where V is the number of
                vertices and input_dim is the number of input features
                per vertex. input_dim has to match the input_dim specified
                in __init__.
            edges: LongTensor of shape (E, 2) where E is the number of edges
                where each edge has the indices of the two vertices which
                form the edge.

        Returns:
            out: FloatTensor of shape (V, output_dim) where output_dim is the
            number of output features per vertex.
        """
        if verts.is_cuda != edges.is_cuda:
            raise ValueError("verts and edges tensors must be on the same device.")
        if verts.shape[0] == 0:
            # empty graph.
            return verts.new_zeros((0, self.output_dim)) * verts.sum()

        verts_w0 = self.w0(verts)  # (V, output_dim)
        verts_w1 = self.w1(verts)  # (V, output_dim)

        if torch.cuda.is_available() and verts.is_cuda and edges.is_cuda:
            neighbor_sums = gather_scatter(verts_w1, edges, self.directed)
        else:
            neighbor_sums = gather_scatter_python(
                verts_w1, edges, self.directed
            )  # (V, output_dim)

        # Add neighbor features to each vertex's features.
        out = verts_w0 + neighbor_sums
        return out

    def __repr__(self):
        Din, Dout, directed = self.input_dim, self.output_dim, self.directed
        return "GraphConv(%d -> %d, directed=%r)" % (Din, Dout, directed)


def gather_scatter_python(input, edges, directed: bool = False):
    """
    Python implementation of gather_scatter for aggregating features of
    neighbor nodes in a graph.

    Given a directed graph: v0 -> v1 -> v2 the updated feature for v1 depends
    on v2 in order to be consistent with Morris et al. AAAI 2019
    (https://arxiv.org/abs/1810.02244). This only affects
    directed graphs; for undirected graphs v1 will depend on both v0 and v2,
    no matter which way the edges are physically stored.

    Args:
        input: Tensor of shape (num_vertices, input_dim).
        edges: Tensor of edge indices of shape (num_edges, 2).
        directed: bool indicating if edges are directed.

    Returns:
        output: Tensor of same shape as input.
    """
    if not (input.dim() == 2):
        raise ValueError("input can only have 2 dimensions.")
    if not (edges.dim() == 2):
        raise ValueError("edges can only have 2 dimensions.")
    if not (edges.shape[1] == 2):
        raise ValueError("edges must be of shape (num_edges, 2).")

    num_vertices, input_feature_dim = input.shape
    num_edges = edges.shape[0]
    output = torch.zeros_like(input)
    idx0 = edges[:, 0].view(num_edges, 1).expand(num_edges, input_feature_dim)
    idx1 = edges[:, 1].view(num_edges, 1).expand(num_edges, input_feature_dim)

    # pyre-fixme[16]: `Tensor` has no attribute `scatter_add`.
    output = output.scatter_add(0, idx0, input.gather(0, idx1))
    if not directed:
        output = output.scatter_add(0, idx1, input.gather(0, idx0))
    return output


class GatherScatter(Function):
    """
    Torch autograd Function wrapper for gather_scatter C++/CUDA implementations.
    """

    @staticmethod
    def forward(ctx, input, edges, directed=False):
        """
        Args:
            ctx: Context object used to calculate gradients.
            input: Tensor of shape (num_vertices, input_dim)
            edges: Tensor of edge indices of shape (num_edges, 2)
            directed: Bool indicating if edges are directed.

        Returns:
            output: Tensor of same shape as input.
        """
        if not (input.dim() == 2):
            raise ValueError("input can only have 2 dimensions.")
        if not (edges.dim() == 2):
            raise ValueError("edges can only have 2 dimensions.")
        if not (edges.shape[1] == 2):
            raise ValueError("edges must be of shape (num_edges, 2).")
        if not (input.dtype == torch.float32):
            raise ValueError("input has to be of type torch.float32.")

        ctx.directed = directed
        input, edges = input.contiguous(), edges.contiguous()
        ctx.save_for_backward(edges)
        backward = False
        output = _C.gather_scatter(input, edges, directed, backward)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        edges = ctx.saved_tensors[0]
        directed = ctx.directed
        backward = True
        grad_input = _C.gather_scatter(grad_output, edges, directed, backward)
        grad_edges = None
        grad_directed = None
        return grad_input, grad_edges, grad_directed


# pyre-fixme[16]: `GatherScatter` has no attribute `apply`.
gather_scatter = GatherScatter.apply
