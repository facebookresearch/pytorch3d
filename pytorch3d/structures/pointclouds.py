# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch

from .. import ops
from . import utils as struct_utils


class Pointclouds(object):
    """
    This class provides functions for working with batches of 3d point clouds,
    and converting between representations.

    Within Pointclouds, there are three different representations of the data.

    List
       - only used for input as a starting point to convert to other representations.
    Padded
       - has specific batch dimension.
    Packed
       - no batch dimension.
       - has auxillary variables used to index into the padded representation.

    Example

    Input list of points = [[P_1], [P_2], ... , [P_N]]
    where P_1, ... , P_N are the number of points in each cloud and N is the
    number of clouds.

    # SPHINX IGNORE
     List                      | Padded                  | Packed
    ---------------------------|-------------------------|------------------------
    [[P_1], ... , [P_N]]       | size = (N, max(P_n), 3) |  size = (sum(P_n), 3)
                               |                         |
    Example for locations      |                         |
    or colors:                 |                         |
                               |                         |
    P_1 = 3, P_2 = 4, P_3 = 5  | size = (3, 5, 3)        |  size = (12, 3)
                               |                         |
    List([                     | tensor([                |  tensor([
      [                        |     [                   |    [0.1, 0.3, 0.5],
        [0.1, 0.3, 0.5],       |       [0.1, 0.3, 0.5],  |    [0.5, 0.2, 0.1],
        [0.5, 0.2, 0.1],       |       [0.5, 0.2, 0.1],  |    [0.6, 0.8, 0.7],
        [0.6, 0.8, 0.7]        |       [0.6, 0.8, 0.7],  |    [0.1, 0.3, 0.3],
      ],                       |       [0,    0,    0],  |    [0.6, 0.7, 0.8],
      [                        |       [0,    0,    0]   |    [0.2, 0.3, 0.4],
        [0.1, 0.3, 0.3],       |     ],                  |    [0.1, 0.5, 0.3],
        [0.6, 0.7, 0.8],       |     [                   |    [0.7, 0.3, 0.6],
        [0.2, 0.3, 0.4],       |       [0.1, 0.3, 0.3],  |    [0.2, 0.4, 0.8],
        [0.1, 0.5, 0.3]        |       [0.6, 0.7, 0.8],  |    [0.9, 0.5, 0.2],
      ],                       |       [0.2, 0.3, 0.4],  |    [0.2, 0.3, 0.4],
      [                        |       [0.1, 0.5, 0.3],  |    [0.9, 0.3, 0.8],
        [0.7, 0.3, 0.6],       |       [0,    0,    0]   |  ])
        [0.2, 0.4, 0.8],       |     ],                  |
        [0.9, 0.5, 0.2],       |     [                   |
        [0.2, 0.3, 0.4],       |       [0.7, 0.3, 0.6],  |
        [0.9, 0.3, 0.8],       |       [0.2, 0.4, 0.8],  |
      ]                        |       [0.9, 0.5, 0.2],  |
   ])                          |       [0.2, 0.3, 0.4],  |
                               |       [0.9, 0.3, 0.8]   |
                               |     ]                   |
                               |  ])                     |
    -----------------------------------------------------------------------------

    Auxillary variables for packed representation

    Name                           |   Size              |  Example from above
    -------------------------------|---------------------|-----------------------
                                   |                     |
    packed_to_cloud_idx            |  size = (sum(P_n))  |   tensor([
                                   |                     |     0, 0, 0, 1, 1, 1,
                                   |                     |     1, 2, 2, 2, 2, 2
                                   |                     |   )]
                                   |                     |   size = (12)
                                   |                     |
    cloud_to_packed_first_idx      |  size = (N)         |   tensor([0, 3, 7])
                                   |                     |   size = (3)
                                   |                     |
    num_points_per_cloud           |  size = (N)         |   tensor([3, 4, 5])
                                   |                     |   size = (3)
                                   |                     |
    padded_to_packed_idx           |  size = (sum(P_n))  |  tensor([
                                   |                     |     0, 1, 2, 5, 6, 7,
                                   |                     |     8, 10, 11, 12, 13,
                                   |                     |     14
                                   |                     |  )]
                                   |                     |  size = (12)
    -----------------------------------------------------------------------------
    # SPHINX IGNORE
    """

    _INTERNAL_TENSORS = [
        "_points_packed",
        "_points_padded",
        "_normals_packed",
        "_normals_padded",
        "_features_packed",
        "_features_padded",
        "_packed_to_cloud_idx",
        "_cloud_to_packed_first_idx",
        "_num_points_per_cloud",
        "_padded_to_packed_idx",
        "valid",
        "equisized",
    ]

    def __init__(self, points, normals=None, features=None):
        """
        Args:
            points:
                Can be either

                - List where each element is a tensor of shape (num_points, 3)
                  containing the (x, y, z) coordinates of each point.
                - Padded float tensor with shape (num_clouds, num_points, 3).
            normals:
                Can be either

                - List where each element is a tensor of shape (num_points, 3)
                  containing the normal vector for each point.
                - Padded float tensor of shape (num_clouds, num_points, 3).
            features:
                Can be either

                - List where each element is a tensor of shape (num_points, C)
                  containing the features for the points in the cloud.
                - Padded float tensor of shape (num_clouds, num_points, C).
                where C is the number of channels in the features.
                For example 3 for RGB color.

        Refer to comments above for descriptions of List and Padded
        representations.
        """
        self.device = None

        # Indicates whether the clouds in the list/batch have the same number
        # of points.
        self.equisized = False

        # Boolean indicator for each cloud in the batch.
        # True if cloud has non zero number of points, False otherwise.
        self.valid = None

        self._N = 0  # batch size (number of clouds)
        self._P = 0  # (max) number of points per cloud
        self._C = None  # number of channels in the features

        # List of Tensors of points and features.
        self._points_list = None
        self._normals_list = None
        self._features_list = None

        # Number of points per cloud.
        self._num_points_per_cloud = None  # N

        # Packed representation.
        self._points_packed = None  # (sum(P_n), 3)
        self._normals_packed = None  # (sum(P_n), 3)
        self._features_packed = None  # (sum(P_n), C)

        self._packed_to_cloud_idx = None  # sum(P_n)

        # Index of each cloud's first point in the packed points.
        # Assumes packing is sequential.
        self._cloud_to_packed_first_idx = None  # N

        # Padded representation.
        self._points_padded = None  # (N, max(P_n), 3)
        self._normals_padded = None  # (N, max(P_n), 3)
        self._features_padded = None  # (N, max(P_n), C)

        # Index to convert points from flattened padded to packed.
        self._padded_to_packed_idx = None  # N * max_P

        # Identify type of points.
        if isinstance(points, list):
            self._points_list = points
            self._N = len(self._points_list)
            self.device = torch.device("cpu")
            self.valid = torch.zeros((self._N,), dtype=torch.bool, device=self.device)
            self._num_points_per_cloud = []

            if self._N > 0:
                for p in self._points_list:
                    if len(p) > 0 and (p.dim() != 2 or p.shape[1] != 3):
                        raise ValueError("Clouds in list must be of shape Px3 or empty")

                self.device = self._points_list[0].device
                num_points_per_cloud = torch.tensor(
                    [len(p) for p in self._points_list], device=self.device
                )
                self._P = num_points_per_cloud.max()
                self.valid = torch.tensor(
                    [len(p) > 0 for p in self._points_list],
                    dtype=torch.bool,
                    device=self.device,
                )

                if len(num_points_per_cloud.unique()) == 1:
                    self.equisized = True
                self._num_points_per_cloud = num_points_per_cloud

        elif torch.is_tensor(points):
            if points.dim() != 3 or points.shape[2] != 3:
                raise ValueError("Points tensor has incorrect dimensions.")
            self._points_padded = points
            self._N = self._points_padded.shape[0]
            self._P = self._points_padded.shape[1]
            self.device = self._points_padded.device
            self.valid = torch.ones((self._N,), dtype=torch.bool, device=self.device)
            self._num_points_per_cloud = torch.tensor(
                [self._P] * self._N, device=self.device
            )
            self.equisized = True
        else:
            raise ValueError(
                "Points must be either a list or a tensor with \
                    shape (batch_size, P, 3) where P is the maximum number of \
                    points in a cloud."
            )

        # parse normals
        normals_parsed = self._parse_auxiliary_input(normals)
        self._normals_list, self._normals_padded, normals_C = normals_parsed
        if normals_C is not None and normals_C != 3:
            raise ValueError("Normals are expected to be 3-dimensional")

        # parse features
        features_parsed = self._parse_auxiliary_input(features)
        self._features_list, self._features_padded, features_C = features_parsed
        if features_C is not None:
            self._C = features_C

    def _parse_auxiliary_input(self, aux_input):
        """
        Interpret the auxiliary inputs (normals, features) given to __init__.

        Args:
            aux_input:
              Can be either

                - List where each element is a tensor of shape (num_points, C)
                  containing the features for the points in the cloud.
                - Padded float tensor of shape (num_clouds, num_points, C).
              For normals, C = 3

        Returns:
            3-element tuple of list, padded, num_channels.
            If aux_input is list, then padded is None. If aux_input is a tensor,
            then list is None.
        """
        if aux_input is None or self._N == 0:
            return None, None, None

        aux_input_C = None

        if isinstance(aux_input, list):
            if len(aux_input) != self._N:
                raise ValueError("Points and auxiliary input must be the same length.")
            for p, d in zip(self._num_points_per_cloud, aux_input):
                if p != d.shape[0]:
                    raise ValueError(
                        "A cloud has mismatched numbers of points and inputs"
                    )
                if p > 0:
                    if d.dim() != 2:
                        raise ValueError(
                            "A cloud auxiliary input must be of shape PxC or empty"
                        )
                    if aux_input_C is None:
                        aux_input_C = d.shape[1]
                    if aux_input_C != d.shape[1]:
                        raise ValueError(
                            "The clouds must have the same number of channels"
                        )
            return aux_input, None, aux_input_C
        elif torch.is_tensor(aux_input):
            if aux_input.dim() != 3:
                raise ValueError("Auxiliary input tensor has incorrect dimensions.")
            if self._N != aux_input.shape[0]:
                raise ValueError("Points and inputs must be the same length.")
            if self._P != aux_input.shape[1]:
                raise ValueError(
                    "Inputs tensor must have the right maximum \
                    number of points in each cloud."
                )
            aux_input_C = aux_input.shape[2]
            return None, aux_input, aux_input_C
        else:
            raise ValueError(
                "Auxiliary input must be either a list or a tensor with \
                    shape (batch_size, P, C) where P is the maximum number of \
                    points in a cloud."
            )

    def __len__(self):
        return self._N

    def __getitem__(self, index):
        """
        Args:
            index: Specifying the index of the cloud to retrieve.
                Can be an int, slice, list of ints or a boolean tensor.

        Returns:
            Pointclouds object with selected clouds. The tensors are not cloned.
        """
        normals, features = None, None
        if isinstance(index, int):
            points = [self.points_list()[index]]
            if self.normals_list() is not None:
                normals = [self.normals_list()[index]]
            if self.features_list() is not None:
                features = [self.features_list()[index]]
        elif isinstance(index, slice):
            points = self.points_list()[index]
            if self.normals_list() is not None:
                normals = self.normals_list()[index]
            if self.features_list() is not None:
                features = self.features_list()[index]
        elif isinstance(index, list):
            points = [self.points_list()[i] for i in index]
            if self.normals_list() is not None:
                normals = [self.normals_list()[i] for i in index]
            if self.features_list() is not None:
                features = [self.features_list()[i] for i in index]
        elif isinstance(index, torch.Tensor):
            if index.dim() != 1 or index.dtype.is_floating_point:
                raise IndexError(index)
            # NOTE consider converting index to cpu for efficiency
            if index.dtype == torch.bool:
                # advanced indexing on a single dimension
                index = index.nonzero()
                index = index.squeeze(1) if index.numel() > 0 else index
                index = index.tolist()
            points = [self.points_list()[i] for i in index]
            if self.normals_list() is not None:
                normals = [self.normals_list()[i] for i in index]
            if self.features_list() is not None:
                features = [self.features_list()[i] for i in index]
        else:
            raise IndexError(index)

        return self.__class__(points=points, normals=normals, features=features)

    def isempty(self) -> bool:
        """
        Checks whether any cloud is valid.

        Returns:
            bool indicating whether there is any data.
        """
        return self._N == 0 or self.valid.eq(False).all()

    def points_list(self):
        """
        Get the list representation of the points.

        Returns:
            list of tensors of points of shape (P_n, 3).
        """
        if self._points_list is None:
            assert (
                self._points_padded is not None
            ), "points_padded is required to compute points_list."
            points_list = []
            for i in range(self._N):
                points_list.append(
                    self._points_padded[i, : self.num_points_per_cloud()[i]]
                )
            self._points_list = points_list
        return self._points_list

    def normals_list(self):
        """
        Get the list representation of the normals.

        Returns:
            list of tensors of normals of shape (P_n, 3).
        """
        if self._normals_list is None:
            if self._normals_padded is None:
                # No normals provided so return None
                return None
            self._normals_list = struct_utils.padded_to_list(
                self._normals_padded, self.num_points_per_cloud().tolist()
            )
        return self._normals_list

    def features_list(self):
        """
        Get the list representation of the features.

        Returns:
            list of tensors of features of shape (P_n, C).
        """
        if self._features_list is None:
            if self._features_padded is None:
                # No features provided so return None
                return None
            self._features_list = struct_utils.padded_to_list(
                self._features_padded, self.num_points_per_cloud().tolist()
            )
        return self._features_list

    def points_packed(self):
        """
        Get the packed representation of the points.

        Returns:
            tensor of points of shape (sum(P_n), 3).
        """
        self._compute_packed()
        return self._points_packed

    def normals_packed(self):
        """
        Get the packed representation of the normals.

        Returns:
            tensor of normals of shape (sum(P_n), 3).
        """
        self._compute_packed()
        return self._normals_packed

    def features_packed(self):
        """
        Get the packed representation of the features.

        Returns:
            tensor of features of shape (sum(P_n), C).
        """
        self._compute_packed()
        return self._features_packed

    def packed_to_cloud_idx(self):
        """
        Return a 1D tensor x with length equal to the total number of points.
        packed_to_cloud_idx()[i] gives the index of the cloud which contains
        points_packed()[i].

        Returns:
            1D tensor of indices.
        """
        self._compute_packed()
        return self._packed_to_cloud_idx

    def cloud_to_packed_first_idx(self):
        """
        Return a 1D tensor x with length equal to the number of clouds such that
        the first point of the ith cloud is points_packed[x[i]].

        Returns:
            1D tensor of indices of first items.
        """
        self._compute_packed()
        return self._cloud_to_packed_first_idx

    def num_points_per_cloud(self):
        """
        Return a 1D tensor x with length equal to the number of clouds giving
        the number of points in each cloud.

        Returns:
            1D tensor of sizes.
        """
        return self._num_points_per_cloud

    def points_padded(self):
        """
        Get the padded representation of the points.

        Returns:
            tensor of points of shape (N, max(P_n), 3).
        """
        self._compute_padded()
        return self._points_padded

    def normals_padded(self):
        """
        Get the padded representation of the normals.

        Returns:
            tensor of normals of shape (N, max(P_n), 3).
        """
        self._compute_padded()
        return self._normals_padded

    def features_padded(self):
        """
        Get the padded representation of the features.

        Returns:
            tensor of features of shape (N, max(P_n), 3).
        """
        self._compute_padded()
        return self._features_padded

    def padded_to_packed_idx(self):
        """
        Return a 1D tensor x with length equal to the total number of points
        such that points_packed()[i] is element x[i] of the flattened padded
        representation.
        The packed representation can be calculated as follows.

        .. code-block:: python

            p = points_padded().reshape(-1, 3)
            points_packed = p[x]

        Returns:
            1D tensor of indices.
        """
        self._compute_packed()
        if self._padded_to_packed_idx is not None:
            return self._padded_to_packed_idx
        if self._N == 0:
            self._padded_to_packed_idx = []
        else:
            self._padded_to_packed_idx = torch.cat(
                [
                    torch.arange(v, dtype=torch.int64, device=self.device) + i * self._P
                    for (i, v) in enumerate(self._num_points_per_cloud)
                ],
                dim=0,
            )
        return self._padded_to_packed_idx

    def _compute_padded(self, refresh: bool = False):
        """
        Computes the padded version from points_list, normals_list and features_list.

        Args:
            refresh: whether to force the recalculation.
        """
        if not (refresh or self._points_padded is None):
            return

        self._normals_padded, self._features_padded = None, None
        if self.isempty():
            self._points_padded = torch.zeros((self._N, 0, 3), device=self.device)
        else:
            self._points_padded = struct_utils.list_to_padded(
                self.points_list(),
                (self._P, 3),
                pad_value=0.0,
                equisized=self.equisized,
            )
            if self.normals_list() is not None:
                self._normals_padded = struct_utils.list_to_padded(
                    self.normals_list(),
                    (self._P, 3),
                    pad_value=0.0,
                    equisized=self.equisized,
                )
            if self.features_list() is not None:
                self._features_padded = struct_utils.list_to_padded(
                    self.features_list(),
                    (self._P, self._C),
                    pad_value=0.0,
                    equisized=self.equisized,
                )

    # TODO(nikhilar) Improve performance of _compute_packed.
    def _compute_packed(self, refresh: bool = False):
        """
        Computes the packed version from points_list, normals_list and
        features_list and sets the values of auxillary tensors.

        Args:
            refresh: Set to True to force recomputation of packed
                representations. Default: False.
        """

        if not (
            refresh
            or any(
                v is None
                for v in [
                    self._points_packed,
                    self._packed_to_cloud_idx,
                    self._cloud_to_packed_first_idx,
                ]
            )
        ):
            return

        # Packed can be calculated from padded or list, so can call the
        # accessor function for the lists.
        points_list = self.points_list()
        normals_list = self.normals_list()
        features_list = self.features_list()
        if self.isempty():
            self._points_packed = torch.zeros(
                (0, 3), dtype=torch.float32, device=self.device
            )
            self._packed_to_cloud_idx = torch.zeros(
                (0,), dtype=torch.int64, device=self.device
            )
            self._cloud_to_packed_first_idx = torch.zeros(
                (0,), dtype=torch.int64, device=self.device
            )
            self._normals_packed = None
            self._features_packed = None
            return

        points_list_to_packed = struct_utils.list_to_packed(points_list)
        self._points_packed = points_list_to_packed[0]
        if not torch.allclose(self._num_points_per_cloud, points_list_to_packed[1]):
            raise ValueError("Inconsistent list to packed conversion")
        self._cloud_to_packed_first_idx = points_list_to_packed[2]
        self._packed_to_cloud_idx = points_list_to_packed[3]

        self._normals_packed, self._features_packed = None, None
        if normals_list is not None:
            normals_list_to_packed = struct_utils.list_to_packed(normals_list)
            self._normals_packed = normals_list_to_packed[0]

        if features_list is not None:
            features_list_to_packed = struct_utils.list_to_packed(features_list)
            self._features_packed = features_list_to_packed[0]

    def clone(self):
        """
        Deep copy of Pointclouds object. All internal tensors are cloned
        individually.

        Returns:
            new Pointclouds object.
        """
        # instantiate new pointcloud with the representation which is not None
        # (either list or tensor) to save compute.
        new_points, new_normals, new_features = None, None, None
        if self._points_list is not None:
            new_points = [v.clone() for v in self.points_list()]
            normals_list = self.normals_list()
            features_list = self.features_list()
            if normals_list is not None:
                new_normals = [n.clone() for n in normals_list]
            if features_list is not None:
                new_features = [f.clone() for f in features_list]
        elif self._points_padded is not None:
            new_points = self.points_padded().clone()
            normals_padded = self.normals_padded()
            features_padded = self.features_padded()
            if normals_padded is not None:
                new_normals = self.normals_padded().clone()
            if features_padded is not None:
                new_features = self.features_padded().clone()
        other = self.__class__(
            points=new_points, normals=new_normals, features=new_features
        )
        for k in self._INTERNAL_TENSORS:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.clone())
        return other

    def to(self, device, copy: bool = False):
        """
        Match functionality of torch.Tensor.to()
        If copy = True or the self Tensor is on a different device, the
        returned tensor is a copy of self with the desired torch.device.
        If copy = False and the self Tensor already has the correct torch.device,
        then self is returned.

        Args:
          device: Device id for the new tensor.
          copy: Boolean indicator whether or not to clone self. Default False.

        Returns:
          Pointclouds object.
        """
        if not copy and self.device == device:
            return self
        other = self.clone()
        if self.device != device:
            other.device = device
            if other._N > 0:
                other._points_list = [v.to(device) for v in other.points_list()]
                if other._normals_list is not None:
                    other._normals_list = [n.to(device) for n in other.normals_list()]
                if other._features_list is not None:
                    other._features_list = [f.to(device) for f in other.features_list()]
            for k in self._INTERNAL_TENSORS:
                v = getattr(self, k)
                if torch.is_tensor(v):
                    setattr(other, k, v.to(device))
        return other

    def cpu(self):
        return self.to(torch.device("cpu"))

    def cuda(self):
        return self.to(torch.device("cuda"))

    def get_cloud(self, index: int):
        """
        Get tensors for a single cloud from the list representation.

        Args:
            index: Integer in the range [0, N).

        Returns:
            points: Tensor of shape (P, 3).
            normals: Tensor of shape (P, 3)
            features: LongTensor of shape (P, C).
        """
        if not isinstance(index, int):
            raise ValueError("Cloud index must be an integer.")
        if index < 0 or index > self._N:
            raise ValueError(
                "Cloud index must be in the range [0, N) where \
            N is the number of clouds in the batch."
            )
        points = self.points_list()[index]
        normals, features = None, None
        if self.normals_list() is not None:
            normals = self.normals_list()[index]
        if self.features_list() is not None:
            features = self.features_list()[index]
        return points, normals, features

    # TODO(nikhilar) Move function to a utils file.
    def split(self, split_sizes: list):
        """
        Splits Pointclouds object of size N into a list of Pointclouds objects
        of size len(split_sizes), where the i-th Pointclouds object is of size
        split_sizes[i]. Similar to torch.split().

        Args:
            split_sizes: List of integer sizes of Pointclouds objects to be
            returned.

        Returns:
            list[PointClouds].
        """
        if not all(isinstance(x, int) for x in split_sizes):
            raise ValueError("Value of split_sizes must be a list of integers.")
        cloudlist = []
        curi = 0
        for i in split_sizes:
            cloudlist.append(self[curi : curi + i])
            curi += i
        return cloudlist

    def offset_(self, offsets_packed):
        """
        Translate the point clouds by an offset. In place operation.

        Args:
            offsets_packed: A Tensor of the same shape as self.points_packed
                giving offsets to be added to all points.
        Returns:
            self.
        """
        points_packed = self.points_packed()
        if offsets_packed.shape != points_packed.shape:
            raise ValueError("Offsets must have dimension (all_p, 3).")
        self._points_packed = points_packed + offsets_packed
        new_points_list = list(
            self._points_packed.split(self.num_points_per_cloud().tolist(), 0)
        )
        # Note that since _compute_packed() has been executed, points_list
        # cannot be None even if not provided during construction.
        self._points_list = new_points_list
        if self._points_padded is not None:
            for i, points in enumerate(new_points_list):
                if len(points) > 0:
                    self._points_padded[i, : points.shape[0], :] = points
        return self

    # TODO(nikhilar) Move out of place operator to a utils file.
    def offset(self, offsets_packed):
        """
        Out of place offset.

        Args:
            offsets_packed: A Tensor of the same shape as self.points_packed
                giving offsets to be added to all points.
        Returns:
            new Pointclouds object.
        """
        new_clouds = self.clone()
        return new_clouds.offset_(offsets_packed)

    def scale_(self, scale):
        """
        Multiply the coordinates of this object by a scalar value.
        - i.e. enlarge/dilate
        In place operation.

        Args:
            scale: A scalar, or a Tensor of shape (N,).

        Returns:
            self.
        """
        if not torch.is_tensor(scale):
            scale = torch.full(len(self), scale)
        new_points_list = []
        points_list = self.points_list()
        for i, old_points in enumerate(points_list):
            new_points_list.append(scale[i] * old_points)
        self._points_list = new_points_list
        if self._points_packed is not None:
            self._points_packed = torch.cat(new_points_list, dim=0)
        if self._points_padded is not None:
            for i, points in enumerate(new_points_list):
                if len(points) > 0:
                    self._points_padded[i, : points.shape[0], :] = points
        return self

    def scale(self, scale):
        """
        Out of place scale_.

        Args:
            scale: A scalar, or a Tensor of shape (N,).

        Returns:
            new Pointclouds object.
        """
        new_clouds = self.clone()
        return new_clouds.scale_(scale)

    # TODO(nikhilar) Move function to utils file.
    def get_bounding_boxes(self):
        """
        Compute an axis-aligned bounding box for each cloud.

        Returns:
            bboxes: Tensor of shape (N, 3, 2) where bbox[i, j] gives the
            min and max values of cloud i along the jth coordinate axis.
        """
        all_mins, all_maxes = [], []
        for points in self.points_list():
            cur_mins = points.min(dim=0)[0]  # (3,)
            cur_maxes = points.max(dim=0)[0]  # (3,)
            all_mins.append(cur_mins)
            all_maxes.append(cur_maxes)
        all_mins = torch.stack(all_mins, dim=0)  # (N, 3)
        all_maxes = torch.stack(all_maxes, dim=0)  # (N, 3)
        bboxes = torch.stack([all_mins, all_maxes], dim=2)
        return bboxes

    def estimate_normals(
        self,
        neighborhood_size: int = 50,
        disambiguate_directions: bool = True,
        assign_to_self: bool = False,
    ):
        """
        Estimates the normals of each point in each cloud and assigns
        them to the internal tensors `self._normals_list` and `self._normals_padded`

        The function uses `ops.estimate_pointcloud_local_coord_frames`
        to estimate the normals. Please refer to this function for more
        detailed information about the implemented algorithm.

        Args:
        **neighborhood_size**: The size of the neighborhood used to estimate the
            geometry around each point.
        **disambiguate_directions**: If `True`, uses the algorithm from [1] to
            ensure sign consistency of the normals of neigboring points.
        **normals**: A tensor of normals for each input point
            of shape `(minibatch, num_point, 3)`.
            If `pointclouds` are of `Pointclouds` class, returns a padded tensor.
        **assign_to_self**: If `True`, assigns the computed normals to the
            internal buffers overwriting any previously stored normals.

        References:
          [1] Tombari, Salti, Di Stefano: Unique Signatures of Histograms for
          Local Surface Description, ECCV 2010.
        """

        # estimate the normals
        normals_est = ops.estimate_pointcloud_normals(
            self,
            neighborhood_size=neighborhood_size,
            disambiguate_directions=disambiguate_directions,
        )

        # assign to self
        if assign_to_self:
            _, self._normals_padded, _ = self._parse_auxiliary_input(normals_est)
            self._normals_list, self._normals_packed = None, None
            if self._points_list is not None:
                # update self._normals_list
                self.normals_list()
            if self._points_packed is not None:
                # update self._normals_packed
                self._normals_packed = torch.cat(self._normals_list, dim=0)

        return normals_est

    def extend(self, N: int):
        """
        Create new Pointclouds which contains each cloud N times.

        Args:
            N: number of new copies of each cloud.

        Returns:
            new Pointclouds object.
        """
        if not isinstance(N, int):
            raise ValueError("N must be an integer.")
        if N <= 0:
            raise ValueError("N must be > 0.")

        new_points_list, new_normals_list, new_features_list = [], None, None
        for points in self.points_list():
            new_points_list.extend(points.clone() for _ in range(N))
        if self.normals_list() is not None:
            new_normals_list = []
            for normals in self.normals_list():
                new_normals_list.extend(normals.clone() for _ in range(N))
        if self.features_list() is not None:
            new_features_list = []
            for features in self.features_list():
                new_features_list.extend(features.clone() for _ in range(N))
        return self.__class__(
            points=new_points_list, normals=new_normals_list, features=new_features_list
        )

    def update_padded(
        self, new_points_padded, new_normals_padded=None, new_features_padded=None
    ):
        """
        Returns a Pointcloud structure with updated padded tensors and copies of
        the auxiliary tensors. This function allows for an update of
        points_padded (and normals and features) without having to explicitly
        convert it to the list representation for heterogeneous batches.

        Args:
            new_points_padded: FloatTensor of shape (N, P, 3)
            new_normals_padded: (optional) FloatTensor of shape (N, P, 3)
            new_features_padded: (optional) FloatTensors of shape (N, P, C)

        Returns:
            Pointcloud with updated padded representations
        """

        def check_shapes(x, size):
            if x.shape[0] != size[0]:
                raise ValueError("new values must have the same batch dimension.")
            if x.shape[1] != size[1]:
                raise ValueError("new values must have the same number of points.")
            if size[2] is not None:
                if x.shape[2] != size[2]:
                    raise ValueError(
                        "new values must have the same number of channels."
                    )

        check_shapes(new_points_padded, [self._N, self._P, 3])
        if new_normals_padded is not None:
            check_shapes(new_normals_padded, [self._N, self._P, 3])
        if new_features_padded is not None:
            check_shapes(new_features_padded, [self._N, self._P, self._C])

        new = self.__class__(
            points=new_points_padded,
            normals=new_normals_padded,
            features=new_features_padded,
        )

        # overwrite the equisized flag
        new.equisized = self.equisized

        # copy normals
        if new_normals_padded is None:
            # If no normals are provided, keep old ones (shallow copy)
            new._normals_list = self._normals_list
            new._normals_padded = self._normals_padded
            new._normals_packed = self._normals_packed

        # copy features
        if new_features_padded is None:
            # If no features are provided, keep old ones (shallow copy)
            new._features_list = self._features_list
            new._features_padded = self._features_padded
            new._features_packed = self._features_packed

        # copy auxiliary tensors
        copy_tensors = [
            "_packed_to_cloud_idx",
            "_cloud_to_packed_first_idx",
            "_num_points_per_cloud",
            "_padded_to_packed_idx",
            "valid",
        ]
        for k in copy_tensors:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(new, k, v)  # shallow copy

        # update points
        new._points_padded = new_points_padded
        assert new._points_list is None
        assert new._points_packed is None

        # update normals and features if provided
        if new_normals_padded is not None:
            new._normals_padded = new_normals_padded
            new._normals_list = None
            new._normals_packed = None
        if new_features_padded is not None:
            new._features_padded = new_features_padded
            new._features_list = None
            new._features_packed = None
        return new

    def inside_box(self, box):
        """
        Finds the points inside a 3D box.

        Args:
            box: FloatTensor of shape (2, 3) or (N, 2, 3) where N is the number
                of clouds.
                    box[..., 0, :] gives the min x, y & z.
                    box[..., 1, :] gives the max x, y & z.
        Returns:
            idx: BoolTensor of length sum(P_i) indicating whether the packed points are
                within the input box.
        """
        if box.dim() > 3 or box.dim() < 2:
            raise ValueError("Input box must be of shape (2, 3) or (N, 2, 3).")

        if box.dim() == 3 and box.shape[0] != 1 and box.shape[0] != self._N:
            raise ValueError(
                "Input box dimension is incompatible with pointcloud size."
            )

        if box.dim() == 2:
            box = box[None]

        if (box[..., 0, :] > box[..., 1, :]).any():
            raise ValueError("Input box is invalid: min values larger than max values.")

        points_packed = self.points_packed()
        sumP = points_packed.shape[0]

        if box.shape[0] == 1:
            box = box.expand(sumP, 2, 3)
        elif box.shape[0] == self._N:
            box = box.unbind(0)
            box = [
                b.expand(p, 2, 3) for (b, p) in zip(box, self.num_points_per_cloud())
            ]
            box = torch.cat(box, 0)

        idx = (points_packed >= box[:, 0]) * (points_packed <= box[:, 1])
        return idx
