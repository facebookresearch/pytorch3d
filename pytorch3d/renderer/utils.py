# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import copy
import inspect
import warnings
from typing import Any, Union

import numpy as np
import torch


class TensorAccessor(object):
    """
    A helper class to be used with the __getitem__ method. This can be used for
    getting/setting the values for an attribute of a class at one particular
    index.  This is useful when the attributes of a class are batched tensors
    and one element in the batch needs to be modified.
    """

    def __init__(self, class_object, index: Union[int, slice]):
        """
        Args:
            class_object: this should be an instance of a class which has
                attributes which are tensors representing a batch of
                values.
            index: int/slice, an index indicating the position in the batch.
                In __setattr__ and __getattr__ only the value of class
                attributes at this index will be accessed.
        """
        self.__dict__["class_object"] = class_object
        self.__dict__["index"] = index

    def __setattr__(self, name: str, value: Any):
        """
        Update the attribute given by `name` to the value given by `value`
        at the index specified by `self.index`.

        Args:
            name: str, name of the attribute.
            value: value to set the attribute to.
        """
        v = getattr(self.class_object, name)
        if not torch.is_tensor(v):
            msg = "Can only set values on attributes which are tensors; got %r"
            raise AttributeError(msg % type(v))

        # Convert the attribute to a tensor if it is not a tensor.
        if not torch.is_tensor(value):
            value = torch.tensor(
                value, device=v.device, dtype=v.dtype, requires_grad=v.requires_grad
            )

        # Check the shapes match the existing shape and the shape of the index.
        if v.dim() > 1 and value.dim() > 1 and value.shape[1:] != v.shape[1:]:
            msg = "Expected value to have shape %r; got %r"
            raise ValueError(msg % (v.shape, value.shape))
        if (
            v.dim() == 0
            and isinstance(self.index, slice)
            and len(value) != len(self.index)
        ):
            msg = "Expected value to have len %r; got %r"
            raise ValueError(msg % (len(self.index), len(value)))
        self.class_object.__dict__[name][self.index] = value

    def __getattr__(self, name: str):
        """
        Return the value of the attribute given by "name" on self.class_object
        at the index specified in self.index.

        Args:
            name: string of the attribute name
        """
        if hasattr(self.class_object, name):
            return self.class_object.__dict__[name][self.index]
        else:
            msg = "Attribue %s not found on %r"
            return AttributeError(msg % (name, self.class_object.__name__))


BROADCAST_TYPES = (float, int, list, tuple, torch.Tensor, np.ndarray)


class TensorProperties(object):
    """
    A mix-in class for storing tensors as properties with helper methods.
    """

    def __init__(self, dtype=torch.float32, device="cpu", **kwargs):
        """
        Args:
            dtype: data type to set for the inputs
            device: str or torch.device
            kwargs: any number of keyword arguments. Any arguments which are
                of type (float/int/tuple/tensor/array) are broadcasted and
                other keyword arguments are set as attributes.
        """
        super().__init__()
        self.device = device
        self._N = 0
        if kwargs is not None:

            # broadcast all inputs which are float/int/list/tuple/tensor/array
            # set as attributes anything else e.g. strings, bools
            args_to_broadcast = {}
            for k, v in kwargs.items():
                if isinstance(v, (str, bool)):
                    setattr(self, k, v)
                elif isinstance(v, BROADCAST_TYPES):
                    args_to_broadcast[k] = v
                else:
                    msg = "Arg %s with type %r is not broadcastable"
                    warnings.warn(msg % (k, type(v)))

            names = args_to_broadcast.keys()
            # convert from type dict.values to tuple
            values = tuple(v for v in args_to_broadcast.values())

            if len(values) > 0:
                broadcasted_values = convert_to_tensors_and_broadcast(
                    *values, device=device
                )

                # Set broadcasted values as attributes on self.
                for i, n in enumerate(names):
                    setattr(self, n, broadcasted_values[i])
                    if self._N == 0:
                        self._N = broadcasted_values[i].shape[0]

    def __len__(self) -> int:
        return self._N

    def isempty(self) -> bool:
        return self._N == 0

    def __getitem__(self, index: Union[int, slice]):
        """

        Args:
            index: an int or slice used to index all the fields.

        Returns:
            if `index` is an index int/slice return a TensorAccessor class
            with getattribute/setattribute methods which return/update the value
            at the index in the original camera.
        """
        if isinstance(index, (int, slice)):
            return TensorAccessor(class_object=self, index=index)

        msg = "Expected index of type int or slice; got %r"
        raise ValueError(msg % type(index))

    def to(self, device: str = "cpu"):
        """
        In place operation to move class properties which are tensors to a
        specified device. If self has a property "device", update this as well.
        """
        for k in dir(self):
            v = getattr(self, k)
            if k == "device":
                setattr(self, k, device)
            if torch.is_tensor(v) and v.device != device:
                setattr(self, k, v.to(device))
        return self

    def clone(self, other):
        """
        Update the tensor properties of other with the cloned properties of self.
        """
        for k in dir(self):
            v = getattr(self, k)
            if inspect.ismethod(v) or k.startswith("__"):
                continue
            if torch.is_tensor(v):
                v_clone = v.clone()
            else:
                v_clone = copy.deepcopy(v)
            setattr(other, k, v_clone)
        return other

    def gather_props(self, batch_idx):
        """
        This is an in place operation to reformat all tensor class attributes
        based on a set of given indices using torch.gather. This is useful when
        attributes which are batched tensors e.g. shape (N, 3) need to be
        multiplied with another tensor which has a different first dimension
        e.g. packed vertices of shape (V, 3).

        Example

        .. code-block:: python

            self.specular_color = (N, 3) tensor of specular colors for each mesh

        A lighting calculation may use

        .. code-block:: python

            verts_packed = meshes.verts_packed()  # (V, 3)

        To multiply these two tensors the batch dimension needs to be the same.
        To achieve this we can do

        .. code-block:: python

            batch_idx = meshes.verts_packed_to_mesh_idx()  # (V)

        This gives index of the mesh for each vertex in verts_packed.

        .. code-block:: python

            self.gather_props(batch_idx)
            self.specular_color = (V, 3) tensor with the specular color for
                                     each packed vertex.

        torch.gather requires the index tensor to have the same shape as the
        input tensor so this method takes care of the reshaping of the index
        tensor to use with class attributes with arbitrary dimensions.

        Args:
            batch_idx: shape (B, ...) where `...` represents an arbitrary
                number of dimensions

        Returns:
            self with all properties reshaped. e.g. a property with shape (N, 3)
            is transformed to shape (B, 3).
        """
        # Iterate through the attributes of the class which are tensors.
        for k in dir(self):
            v = getattr(self, k)
            if torch.is_tensor(v):
                if v.shape[0] > 1:
                    # There are different values for each batch element
                    # so gather these using the batch_idx.
                    # First clone the input batch_idx tensor before
                    # modifying it.
                    _batch_idx = batch_idx.clone()
                    idx_dims = _batch_idx.shape
                    tensor_dims = v.shape
                    if len(idx_dims) > len(tensor_dims):
                        msg = "batch_idx cannot have more dimensions than %s. "
                        msg += "got shape %r and %s has shape %r"
                        raise ValueError(msg % (k, idx_dims, k, tensor_dims))
                    if idx_dims != tensor_dims:
                        # To use torch.gather the index tensor (_batch_idx) has
                        # to have the same shape as the input tensor.
                        new_dims = len(tensor_dims) - len(idx_dims)
                        new_shape = idx_dims + (1,) * new_dims
                        expand_dims = (-1,) + tensor_dims[1:]
                        _batch_idx = _batch_idx.view(*new_shape)
                        _batch_idx = _batch_idx.expand(*expand_dims)

                    v = v.gather(0, _batch_idx)
                    setattr(self, k, v)
        return self


def format_tensor(input, dtype=torch.float32, device: str = "cpu") -> torch.Tensor:
    """
    Helper function for converting a scalar value to a tensor.

    Args:
        input: Python scalar, Python list/tuple, torch scalar, 1D torch tensor
        dtype: data type for the input
        device: torch device on which the tensor should be placed.

    Returns:
        input_vec: torch tensor with optional added batch dimension.
    """
    if not torch.is_tensor(input):
        input = torch.tensor(input, dtype=dtype, device=device)
    if input.dim() == 0:
        input = input.view(1)
    if input.device != device:
        input = input.to(device=device)
    return input


def convert_to_tensors_and_broadcast(*args, dtype=torch.float32, device: str = "cpu"):
    """
    Helper function to handle parsing an arbitrary number of inputs (*args)
    which all need to have the same batch dimension.
    The output is a list of tensors.

    Args:
        *args: an arbitrary number of inputs
            Each of the values in `args` can be one of the following
                - Python scalar
                - Torch scalar
                - Torch tensor of shape (N, K_i) or (1, K_i) where K_i are
                  an arbitrary number of dimensions which can vary for each
                  value in args. In this case each input is broadcast to a
                  tensor of shape (N, K_i)
        dtype: data type to use when creating new tensors.
        device: torch device on which the tensors should be placed.

    Output:
        args: A list of tensors of shape (N, K_i)
    """
    # Convert all inputs to tensors with a batch dimension
    args_1d = [format_tensor(c, dtype, device) for c in args]

    # Find broadcast size
    sizes = [c.shape[0] for c in args_1d]
    N = max(sizes)

    args_Nd = []
    for c in args_1d:
        if c.shape[0] != 1 and c.shape[0] != N:
            msg = "Got non-broadcastable sizes %r" % (sizes)
            raise ValueError(msg)

        # Expand broadcast dim and keep non broadcast dims the same size
        expand_sizes = (N,) + (-1,) * len(c.shape[1:])
        args_Nd.append(c.expand(*expand_sizes))

    if len(args) == 1:
        args_Nd = args_Nd[0]  # Return the first element

    return args_Nd
