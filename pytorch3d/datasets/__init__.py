# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
from .shapenet import ShapeNetCore


__all__ = [k for k in globals().keys() if not k.startswith("_")]
