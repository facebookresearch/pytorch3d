# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import unittest

import torch
from pytorch3d.implicitron.tools.config import expand_args_fields, get_default_args

from ..impl.optimizer_factory import (
    ImplicitronOptimizerFactory,
    logger as factory_logger,
)

internal = os.environ.get("FB_TEST", False)


class TestOptimizerFactory(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        expand_args_fields(ImplicitronOptimizerFactory)

    def _get_param_groups(self, model):
        default_cfg = get_default_args(ImplicitronOptimizerFactory)
        factory = ImplicitronOptimizerFactory(default_cfg)
        oldlevel = factory_logger.level
        factory_logger.setLevel(logging.ERROR)
        out = factory._get_param_groups(model)
        factory_logger.setLevel(oldlevel)
        return out

    def _assert_allin(self, a, param_groups, key):
        """
        Asserts that all the parameters in a are in the group
        named by key.
        """
        with self.subTest(f"Testing key {key}"):
            b = param_groups[key]
            for el in a:
                if el not in b:
                    raise ValueError(
                        f"Element {el}\n\n from:\n\n {a}\n\n not in:\n\n {b}\n\n."
                        + f" Full param groups = \n\n{param_groups}"
                    )
            for el in b:
                if el not in a:
                    raise ValueError(
                        f"Element {el}\n\n from:\n\n {b}\n\n not in:\n\n {a}\n\n."
                        + f" Full param groups = \n\n{param_groups}"
                    )

    def test_default_param_group_assignment(self):
        pa, pb, pc = [torch.nn.Parameter(data=torch.tensor(i * 1.0)) for i in range(3)]
        na, nb = Node(params=[pa]), Node(params=[pb])
        root = Node(children=[na, nb], params=[pc])
        param_groups = self._get_param_groups(root)
        self._assert_allin([pa, pb, pc], param_groups, "default")

    def test_member_overrides_default_param_group_assignment(self):
        pa, pb, pc = [torch.nn.Parameter(data=torch.tensor(i * 1.0)) for i in range(3)]
        na, nb = Node(params=[pa]), Node(params=[pb])
        root = Node(children=[na, nb], params=[pc], param_groups={"m1": "pb"})
        param_groups = self._get_param_groups(root)
        self._assert_allin([pa, pc], param_groups, "default")
        self._assert_allin([pb], param_groups, "pb")

    def test_self_overrides_member_param_group_assignment(self):
        pa, pb, pc = [torch.nn.Parameter(data=torch.tensor(i * 1.0)) for i in range(3)]
        na, nb = Node(params=[pa]), Node(params=[pb], param_groups={"self": "pb_self"})
        root = Node(children=[na, nb], params=[pc], param_groups={"m1": "pb_member"})
        param_groups = self._get_param_groups(root)
        self._assert_allin([pa, pc], param_groups, "default")
        self._assert_allin([pb], param_groups, "pb_self")
        assert len(param_groups["pb_member"]) == 0, param_groups

    def test_param_overrides_self_param_group_assignment(self):
        pa, pb, pc = [torch.nn.Parameter(data=torch.tensor(i * 1.0)) for i in range(3)]
        na, nb = Node(params=[pa]), Node(
            params=[pb], param_groups={"self": "pb_self", "p1": "pb_param"}
        )
        root = Node(children=[na, nb], params=[pc], param_groups={"m1": "pb_member"})
        param_groups = self._get_param_groups(root)
        self._assert_allin([pa, pc], param_groups, "default")
        self._assert_allin([pb], param_groups, "pb_self")
        assert len(param_groups["pb_member"]) == 0, param_groups

    def test_no_param_groups_defined(self):
        pa, pb, pc = [torch.nn.Parameter(data=torch.tensor(i * 1.0)) for i in range(3)]
        na, nb = Node(params=[pa]), Node(params=[pb])
        root = Node(children=[na, nb], params=[pc])
        param_groups = self._get_param_groups(root)
        self._assert_allin([pa, pb, pc], param_groups, "default")

    def test_double_dotted(self):
        pa, pb = [torch.nn.Parameter(data=torch.tensor(i * 1.0)) for i in range(2)]
        na = Node(params=[pa, pb])
        nb = Node(children=[na])
        root = Node(children=[nb], param_groups={"m0.m0.p0": "X", "m0.m0": "Y"})
        param_groups = self._get_param_groups(root)
        self._assert_allin([pa], param_groups, "X")
        self._assert_allin([pb], param_groups, "Y")

    def test_tree_param_groups_defined(self):
        """
        Test generic tree assignment.

        A0
        |---------------------------
        |              |           |
        Bb             M           J-
        |-----                     |-------
        |     |                    |      |
        C     Ddg                  K      Ll
              |--------------
              |    |    |    |
              E4   Ff   G    H-

        All nodes have one parameter. Character next to the capital
        letter means they have added something to their `parameter_groups`:
            - small letter same as capital means self is set to that letter
            - small letter different then capital means that member is set
                (the one that is named like that)
            - number means parameter's parameter_group is set like that
            - "-" means it does not have `parameter_groups` member
        """
        p = [torch.nn.Parameter(data=torch.tensor(i * 1.0)) for i in range(12)]
        L = Node(params=[p[11]], param_groups={"self": "l"})
        K = Node(params=[p[10]], param_groups={})
        J = Node(params=[p[9]], param_groups=None, children=[K, L])
        M = Node(params=[p[8]], param_groups={})

        E = Node(params=[p[4]], param_groups={"p0": "4"})
        F = Node(params=[p[5]], param_groups={"self": "f"})
        G = Node(params=[p[6]], param_groups={})
        H = Node(params=[p[7]], param_groups=None)

        D = Node(
            params=[p[3]], param_groups={"self": "d", "m2": "g"}, children=[E, F, G, H]
        )
        C = Node(params=[p[2]], param_groups={})

        B = Node(params=[p[1]], param_groups={"self": "b"}, children=[C, D])

        A = Node(params=[p[0]], param_groups={"p0": "0"}, children=[B, M, J])

        param_groups = self._get_param_groups(A)

        # if parts of the group belong to two different categories assert is repeated
        # parameter level
        self._assert_allin([p[0]], param_groups, "0")
        self._assert_allin([p[4]], param_groups, "4")
        # self level
        self._assert_allin([p[5]], param_groups, "f")
        self._assert_allin([p[11]], param_groups, "l")
        self._assert_allin([p[2], p[1]], param_groups, "b")
        self._assert_allin([p[7], p[3]], param_groups, "d")
        # member level
        self._assert_allin([p[6]], param_groups, "g")
        # inherit level
        self._assert_allin([p[7], p[3]], param_groups, "d")
        self._assert_allin([p[2], p[1]], param_groups, "b")
        # default level
        self._assert_allin([p[8], p[9], p[10]], param_groups, "default")


class Node(torch.nn.Module):
    def __init__(self, children=(), params=(), param_groups=None):
        super().__init__()
        for i, child in enumerate(children):
            self.add_module("m" + str(i), child)
        for i, param in enumerate(params):
            setattr(self, "p" + str(i), param)
        if param_groups is not None:
            self.param_groups = param_groups

    def __str__(self):
        return (
            "modules:\n" + str(self._modules) + "\nparameters\n" + str(self._parameters)
        )
