# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import textwrap
import unittest
from dataclasses import dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock

from omegaconf import DictConfig, ListConfig, OmegaConf, ValidationError
from pytorch3d.implicitron.tools.config import (
    _get_type_to_process,
    _is_actually_dataclass,
    _ProcessType,
    _Registry,
    Configurable,
    enable_get_default_args,
    expand_args_fields,
    get_default_args,
    get_default_args_field,
    registry,
    remove_unused_components,
    ReplaceableBase,
    run_auto_creation,
)


@dataclass
class Animal(ReplaceableBase):
    pass


class Fruit(ReplaceableBase):
    pass


@registry.register
class Banana(Fruit):
    pips: int
    spots: int
    bananame: str


@registry.register
class Pear(Fruit):
    n_pips: int = 13


class Pineapple(Fruit):
    pass


@registry.register
class Orange(Fruit):
    pass


@registry.register
class Kiwi(Fruit):
    pass


@registry.register
class LargePear(Pear):
    pass


class BoringConfigurable(Configurable):
    pass


class MainTest(Configurable):
    the_fruit: Fruit
    n_ids: int
    n_reps: int = 8
    the_second_fruit: Fruit

    def create_the_second_fruit(self):
        expand_args_fields(Pineapple)
        self.the_second_fruit = Pineapple()

    def __post_init__(self):
        run_auto_creation(self)


class TestConfig(unittest.TestCase):
    def test_is_actually_dataclass(self):
        @dataclass
        class A:
            pass

        self.assertTrue(_is_actually_dataclass(A))
        self.assertTrue(is_dataclass(A))

        class B(A):
            a: int

        self.assertFalse(_is_actually_dataclass(B))
        self.assertTrue(is_dataclass(B))

    def test_get_type_to_process(self):
        gt = _get_type_to_process
        self.assertIsNone(gt(int))
        self.assertEqual(gt(Fruit), (Fruit, _ProcessType.REPLACEABLE))
        self.assertEqual(
            gt(Optional[Fruit]), (Fruit, _ProcessType.OPTIONAL_REPLACEABLE)
        )
        self.assertEqual(gt(MainTest), (MainTest, _ProcessType.CONFIGURABLE))
        self.assertEqual(
            gt(Optional[MainTest]), (MainTest, _ProcessType.OPTIONAL_CONFIGURABLE)
        )
        self.assertIsNone(gt(Optional[int]))
        self.assertIsNone(gt(Tuple[Fruit]))
        self.assertIsNone(gt(Tuple[Fruit, Animal]))
        self.assertIsNone(gt(Optional[List[int]]))

    def test_simple_replacement(self):
        struct = get_default_args(MainTest)
        struct.n_ids = 9780
        struct.the_fruit_Pear_args.n_pips = 3
        struct.the_fruit_class_type = "Pear"
        struct.the_second_fruit_class_type = "Pear"

        main = MainTest(**struct)
        self.assertIsInstance(main.the_fruit, Pear)
        self.assertEqual(main.n_reps, 8)
        self.assertEqual(main.n_ids, 9780)
        self.assertEqual(main.the_fruit.n_pips, 3)
        self.assertIsInstance(main.the_second_fruit, Pineapple)

        struct2 = get_default_args(MainTest)
        self.assertEqual(struct2.the_fruit_Pear_args.n_pips, 13)

        self.assertEqual(
            MainTest._creation_functions,
            ("create_the_fruit", "create_the_second_fruit"),
        )

    def test_detect_bases(self):
        # testing the _base_class_from_class function
        self.assertIsNone(_Registry._base_class_from_class(ReplaceableBase))
        self.assertIsNone(_Registry._base_class_from_class(MainTest))
        self.assertIs(_Registry._base_class_from_class(Fruit), Fruit)
        self.assertIs(_Registry._base_class_from_class(Pear), Fruit)

        class PricklyPear(Pear):
            pass

        self.assertIs(_Registry._base_class_from_class(PricklyPear), Fruit)

    def test_registry_entries(self):
        self.assertIs(registry.get(Fruit, "Banana"), Banana)
        with self.assertRaisesRegex(ValueError, "Banana has not been registered."):
            registry.get(Animal, "Banana")
        with self.assertRaisesRegex(ValueError, "PricklyPear has not been registered."):
            registry.get(Fruit, "PricklyPear")

        self.assertIs(registry.get(Pear, "Pear"), Pear)
        self.assertIs(registry.get(Pear, "LargePear"), LargePear)
        with self.assertRaisesRegex(ValueError, "Banana resolves to"):
            registry.get(Pear, "Banana")

        all_fruit = set(registry.get_all(Fruit))
        self.assertIn(Banana, all_fruit)
        self.assertIn(Pear, all_fruit)
        self.assertIn(LargePear, all_fruit)
        self.assertEqual(registry.get_all(Pear), [LargePear])

        @registry.register
        class Apple(Fruit):
            pass

        @registry.register
        class CrabApple(Apple):
            pass

        self.assertEqual(registry.get_all(Apple), [CrabApple])

        self.assertIs(registry.get(Fruit, "CrabApple"), CrabApple)

        with self.assertRaisesRegex(ValueError, "Cannot tell what it is."):

            @registry.register
            class NotAFruit:
                pass

    def test_recursion(self):
        class Shape(ReplaceableBase):
            pass

        @registry.register
        class Triangle(Shape):
            a: float = 5.0

        @registry.register
        class Square(Shape):
            a: float = 3.0

        @registry.register
        class LargeShape(Shape):
            inner: Shape

            def __post_init__(self):
                run_auto_creation(self)

        class ShapeContainer(Configurable):
            shape: Shape

        container = ShapeContainer(**get_default_args(ShapeContainer))
        # This is because ShapeContainer is missing __post_init__
        with self.assertRaises(AttributeError):
            container.shape

        class ShapeContainer2(Configurable):
            x: Shape
            x_class_type: str = "LargeShape"

            def __post_init__(self):
                self.x_LargeShape_args.inner_class_type = "Triangle"
                run_auto_creation(self)

        container2_args = get_default_args(ShapeContainer2)
        container2_args.x_LargeShape_args.inner_Triangle_args.a += 10
        self.assertIn("inner_Square_args", container2_args.x_LargeShape_args)
        # We do not perform expansion that would result in an infinite recursion,
        # so this member is not present.
        self.assertNotIn("inner_LargeShape_args", container2_args.x_LargeShape_args)
        container2_args.x_LargeShape_args.inner_Square_args.a += 100
        container2 = ShapeContainer2(**container2_args)
        self.assertIsInstance(container2.x, LargeShape)
        self.assertIsInstance(container2.x.inner, Triangle)
        self.assertEqual(container2.x.inner.a, 15.0)

    def test_simpleclass_member(self):
        # Members which are not dataclasses are
        # tolerated. But it would be nice to be able to
        # configure them.
        class Foo:
            def __init__(self, a: Any = 1, b: Any = 2):
                self.a, self.b = a, b

        enable_get_default_args(Foo)

        @dataclass()
        class Bar:
            aa: int = 9
            bb: int = 9

        class Container(Configurable):
            bar: Bar = Bar()
            # TODO make this work?
            # foo: Foo = Foo()
            fruit: Fruit
            fruit_class_type: str = "Orange"

            def __post_init__(self):
                run_auto_creation(self)

        self.assertEqual(get_default_args(Foo), {"a": 1, "b": 2})
        container_args = get_default_args(Container)
        container = Container(**container_args)
        self.assertIsInstance(container.fruit, Orange)
        self.assertEqual(Container._processed_members, {"fruit": Fruit})
        self.assertEqual(container._processed_members, {"fruit": Fruit})

        container_defaulted = Container()
        container_defaulted.fruit_Pear_args.n_pips += 4

        container_args2 = get_default_args(Container)
        container = Container(**container_args2)
        self.assertEqual(container.fruit_Pear_args.n_pips, 13)

    def test_inheritance(self):
        # Also exercises optional replaceables
        class FruitBowl(ReplaceableBase):
            main_fruit: Fruit
            main_fruit_class_type: str = "Orange"

            def __post_init__(self):
                raise ValueError("This doesn't get called")

        class LargeFruitBowl(FruitBowl):
            extra_fruit: Optional[Fruit]
            extra_fruit_class_type: str = "Kiwi"
            no_fruit: Optional[Fruit]
            no_fruit_class_type: Optional[str] = None

            def __post_init__(self):
                run_auto_creation(self)

        large_args = get_default_args(LargeFruitBowl)
        self.assertNotIn("extra_fruit", large_args)
        self.assertNotIn("main_fruit", large_args)
        large = LargeFruitBowl(**large_args)
        self.assertIsInstance(large.main_fruit, Orange)
        self.assertIsInstance(large.extra_fruit, Kiwi)
        self.assertIsNone(large.no_fruit)
        self.assertIn("no_fruit_Kiwi_args", large_args)

        remove_unused_components(large_args)
        large2 = LargeFruitBowl(**large_args)
        self.assertIsInstance(large2.main_fruit, Orange)
        self.assertIsInstance(large2.extra_fruit, Kiwi)
        self.assertIsNone(large2.no_fruit)
        needed_args = [
            "extra_fruit_Kiwi_args",
            "extra_fruit_class_type",
            "main_fruit_Orange_args",
            "main_fruit_class_type",
            "no_fruit_class_type",
        ]
        self.assertEqual(sorted(large_args.keys()), needed_args)

        with self.assertRaisesRegex(ValueError, "NotAFruit has not been registered."):
            LargeFruitBowl(extra_fruit_class_type="NotAFruit")

    def test_inheritance2(self):
        # This is a case where a class could contain an instance
        # of a subclass, which is ignored.
        class Parent(ReplaceableBase):
            pass

        class Main(Configurable):
            parent: Parent
            # Note - no __post__init__

        @registry.register
        class Derived(Parent, Main):
            pass

        args = get_default_args(Main)
        # Derived has been ignored in processing Main.
        self.assertCountEqual(args.keys(), ["parent_class_type"])

        main = Main(**args)

        with self.assertRaisesRegex(ValueError, "UNDEFAULTED has not been registered."):
            run_auto_creation(main)

        main.parent_class_type = "Derived"
        # Illustrates that a dict works fine instead of a DictConfig.
        main.parent_Derived_args = {}
        with self.assertRaises(AttributeError):
            main.parent
        run_auto_creation(main)
        self.assertIsInstance(main.parent, Derived)

    def test_redefine(self):
        class FruitBowl(ReplaceableBase):
            main_fruit: Fruit
            main_fruit_class_type: str = "Grape"

            def __post_init__(self):
                run_auto_creation(self)

        @registry.register
        @dataclass
        class Grape(Fruit):
            large: bool = False

            def get_color(self):
                return "red"

            def __post_init__(self):
                raise ValueError("This doesn't get called")

        bowl_args = get_default_args(FruitBowl)

        @registry.register
        @dataclass
        class Grape(Fruit):  # noqa: F811
            large: bool = True

            def get_color(self):
                return "green"

        with self.assertWarnsRegex(
            UserWarning, "New implementation of Grape is being chosen."
        ):
            defaulted_bowl = FruitBowl()
        self.assertIsInstance(defaulted_bowl.main_fruit, Grape)
        self.assertEqual(defaulted_bowl.main_fruit.large, True)
        self.assertEqual(defaulted_bowl.main_fruit.get_color(), "green")

        with self.assertWarnsRegex(
            UserWarning, "New implementation of Grape is being chosen."
        ):
            args_bowl = FruitBowl(**bowl_args)
        self.assertIsInstance(args_bowl.main_fruit, Grape)
        # Redefining the same class won't help with defaults because encoded in args
        self.assertEqual(args_bowl.main_fruit.large, False)
        # But the override worked.
        self.assertEqual(args_bowl.main_fruit.get_color(), "green")

        # 2. Try redefining without the dataclass modifier
        # This relies on the fact that default creation processes the class.
        # (otherwise incomprehensible messages)
        @registry.register
        class Grape(Fruit):  # noqa: F811
            large: bool = True

        with self.assertWarnsRegex(
            UserWarning, "New implementation of Grape is being chosen."
        ):
            FruitBowl(**bowl_args)

        # 3. Adding a new class doesn't get picked up, because the first
        # get_default_args call has frozen FruitBowl. This is intrinsic to
        # the way dataclass and expand_args_fields work in-place but
        # expand_args_fields is not pure - it depends on the registry.
        @registry.register
        class Fig(Fruit):
            pass

        bowl_args2 = get_default_args(FruitBowl)
        self.assertIn("main_fruit_Grape_args", bowl_args2)
        self.assertNotIn("main_fruit_Fig_args", bowl_args2)

        # TODO Is it possible to make this work?
        # bowl_args2["main_fruit_Fig_args"] = get_default_args(Fig)
        # bowl_args2.main_fruit_class_type = "Fig"
        # bowl2 = FruitBowl(**bowl_args2)  <= unexpected argument

        # Note that it is possible to use Fig if you can set
        # bowl2.main_fruit_Fig_args explicitly (not in bowl_args2)
        # before run_auto_creation happens. See test_inheritance2
        # for an example.

    def test_no_replacement(self):
        # Test of Configurables without ReplaceableBase
        class A(Configurable):
            n: int = 9

        class B(Configurable):
            a: A

            def __post_init__(self):
                run_auto_creation(self)

        class C(Configurable):
            b1: B
            b2: Optional[B]
            b3: Optional[B]
            b2_enabled: bool = True
            b3_enabled: bool = False

            def __post_init__(self):
                run_auto_creation(self)

        c_args = get_default_args(C)
        c = C(**c_args)
        self.assertIsInstance(c.b1.a, A)
        self.assertEqual(c.b1.a.n, 9)
        self.assertFalse(hasattr(c, "b1_enabled"))
        self.assertIsInstance(c.b2.a, A)
        self.assertEqual(c.b2.a.n, 9)
        self.assertTrue(c.b2_enabled)
        self.assertIsNone(c.b3)
        self.assertFalse(c.b3_enabled)

    def test_doc(self):
        # The case in the docstring.
        class A(ReplaceableBase):
            k: int = 1

        @registry.register
        class A1(A):
            m: int = 3

        @registry.register
        class A2(A):
            n: str = "2"

        class B(Configurable):
            a: A
            a_class_type: str = "A2"

            def __post_init__(self):
                run_auto_creation(self)

        b_args = get_default_args(B)
        self.assertNotIn("a", b_args)
        b = B(**b_args)
        self.assertEqual(b.a.n, "2")

    def test_raw_types(self):
        @dataclass
        class MyDataclass:
            int_field: int = 0
            none_field: Optional[int] = None
            float_field: float = 9.3
            bool_field: bool = True
            tuple_field: Tuple[int, ...] = (3,)

        class SimpleClass:
            def __init__(
                self,
                tuple_member_: Tuple[int, int] = (3, 4),
            ):
                self.tuple_member = tuple_member_

            def get_tuple(self):
                return self.tuple_member

        enable_get_default_args(SimpleClass)

        def f(*, a: int = 3, b: str = "kj"):
            self.assertEqual(a, 3)
            self.assertEqual(b, "kj")

        enable_get_default_args(f)

        class C(Configurable):
            simple: DictConfig = get_default_args_field(SimpleClass)
            # simple2: SimpleClass2 = SimpleClass2()
            mydata: DictConfig = get_default_args_field(MyDataclass)
            a_tuple: Tuple[float] = (4.0, 3.0)
            f_args: DictConfig = get_default_args_field(f)

        args = get_default_args(C)
        c = C(**args)
        self.assertCountEqual(args.keys(), ["simple", "mydata", "a_tuple", "f_args"])

        mydata = MyDataclass(**c.mydata)
        simple = SimpleClass(**c.simple)

        # OmegaConf converts tuples to ListConfigs (which act like lists).
        self.assertEqual(simple.get_tuple(), [3, 4])
        self.assertTrue(isinstance(simple.get_tuple(), ListConfig))
        # get_default_args converts sets to ListConfigs (which act like lists).
        self.assertEqual(c.a_tuple, [4.0, 3.0])
        self.assertTrue(isinstance(c.a_tuple, ListConfig))
        self.assertEqual(mydata.tuple_field, (3,))
        self.assertTrue(isinstance(mydata.tuple_field, ListConfig))
        f(**c.f_args)

    def test_irrelevant_bases(self):
        class NotADataclass:
            # Like torch.nn.Module, this class contains annotations
            # but is not designed to be dataclass'd.
            # This test ensures that such classes, when inherited fron,
            # are not accidentally affected by expand_args_fields.
            a: int = 9
            b: int

        class LeftConfigured(Configurable, NotADataclass):
            left: int = 1

        class RightConfigured(NotADataclass, Configurable):
            right: int = 2

        class Outer(Configurable):
            left: LeftConfigured
            right: RightConfigured

            def __post_init__(self):
                run_auto_creation(self)

        outer = Outer(**get_default_args(Outer))
        self.assertEqual(outer.left.left, 1)
        self.assertEqual(outer.right.right, 2)
        with self.assertRaisesRegex(TypeError, "non-default argument"):
            dataclass(NotADataclass)

    def test_unprocessed(self):
        # behavior of Configurable classes which need processing in __new__,
        class UnprocessedConfigurable(Configurable):
            a: int = 9

        class UnprocessedReplaceable(ReplaceableBase):
            a: int = 9

        for Unprocessed in [UnprocessedConfigurable, UnprocessedReplaceable]:

            self.assertFalse(_is_actually_dataclass(Unprocessed))
            unprocessed = Unprocessed()
            self.assertTrue(_is_actually_dataclass(Unprocessed))
            self.assertTrue(isinstance(unprocessed, Unprocessed))
            self.assertEqual(unprocessed.a, 9)

    def test_enum(self):
        # Test that enum values are kept, i.e. that OmegaConf's runtime checks
        # are in use.

        class A(Enum):
            B1 = "b1"
            B2 = "b2"

        # Test for a Configurable class, a function, and a regular class.
        class C(Configurable):
            a: A = A.B1

        # Also test for a calllable with enum arguments.
        def C_fn(a: A = A.B1):
            pass

        enable_get_default_args(C_fn)

        class C_cl:
            def __init__(self, a: A = A.B1) -> None:
                pass

        enable_get_default_args(C_cl)

        for C_ in [C, C_fn, C_cl]:
            base = get_default_args(C_)
            self.assertEqual(OmegaConf.to_yaml(base), "a: B1\n")
            self.assertEqual(base.a, A.B1)
            replaced = OmegaConf.merge(base, {"a": "B2"})
            self.assertEqual(replaced.a, A.B2)
            with self.assertRaises(ValidationError):
                # You can't use a value which is not one of the
                # choices, even if it is the str representation
                # of one of the choices.
                OmegaConf.merge(base, {"a": "b2"})

            remerged = OmegaConf.merge(base, OmegaConf.create(OmegaConf.to_yaml(base)))
            self.assertEqual(remerged.a, A.B1)

    def test_pickle(self):
        def func(a: int = 1, b: str = "3"):
            pass

        enable_get_default_args(func)

        args = get_default_args(func)
        args2 = pickle.loads(pickle.dumps(args))
        self.assertEqual(args2.a, 1)
        self.assertEqual(args2.b, "3")

        args_regenerated = get_default_args(func)
        pickle.dumps(args_regenerated)
        pickle.dumps(args)

    def test_remove_unused_components(self):
        struct = get_default_args(MainTest)
        struct.n_ids = 32
        struct.the_fruit_class_type = "Pear"
        struct.the_second_fruit_class_type = "Banana"
        remove_unused_components(struct)
        expected_keys = [
            "n_ids",
            "n_reps",
            "the_fruit_Pear_args",
            "the_fruit_class_type",
            "the_second_fruit_Banana_args",
            "the_second_fruit_class_type",
        ]
        expected_yaml = textwrap.dedent(
            """\
            n_ids: 32
            n_reps: 8
            the_fruit_class_type: Pear
            the_fruit_Pear_args:
              n_pips: 13
            the_second_fruit_class_type: Banana
            the_second_fruit_Banana_args:
              pips: ???
              spots: ???
              bananame: ???
            """
        )
        self.assertEqual(sorted(struct.keys()), expected_keys)

        # Check that struct is what we expect
        expected = OmegaConf.create(expected_yaml)
        self.assertEqual(struct, expected)

        # Check that we get what we expect when writing to yaml.
        self.assertEqual(OmegaConf.to_yaml(struct, sort_keys=False), expected_yaml)

        main = MainTest(**struct)
        instance_data = OmegaConf.structured(main)
        remove_unused_components(instance_data)
        self.assertEqual(sorted(instance_data.keys()), expected_keys)
        self.assertEqual(instance_data, expected)

    def test_remove_unused_components_optional(self):
        class MainTestWrapper(Configurable):
            mt: Optional[MainTest]
            mt_enabled: bool = False

        args = get_default_args(MainTestWrapper)
        self.assertEqual(list(args.keys()), ["mt_enabled", "mt_args"])
        remove_unused_components(args)
        self.assertEqual(OmegaConf.to_yaml(args), "mt_enabled: false\n")

    def test_get_instance_args(self):
        mt1, mt2 = [
            MainTest(
                n_ids=0,
                n_reps=909,
                the_fruit_class_type="Pear",
                the_second_fruit_class_type="Pear",
                the_fruit_Pear_args=DictConfig({}),
                the_second_fruit_Pear_args={},
            )
            for _ in range(2)
        ]
        # Two equivalent ways to get the DictConfig back out of an instance.
        cfg1 = OmegaConf.structured(mt1)
        cfg2 = get_default_args(mt2)
        self.assertEqual(cfg1, cfg2)
        self.assertEqual(len(cfg1.the_second_fruit_Pear_args), 0)
        self.assertEqual(len(mt2.the_second_fruit_Pear_args), 0)

        from_cfg = MainTest(**cfg2)
        self.assertEqual(len(from_cfg.the_second_fruit_Pear_args), 0)

        # If you want the complete args, merge with the defaults.
        merged_args = OmegaConf.merge(get_default_args(MainTest), cfg2)
        from_merged = MainTest(**merged_args)
        self.assertEqual(len(from_merged.the_second_fruit_Pear_args), 1)
        self.assertEqual(from_merged.n_reps, 909)

    def test_tweak_hook(self):
        class A(Configurable):
            n: int = 9

        class Wrapper(Configurable):
            fruit: Fruit
            fruit_class_type: str = "Pear"
            fruit2: Fruit
            fruit2_class_type: str = "Pear"
            a: A
            a2: A
            a3: A

            @classmethod
            def a_tweak_args(cls, type, args):
                assert type == A
                args.n = 993

            @classmethod
            def a3_tweak_args(cls, type, args):
                del args["n"]

            @classmethod
            def fruit_tweak_args(cls, type, args):
                assert issubclass(type, Fruit)
                if type == Pear:
                    assert args.n_pips == 13
                    args.n_pips = 19

        args = get_default_args(Wrapper)
        self.assertEqual(args.a_args.n, 993)
        self.assertEqual(args.a2_args.n, 9)
        self.assertEqual(args.a3_args, {})
        self.assertEqual(args.fruit_Pear_args.n_pips, 19)
        self.assertEqual(args.fruit2_Pear_args.n_pips, 13)

    def test_impls(self):
        # Check that create_x actually uses create_x_impl to do its work
        # by using all the member types, both with a faked impl function
        # and without.
        # members with _0 are optional and absent, those with _o are
        # optional and present.
        control_args = []

        def fake_impl(self, control, args):
            control_args.append(control)

        for fake in [False, True]:

            class MyClass(Configurable):
                fruit: Fruit
                fruit_class_type: str = "Orange"
                fruit_o: Optional[Fruit]
                fruit_o_class_type: str = "Orange"
                fruit_0: Optional[Fruit]
                fruit_0_class_type: Optional[str] = None
                boring: BoringConfigurable
                boring_o: Optional[BoringConfigurable]
                boring_o_enabled: bool = True
                boring_0: Optional[BoringConfigurable]
                boring_0_enabled: bool = False

                def __post_init__(self):
                    run_auto_creation(self)

            if fake:
                MyClass.create_fruit_impl = fake_impl
                MyClass.create_fruit_o_impl = fake_impl
                MyClass.create_boring_impl = fake_impl
                MyClass.create_boring_o_impl = fake_impl

            expand_args_fields(MyClass)
            instance = MyClass()
            for name in ["fruit", "fruit_o", "boring", "boring_o"]:
                self.assertEqual(
                    hasattr(instance, name), not fake, msg=f"{name} {fake}"
                )

            self.assertIsNone(instance.fruit_0)
            self.assertIsNone(instance.boring_0)
            if not fake:
                self.assertIsInstance(instance.fruit, Orange)
                self.assertIsInstance(instance.fruit_o, Orange)
                self.assertIsInstance(instance.boring, BoringConfigurable)
                self.assertIsInstance(instance.boring_o, BoringConfigurable)

        self.assertEqual(control_args, ["Orange", "Orange", True, True])

    def test_pre_expand(self):
        # Check that the precreate method of a class is called once before
        # when expand_args_fields is called on the class.

        class A(Configurable):
            n: int = 9

            @classmethod
            def pre_expand(cls):
                pass

        A.pre_expand = Mock()
        expand_args_fields(A)
        A.pre_expand.assert_called()

    def test_pre_expand_replaceable(self):
        # Check that the precreate method of a class is called once before
        # when expand_args_fields is called on the class.

        class A(ReplaceableBase):
            pass

            @classmethod
            def pre_expand(cls):
                pass

        class A1(A):
            n: 9

        A.pre_expand = Mock()
        expand_args_fields(A1)
        A.pre_expand.assert_called()


@dataclass(eq=False)
class MockDataclass:
    field_no_default: int
    field_primitive_type: int = 42
    field_optional_none: Optional[int] = None
    field_optional_dict_none: Optional[Dict] = None
    field_optional_with_value: Optional[int] = 42
    field_list_type: List[int] = field(default_factory=lambda: [])


class RefObject:
    pass


REF_OBJECT = RefObject()


class MockClassWithInit:  # noqa: B903
    def __init__(
        self,
        field_no_nothing,
        field_no_default: int,
        field_primitive_type: int = 42,
        field_optional_none: Optional[int] = None,
        field_optional_dict_none: Optional[Dict] = None,
        field_optional_with_value: Optional[int] = 42,
        field_list_type: List[int] = [],  # noqa: B006
        field_reference_type: RefObject = REF_OBJECT,
    ):
        self.field_no_nothing = field_no_nothing
        self.field_no_default = field_no_default
        self.field_primitive_type = field_primitive_type
        self.field_optional_none = field_optional_none
        self.field_optional_dict_none = field_optional_dict_none
        self.field_optional_with_value = field_optional_with_value
        self.field_list_type = field_list_type
        self.field_reference_type = field_reference_type


enable_get_default_args(MockClassWithInit)


class TestRawClasses(unittest.TestCase):
    def setUp(self) -> None:
        self._instances = {
            MockDataclass: MockDataclass(field_no_default=0),
            MockClassWithInit: MockClassWithInit(
                field_no_nothing="tratata", field_no_default=0
            ),
        }

    def test_get_default_args(self):
        for cls in [MockDataclass, MockClassWithInit]:
            dataclass_defaults = get_default_args(cls)
            # DictConfig fields with missing values are `not in`
            self.assertNotIn("field_no_default", dataclass_defaults)
            self.assertNotIn("field_no_nothing", dataclass_defaults)
            self.assertNotIn("field_reference_type", dataclass_defaults)
            expected_defaults = [
                "field_primitive_type",
                "field_optional_none",
                "field_optional_dict_none",
                "field_optional_with_value",
                "field_list_type",
            ]

            if cls == MockDataclass:  # we don't remove undefaulted from dataclasses
                dataclass_defaults.field_no_default = 0
                expected_defaults.insert(0, "field_no_default")
            self.assertEqual(list(dataclass_defaults), expected_defaults)
            for name, val in dataclass_defaults.items():
                self.assertTrue(hasattr(self._instances[cls], name))
                self.assertEqual(val, getattr(self._instances[cls], name))

    def test_get_default_args_readonly(self):
        for cls in [MockDataclass, MockClassWithInit]:
            dataclass_defaults = get_default_args(cls)
            dataclass_defaults["field_list_type"].append(13)
            self.assertEqual(self._instances[cls].field_list_type, [])
