# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import inspect
import itertools
import sys
import warnings
from collections import Counter, defaultdict
from enum import Enum
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    get_args,
    get_origin,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from omegaconf import DictConfig, OmegaConf, open_dict


"""
This functionality allows a configurable system to be determined in a dataclass-type
way. It is a generalization of omegaconf's "structured", in the dataclass case.
Core functionality:

- Configurable -- A base class used to label a class as being one which uses this
                    system. Uses class members and __post_init__ like a dataclass.

- expand_args_fields -- Expands a class like `dataclasses.dataclass`. Runs automatically.

- get_default_args -- gets an omegaconf.DictConfig for initializing a given class.

- run_auto_creation -- Initialises nested members. To be called in __post_init__.


In addition, a Configurable may contain members whose type is decided at runtime.

- ReplaceableBase -- As a base instead of Configurable, labels a class to say that
                     any child class can be used instead.

- registry -- A global store of named child classes of  ReplaceableBase classes.
              Used as `@registry.register` decorator on class definition.


Additional utility functions:

- remove_unused_components -- used for simplifying a DictConfig instance.
- get_default_args_field -- default for DictConfig member of another configurable.
- enable_get_default_args -- Allows get_default_args on a function or plain class.


1. The simplest usage of this functionality is as follows. First a schema is defined
in dataclass style.

    class A(Configurable):
        n: int = 9

    class B(Configurable):
        a: A

        def __post_init__(self):
            run_auto_creation(self)

Then it can be used like

    b_args = get_default_args(B)
    b = B(**b_args)

In this case, get_default_args(B) returns an omegaconf.DictConfig with the right
members {"a_args": {"n": 9}}. It also modifies the definitions of the classes to
something like the following. (The modification itself is done by the function
`expand_args_fields`, which is called inside `get_default_args`.)

    @dataclasses.dataclass
    class A:
        n: int = 9

    @dataclasses.dataclass
    class B:
        a_args: DictConfig = dataclasses.field(default_factory=lambda: DictConfig({"n": 9}))

        def __post_init__(self):
            self.a = A(**self.a_args)

2. Pluggability. Instead of a dataclass-style member being given a concrete class,
it can be given a base class and the implementation will be looked up by name in the
global `registry` in this module. E.g.

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
        b: Optional[A]
        b_class_type: Optional[str] = "A2"

        def __post_init__(self):
            run_auto_creation(self)

will expand to

    @dataclasses.dataclass
    class A:
        k: int = 1

    @dataclasses.dataclass
    class A1(A):
        m: int = 3

    @dataclasses.dataclass
    class A2(A):
        n: str = "2"

    @dataclasses.dataclass
    class B:
        a_class_type: str = "A2"
        a_A1_args: DictConfig = dataclasses.field(
            default_factory=lambda: DictConfig({"k": 1, "m": 3}
        )
        a_A2_args: DictConfig = dataclasses.field(
            default_factory=lambda: DictConfig({"k": 1, "n": 2}
        )
        b_class_type: Optional[str] = "A2"
        b_A1_args: DictConfig = dataclasses.field(
            default_factory=lambda: DictConfig({"k": 1, "m": 3}
        )
        b_A2_args: DictConfig = dataclasses.field(
            default_factory=lambda: DictConfig({"k": 1, "n": 2}
        )

        def __post_init__(self):
            if self.a_class_type == "A1":
                self.a = A1(**self.a_A1_args)
            elif self.a_class_type == "A2":
                self.a = A2(**self.a_A2_args)
            else:
                raise ValueError(...)

            if self.b_class_type is None:
                self.b = None
            elif self.b_class_type == "A1":
                self.b = A1(**self.b_A1_args)
            elif self.b_class_type == "A2":
                self.b = A2(**self.b_A2_args)
            else:
                raise ValueError(...)

3. Aside from these classes, the members of these classes should be things
which DictConfig is happy with: e.g. (bool, int, str, None, float) and what
can be built from them with `DictConfig`s and lists of them.

In addition, you can call `get_default_args` on a function or class to get
the `DictConfig` of its defaulted arguments, assuming those are all things
which `DictConfig` is happy with, so long as you add a call to
`enable_get_default_args` after its definition. If you want to use such a
thing as the default for a member of another configured class,
`get_default_args_field` is a helper.
"""


TYPE_SUFFIX: str = "_class_type"
ARGS_SUFFIX: str = "_args"
ENABLED_SUFFIX: str = "_enabled"
CREATE_PREFIX: str = "create_"
IMPL_SUFFIX: str = "_impl"
TWEAK_SUFFIX: str = "_tweak_args"
_DATACLASS_INIT: str = "__dataclass_own_init__"
PRE_EXPAND_NAME: str = "pre_expand"


class ReplaceableBase:
    """
    Base class for a class (a "replaceable") which is a base class for
    dataclass-style implementations. The implementations can be stored
    in the registry. They get expanded into dataclasses with expand_args_fields.
    This expansion is delayed.
    """

    def __new__(cls, *args, **kwargs):
        """
        These classes should be expanded only when needed (because processing
        fixes the list of replaceable subclasses of members of the class). It
        is safer if users expand the classes explicitly. But if the class gets
        instantiated when it hasn't been processed, we expand it here.
        """
        obj = super().__new__(cls)
        if cls is not ReplaceableBase and not _is_actually_dataclass(cls):
            expand_args_fields(cls)
        return obj


class Configurable:
    """
    Base class for dataclass-style classes which are not replaceable. These get
    expanded into a dataclass with expand_args_fields.
    This expansion is delayed.
    """

    def __new__(cls, *args, **kwargs):
        """
        These classes should be expanded only when needed (because processing
        fixes the list of replaceable subclasses of members of the class). It
        is safer if users expand the classes explicitly. But if the class gets
        instantiated when it hasn't been processed, we expand it here.
        """
        obj = super().__new__(cls)
        if cls is not Configurable and not _is_actually_dataclass(cls):
            expand_args_fields(cls)
        return obj


_X = TypeVar("X", bound=ReplaceableBase)
_Y = TypeVar("Y", bound=Union[ReplaceableBase, Configurable])


class _Registry:
    """
    Register from names to classes. In particular, we say that direct subclasses of
    ReplaceableBase are "base classes" and we register subclasses of each base class
    in a separate namespace.
    """

    def __init__(self) -> None:
        self._mapping: Dict[
            Type[ReplaceableBase], Dict[str, Type[ReplaceableBase]]
        ] = defaultdict(dict)

    def register(self, some_class: Type[_X]) -> Type[_X]:
        """
        A class decorator, to register a class in self.
        """
        name = some_class.__name__
        self._register(some_class, name=name)
        return some_class

    def _register(
        self,
        some_class: Type[ReplaceableBase],
        *,
        base_class: Optional[Type[ReplaceableBase]] = None,
        name: str,
    ) -> None:
        """
        Register a new member.

        Args:
            cls: the new member
            base_class: (optional) what the new member is a type for
            name: name for the new member
        """
        if base_class is None:
            base_class = self._base_class_from_class(some_class)
            if base_class is None:
                raise ValueError(
                    f"Cannot register {some_class}. Cannot tell what it is."
                )
        self._mapping[base_class][name] = some_class

    def get(self, base_class_wanted: Type[_X], name: str) -> Type[_X]:
        """
        Retrieve a class from the registry by name

        Args:
            base_class_wanted: parent type of type we are looking for.
                        It determines the namespace.
                        This will typically be a direct subclass of ReplaceableBase.
            name: what to look for

        Returns:
            class type
        """
        if self._is_base_class(base_class_wanted):
            base_class = base_class_wanted
        else:
            base_class = self._base_class_from_class(base_class_wanted)
            if base_class is None:
                raise ValueError(
                    f"Cannot look up {base_class_wanted}. Cannot tell what it is."
                )
        if not isinstance(name, str):
            raise ValueError(
                f"Cannot look up a {type(name)} in the registry. Got {name}."
            )
        result = self._mapping[base_class].get(name)
        if result is None:
            raise ValueError(f"{name} has not been registered.")
        if not issubclass(result, base_class_wanted):
            raise ValueError(
                f"{name} resolves to {result} which does not subclass {base_class_wanted}"
            )
        # pyre-ignore[7]
        return result

    def get_all(
        self, base_class_wanted: Type[ReplaceableBase]
    ) -> List[Type[ReplaceableBase]]:
        """
        Retrieve all registered implementations from the registry

        Args:
            base_class_wanted: parent type of type we are looking for.
                        It determines the namespace.
                        This will typically be a direct subclass of ReplaceableBase.
        Returns:
            list of class types in alphabetical order of registered name.
        """
        if self._is_base_class(base_class_wanted):
            source = self._mapping[base_class_wanted]
            return [source[key] for key in sorted(source)]

        base_class = self._base_class_from_class(base_class_wanted)
        if base_class is None:
            raise ValueError(
                f"Cannot look up {base_class_wanted}. Cannot tell what it is."
            )
        source = self._mapping[base_class]
        return [
            source[key]
            for key in sorted(source)
            if issubclass(source[key], base_class_wanted)
            and source[key] is not base_class_wanted
        ]

    @staticmethod
    def _is_base_class(some_class: Type[ReplaceableBase]) -> bool:
        """
        Return whether the given type is a direct subclass of ReplaceableBase
        and so gets used as a namespace.
        """
        return ReplaceableBase in some_class.__bases__

    @staticmethod
    def _base_class_from_class(
        some_class: Type[ReplaceableBase],
    ) -> Optional[Type[ReplaceableBase]]:
        """
        Find the parent class of some_class which inherits ReplaceableBase, or None
        """
        for base in some_class.mro()[-3::-1]:
            if base is not ReplaceableBase and issubclass(base, ReplaceableBase):
                return base
        return None


# Global instance of the registry
registry = _Registry()


class _ProcessType(Enum):
    """
    Type of member which gets rewritten by expand_args_fields.
    """

    CONFIGURABLE = 1
    REPLACEABLE = 2
    OPTIONAL_CONFIGURABLE = 3
    OPTIONAL_REPLACEABLE = 4


def _default_create(
    name: str, type_: Type, process_type: _ProcessType
) -> Callable[[Any], None]:
    """
    Return the default creation function for a member. This is a function which
    could be called in __post_init__ to initialise the member, and will be called
    from run_auto_creation.

    Args:
        name: name of the member
        type_: type of the member (with any Optional removed)
        process_type: Shows whether member's declared type inherits ReplaceableBase,
                    in which case the actual type to be created is decided at
                    runtime.

    Returns:
        Function taking one argument, the object whose member should be
            initialized, i.e. self.
    """
    impl_name = f"{CREATE_PREFIX}{name}{IMPL_SUFFIX}"

    def inner(self):
        expand_args_fields(type_)
        impl = getattr(self, impl_name)
        args = getattr(self, name + ARGS_SUFFIX)
        impl(True, args)

    def inner_optional(self):
        expand_args_fields(type_)
        impl = getattr(self, impl_name)
        enabled = getattr(self, name + ENABLED_SUFFIX)
        args = getattr(self, name + ARGS_SUFFIX)
        impl(enabled, args)

    def inner_pluggable(self):
        type_name = getattr(self, name + TYPE_SUFFIX)
        impl = getattr(self, impl_name)
        if type_name is None:
            args = None
        else:
            args = getattr(self, f"{name}_{type_name}{ARGS_SUFFIX}", None)
        impl(type_name, args)

    if process_type == _ProcessType.OPTIONAL_CONFIGURABLE:
        return inner_optional
    return inner if process_type == _ProcessType.CONFIGURABLE else inner_pluggable


def _default_create_impl(
    name: str, type_: Type, process_type: _ProcessType
) -> Callable[[Any, Any, DictConfig], None]:
    """
    Return the default internal function for initialising a member. This is a function
    which could be called in the create_ function to initialise the member.

    Args:
        name: name of the member
        type_: type of the member (with any Optional removed)
        process_type: Shows whether member's declared type inherits ReplaceableBase,
                    in which case the actual type to be created is decided at
                    runtime.

    Returns:
        Function taking
            - self, the object whose member should be initialized.
            - option for what to do. This is
                - for pluggables, the type to initialise or None to do nothing
                - for non pluggables, a bool indicating whether to initialise.
            - the args for initializing the member.
    """

    def create_configurable(self, enabled, args):
        if enabled:
            expand_args_fields(type_)
            setattr(self, name, type_(**args))
        else:
            setattr(self, name, None)

    def create_pluggable(self, type_name, args):
        if type_name is None:
            setattr(self, name, None)
            return

        if not isinstance(type_name, str):
            raise ValueError(
                f"A {type(type_name)} was received as the type of {name}."
                + f" Perhaps this is from {name}{TYPE_SUFFIX}?"
            )
        chosen_class = registry.get(type_, type_name)
        if self._known_implementations.get(type_name, chosen_class) is not chosen_class:
            # If this warning is raised, it means that a new definition of
            # the chosen class has been registered since our class was processed
            # (i.e. expanded). A DictConfig which comes from our get_default_args
            # (which might have triggered the processing) will contain the old default
            # values for the members of the chosen class. Changes to those defaults which
            # were made in the redefinition will not be reflected here.
            warnings.warn(f"New implementation of {type_name} is being chosen.")
        expand_args_fields(chosen_class)
        setattr(self, name, chosen_class(**args))

    if process_type in (_ProcessType.CONFIGURABLE, _ProcessType.OPTIONAL_CONFIGURABLE):
        return create_configurable
    return create_pluggable


def run_auto_creation(self: Any) -> None:
    """
    Run all the functions named in self._creation_functions.
    """
    for create_function in self._creation_functions:
        getattr(self, create_function)()


def _is_configurable_class(C) -> bool:
    return isinstance(C, type) and issubclass(C, (Configurable, ReplaceableBase))


def get_default_args(C, *, _do_not_process: Tuple[type, ...] = ()) -> DictConfig:
    """
    Get the DictConfig corresponding to the defaults in a dataclass or
    configurable. Normal use is to provide a dataclass can be provided as C.
    If enable_get_default_args has been called on a function or plain class,
    then that function or class can be provided as C.

    If C is a subclass of Configurable or ReplaceableBase, we make sure
    it has been processed with expand_args_fields.

    Args:
        C: the class or function to be processed
        _do_not_process: (internal use) When this function is called from
                    expand_args_fields, we specify any class currently being
                    processed, to make sure we don't try to process a class
                    while it is already being processed.

    Returns:
        new DictConfig object, which is typed.
    """
    if C is None:
        return DictConfig({})

    if _is_configurable_class(C):
        if C in _do_not_process:
            raise ValueError(
                f"Internal recursion error. Need processed {C},"
                f" but cannot get it. _do_not_process={_do_not_process}"
            )
        # This is safe to run multiple times. It will return
        # straight away if C has already been processed.
        expand_args_fields(C, _do_not_process=_do_not_process)

    if dataclasses.is_dataclass(C):
        # Note that if get_default_args_field is used somewhere in C,
        # this call is recursive. No special care is needed,
        # because in practice get_default_args_field is used for
        # separate types than the outer type.

        try:
            out: DictConfig = OmegaConf.structured(C)
        except Exception:
            print(f"### OmegaConf.structured({C}) failed ###")
            # We don't use `raise From` here, because that gets the original
            # exception hidden by the OC_CAUSE logic in the case where we are
            # called by hydra.
            raise
        exclude = getattr(C, "_processed_members", ())
        with open_dict(out):
            for field in exclude:
                out.pop(field, None)
        return out

    if _is_configurable_class(C):
        raise ValueError(f"Failed to process {C}")

    if not inspect.isfunction(C) and not inspect.isclass(C):
        raise ValueError(f"Unexpected {C}")

    dataclass_name = _dataclass_name_for_function(C)
    dataclass = getattr(sys.modules[C.__module__], dataclass_name, None)
    if dataclass is None:
        raise ValueError(
            f"Cannot get args for {C}. Was enable_get_default_args forgotten?"
        )

    try:
        out: DictConfig = OmegaConf.structured(dataclass)
    except Exception:
        print(f"### OmegaConf.structured failed for {C.__name__} ###")
        raise
    return out


def _dataclass_name_for_function(C: Any) -> str:
    """
    Returns the name of the dataclass which enable_get_default_args(C)
    creates.
    """
    name = f"_{C.__name__}_default_args_"
    return name


def _field_annotations_for_default_args(
    C: Any,
) -> List[Tuple[str, Any, dataclasses.Field]]:
    """
    If C is a function or a plain class with an __init__ function,
    return the fields which `enable_get_default_args(C)` will need
    to make a dataclass with.

    Args:
        C: a function, or a class with an __init__ function. Must
            have types for all its defaulted args.

    Returns:
        a list of fields for a dataclass.
    """

    field_annotations = []
    for pname, defval in _params_iter(C):
        default = defval.default
        if default == inspect.Parameter.empty:
            # we do not have a default value for the parameter
            continue

        if defval.annotation == inspect._empty:
            raise ValueError(
                "All arguments of the input to enable_get_default_args have to"
                f" be typed. Argument '{pname}' does not have a type annotation."
            )

        _, annotation = _resolve_optional(defval.annotation)

        if isinstance(default, set):  # force OmegaConf to convert it to ListConfig
            default = tuple(default)

        if isinstance(default, (list, dict)):
            # OmegaConf will convert to [Dict|List]Config, so it is safe to reuse the value
            field_ = dataclasses.field(default_factory=lambda default=default: default)
        elif not _is_immutable_type(annotation, default):
            continue
        else:
            # we can use a simple default argument for dataclass.field
            field_ = dataclasses.field(default=default)
        field_annotations.append((pname, defval.annotation, field_))

    return field_annotations


def enable_get_default_args(C: Any, *, overwrite: bool = True) -> None:
    """
    If C is a function or a plain class with an __init__ function,
    and you want get_default_args(C) to work, then add
    `enable_get_default_args(C)` straight after the definition of C.
    This makes a dataclass corresponding to the default arguments of C
    and stores it in the same module as C.

    Args:
        C: a function, or a class with an __init__ function. Must
            have types for all its defaulted args.
        overwrite: whether to allow calling this a second time on
            the same function.
    """
    if not inspect.isfunction(C) and not inspect.isclass(C):
        raise ValueError(f"Unexpected {C}")

    field_annotations = _field_annotations_for_default_args(C)

    name = _dataclass_name_for_function(C)
    module = sys.modules[C.__module__]
    if hasattr(module, name):
        if overwrite:
            warnings.warn(f"Overwriting {name} in {C.__module__}.")
        else:
            raise ValueError(f"Cannot overwrite {name} in {C.__module__}.")
    dc = dataclasses.make_dataclass(name, field_annotations)
    dc.__module__ = C.__module__
    setattr(module, name, dc)


def _params_iter(C):
    """Returns dict of keyword args of a class or function C."""
    if inspect.isclass(C):
        return itertools.islice(  # exclude `self`
            inspect.signature(C.__init__).parameters.items(), 1, None
        )

    return inspect.signature(C).parameters.items()


def _is_immutable_type(type_: Type, val: Any) -> bool:
    if val is None:
        return True

    PRIMITIVE_TYPES = (int, float, bool, str, bytes, tuple)
    # sometimes type can be too relaxed (e.g. Any), so we also check values
    if isinstance(val, PRIMITIVE_TYPES):
        return True

    return type_ in PRIMITIVE_TYPES or (
        inspect.isclass(type_) and issubclass(type_, Enum)
    )


# copied from OmegaConf
def _resolve_optional(type_: Any) -> Tuple[bool, Any]:
    """Check whether `type_` is equivalent to `typing.Optional[T]` for some T."""
    if get_origin(type_) is Union:
        args = get_args(type_)
        if len(args) == 2 and args[1] == type(None):  # noqa E721
            return True, args[0]
    if type_ is Any:
        return True, Any

    return False, type_


def _is_actually_dataclass(some_class) -> bool:
    # Return whether the class some_class has been processed with
    # the dataclass annotation. This is more specific than
    # dataclasses.is_dataclass which returns True on anything
    # deriving from a dataclass.

    # Checking for __init__ would also work for our purpose.
    return "__dataclass_fields__" in some_class.__dict__


def expand_args_fields(
    some_class: Type[_Y], *, _do_not_process: Tuple[type, ...] = ()
) -> Type[_Y]:
    """
    This expands a class which inherits Configurable or ReplaceableBase classes,
    including dataclass processing. some_class is modified in place by this function.
    If expand_args_fields(some_class) has already been called, subsequent calls do
    nothing and return some_class unmodified.
    For classes of type ReplaceableBase, you can add some_class to the registry before
    or after calling this function. But potential inner classes need to be registered
    before this function is run on the outer class.

    The transformations this function makes, before the concluding
    dataclasses.dataclass, are as follows. If X is a base class with registered
    subclasses Y and Z, replace a class member

        x: X

    and optionally

        x_class_type: str = "Y"
        def create_x(self):...

    with

        x_Y_args: dict = dataclasses.field(default_factory=lambda: get_default_args(Y))
        x_Z_args: dict = dataclasses.field(default_factory=lambda: get_default_args(Z))
        def create_x(self):
            args = self.getattr(f"x_{self.x_class_type}_args")
            self.create_x_impl(self.x_class_type, args)
        def create_x_impl(self, x_type, args):
            x_type = registry.get(X, x_type)
            expand_args_fields(x_type)
            self.x = x_type(**args)
        x_class_type: str = "UNDEFAULTED"

    without adding the optional attributes if they are already there.

    Similarly, replace

        x: Optional[X]

    and optionally

        x_class_type: Optional[str] = "Y"
        def create_x(self):...

    with

        x_Y_args: dict = dataclasses.field(default_factory=lambda: get_default_args(Y))
        x_Z_args: dict = dataclasses.field(default_factory=lambda: get_default_args(Z))
        def create_x(self):
            if self.x_class_type is None:
                args = None
            else:
                args = self.getattr(f"x_{self.x_class_type}_args", None)
            self.create_x_impl(self.x_class_type, args)
        def create_x_impl(self, x_class_type, args):
            if x_class_type is None:
                self.x = None
                return

            x_type = registry.get(X, x_class_type)
            expand_args_fields(x_type)
            assert args is not None
            self.x = x_type(**args)
        x_class_type: Optional[str] = "UNDEFAULTED"

    without adding the optional attributes if they are already there.

    Similarly, if X is a subclass of Configurable,

        x: X

    and optionally

        def create_x(self):...

    will be replaced with

        x_args: dict = dataclasses.field(default_factory=lambda: get_default_args(X))
        def create_x(self):
            self.create_x_impl(True, self.x_args)

        def create_x_impl(self, enabled, args):
            if enabled:
                expand_args_fields(X)
                self.x = X(**args)
            else:
                self.x = None

    Similarly, replace,

        x: Optional[X]
        x_enabled: bool = ...

    and optionally

        def create_x(self):...

    with

        x_args: dict = dataclasses.field(default_factory=lambda: get_default_args(X))
        x_enabled: bool = ...
        def create_x(self):
            self.create_x_impl(self.x_enabled, self.x_args)

        def create_x_impl(self, enabled, args):
            if enabled:
                expand_args_fields(X)
                self.x = X(**args)
            else:
                self.x = None


    Also adds the following class members, unannotated so that dataclass
    ignores them.
        - _creation_functions: Tuple[str, ...] of all the create_ functions,
            including those from base classes (not the create_x_impl ones).
        - _known_implementations: Dict[str, Type] containing the classes which
            have been found from the registry.
            (used only to raise a warning if it one has been overwritten)
        - _processed_members: a Dict[str, Any] of all the members which have been
            transformed, with values giving the types they were declared to have.
            (E.g. {"x": X} or {"x": Optional[X]} in the cases above.)

    In addition, if the class has a member function

        @classmethod
        def x_tweak_args(cls, member_type: Type, args: DictConfig) -> None

    then the default_factory of x_args will also have a call to x_tweak_args(X, x_args) and
    the default_factory of x_Y_args will also have a call to x_tweak_args(Y, x_Y_args).

    In addition, if the class inherits torch.nn.Module, the generated __init__ will
    call torch.nn.Module's __init__ before doing anything else.

    Before any transformation of the class, if the class has a classmethod called
    `pre_expand`, it will be called with no arguments.

    Note that although the *_args members are intended to have type DictConfig, they
    are actually internally annotated as dicts. OmegaConf is happy to see a DictConfig
    in place of a dict, but not vice-versa. Allowing dict lets a class user specify
    x_args as an explicit dict without getting an incomprehensible error.

    Args:
        some_class: the class to be processed
        _do_not_process: Internal use for get_default_args: Because get_default_args calls
                        and is called by this function, we let it specify any class currently
                        being processed, to make sure we don't try to process a class while
                        it is already being processed.


    Returns:
        some_class itself, which has been modified in place. This
        allows this function to be used as a class decorator.
    """
    if _is_actually_dataclass(some_class):
        return some_class

    if hasattr(some_class, PRE_EXPAND_NAME):
        getattr(some_class, PRE_EXPAND_NAME)()

    # The functions this class's run_auto_creation will run.
    creation_functions: List[str] = []
    # The classes which this type knows about from the registry
    # We could use a weakref.WeakValueDictionary here which would mean
    # that we don't warn if the class we should have expected is elsewhere
    # unused.
    known_implementations: Dict[str, Type] = {}
    # Names of members which have been processed.
    processed_members: Dict[str, Any] = {}

    # For all bases except ReplaceableBase and Configurable and object,
    # we need to process them before our own processing. This is
    # because dataclasses expect to inherit dataclasses and not unprocessed
    # dataclasses.
    for base in some_class.mro()[-3:0:-1]:
        if base is ReplaceableBase:
            continue
        if base is Configurable:
            continue
        if not issubclass(base, (Configurable, ReplaceableBase)):
            continue
        expand_args_fields(base, _do_not_process=_do_not_process)
        if "_creation_functions" in base.__dict__:
            creation_functions.extend(base._creation_functions)
        if "_known_implementations" in base.__dict__:
            known_implementations.update(base._known_implementations)
        if "_processed_members" in base.__dict__:
            processed_members.update(base._processed_members)

    to_process: List[Tuple[str, Type, _ProcessType]] = []
    if "__annotations__" in some_class.__dict__:
        for name, type_ in some_class.__annotations__.items():
            underlying_and_process_type = _get_type_to_process(type_)
            if underlying_and_process_type is None:
                continue
            underlying_type, process_type = underlying_and_process_type
            to_process.append((name, underlying_type, process_type))

    for name, underlying_type, process_type in to_process:
        processed_members[name] = some_class.__annotations__[name]
        _process_member(
            name=name,
            type_=underlying_type,
            process_type=process_type,
            some_class=some_class,
            creation_functions=creation_functions,
            _do_not_process=_do_not_process,
            known_implementations=known_implementations,
        )

    for key, count in Counter(creation_functions).items():
        if count > 1:
            warnings.warn(f"Clash with {key} in a base class.")
    some_class._creation_functions = tuple(creation_functions)
    some_class._processed_members = processed_members
    some_class._known_implementations = known_implementations

    dataclasses.dataclass(eq=False)(some_class)
    _fixup_class_init(some_class)
    return some_class


def _fixup_class_init(some_class) -> None:
    """
    In-place modification of the some_class class which happens
    after dataclass processing.

    If the dataclass some_class inherits torch.nn.Module, then
    makes torch.nn.Module's __init__ be called before anything else
    on instantiation of some_class.
    This is a bit like attr's __pre_init__.
    """

    assert _is_actually_dataclass(some_class)
    try:
        import torch
    except ModuleNotFoundError:
        return

    if not issubclass(some_class, torch.nn.Module):
        return

    def init(self, *args, **kwargs) -> None:
        torch.nn.Module.__init__(self)
        getattr(self, _DATACLASS_INIT)(*args, **kwargs)

    assert _DATACLASS_INIT not in some_class.__dict__

    setattr(some_class, _DATACLASS_INIT, some_class.__init__)
    some_class.__init__ = init


def get_default_args_field(
    C,
    *,
    _do_not_process: Tuple[type, ...] = (),
    _hook: Optional[Callable[[DictConfig], None]] = None,
):
    """
    Get a dataclass field which defaults to get_default_args(...)

    Args:
        C: As for get_default_args.
        _do_not_process: As for get_default_args
        _hook: Function called on the result before returning.

    Returns:
        function to return new DictConfig object
    """

    def create():
        args = get_default_args(C, _do_not_process=_do_not_process)
        if _hook is not None:
            with open_dict(args):
                _hook(args)
        return args

    return dataclasses.field(default_factory=create)


def _get_default_args_field_from_registry(
    *,
    base_class_wanted: Type[_X],
    name: str,
    _do_not_process: Tuple[type, ...] = (),
    _hook: Optional[Callable[[DictConfig], None]] = None,
):
    """
    Get a dataclass field which defaults to
    get_default_args(registry.get(base_class_wanted, name)).

    This is used internally in place of get_default_args_field in
    order that default values are updated if a class is redefined.

    Args:
        base_class_wanted: As for registry.get.
        name: As for registry.get.
        _do_not_process: As for get_default_args
        _hook: Function called on the result before returning.

    Returns:
        function to return new DictConfig object
    """

    def create():
        C = registry.get(base_class_wanted=base_class_wanted, name=name)
        args = get_default_args(C, _do_not_process=_do_not_process)
        if _hook is not None:
            with open_dict(args):
                _hook(args)
        return args

    return dataclasses.field(default_factory=create)


def _get_type_to_process(type_) -> Optional[Tuple[Type, _ProcessType]]:
    """
    If a member is annotated as `type_`, and that should expanded in
    expand_args_fields, return how it should be expanded.
    """
    if get_origin(type_) == Union:
        # We look for Optional[X] which is a Union of X with None.
        args = get_args(type_)
        if len(args) != 2 or all(a is not type(None) for a in args):  # noqa: E721
            return
        underlying = args[0] if args[1] is type(None) else args[1]  # noqa: E721
        if (
            isinstance(underlying, type)
            and issubclass(underlying, ReplaceableBase)
            and ReplaceableBase in underlying.__bases__
        ):
            return underlying, _ProcessType.OPTIONAL_REPLACEABLE

        if isinstance(underlying, type) and issubclass(underlying, Configurable):
            return underlying, _ProcessType.OPTIONAL_CONFIGURABLE

    if not isinstance(type_, type):
        # e.g. any other Union or Tuple. Or ClassVar.
        return

    if issubclass(type_, ReplaceableBase) and ReplaceableBase in type_.__bases__:
        return type_, _ProcessType.REPLACEABLE

    if issubclass(type_, Configurable):
        return type_, _ProcessType.CONFIGURABLE


def _process_member(
    *,
    name: str,
    type_: Type,
    process_type: _ProcessType,
    some_class: Type,
    creation_functions: List[str],
    _do_not_process: Tuple[type, ...],
    known_implementations: Dict[str, Type],
) -> None:
    """
    Make the modification (of expand_args_fields) to some_class for a single member.

    Args:
        name: member name
        type_: member type (with Optional removed if needed)
        process_type: whether member has dynamic type
        some_class: (MODIFIED IN PLACE) the class being processed
        creation_functions: (MODIFIED IN PLACE) the names of the create functions
        _do_not_process: as for expand_args_fields.
        known_implementations: (MODIFIED IN PLACE) known types from the registry
    """
    # Because we are adding defaultable members, make
    # sure they go at the end of __annotations__ in case
    # there are non-defaulted standard class members.
    del some_class.__annotations__[name]
    hook = getattr(some_class, name + TWEAK_SUFFIX, None)

    if process_type in (_ProcessType.REPLACEABLE, _ProcessType.OPTIONAL_REPLACEABLE):
        type_name = name + TYPE_SUFFIX
        if type_name not in some_class.__annotations__:
            if process_type == _ProcessType.OPTIONAL_REPLACEABLE:
                some_class.__annotations__[type_name] = Optional[str]
            else:
                some_class.__annotations__[type_name] = str
            setattr(some_class, type_name, "UNDEFAULTED")

        for derived_type in registry.get_all(type_):
            if derived_type in _do_not_process:
                continue
            if issubclass(derived_type, some_class):
                # When derived_type is some_class we have a simple
                # recursion to avoid. When it's a strict subclass the
                # situation is even worse.
                continue
            known_implementations[derived_type.__name__] = derived_type
            args_name = f"{name}_{derived_type.__name__}{ARGS_SUFFIX}"
            if args_name in some_class.__annotations__:
                raise ValueError(
                    f"Cannot generate {args_name} because it is already present."
                )
            some_class.__annotations__[args_name] = dict
            if hook is not None:
                hook_closed = partial(hook, derived_type)
            else:
                hook_closed = None
            setattr(
                some_class,
                args_name,
                _get_default_args_field_from_registry(
                    base_class_wanted=type_,
                    name=derived_type.__name__,
                    _do_not_process=_do_not_process + (some_class,),
                    _hook=hook_closed,
                ),
            )
    else:
        args_name = name + ARGS_SUFFIX
        if args_name in some_class.__annotations__:
            raise ValueError(
                f"Cannot generate {args_name} because it is already present."
            )
        if issubclass(type_, some_class) or type_ in _do_not_process:
            raise ValueError(f"Cannot process {type_} inside {some_class}")

        some_class.__annotations__[args_name] = dict
        if hook is not None:
            hook_closed = partial(hook, type_)
        else:
            hook_closed = None
        setattr(
            some_class,
            args_name,
            get_default_args_field(
                type_,
                _do_not_process=_do_not_process + (some_class,),
                _hook=hook_closed,
            ),
        )
        if process_type == _ProcessType.OPTIONAL_CONFIGURABLE:
            enabled_name = name + ENABLED_SUFFIX
            if enabled_name not in some_class.__annotations__:
                raise ValueError(
                    f"{name} is an Optional[{type_.__name__}] member "
                    f"but there is no corresponding member {enabled_name}."
                )

    creation_function_name = f"{CREATE_PREFIX}{name}"
    if not hasattr(some_class, creation_function_name):
        setattr(
            some_class,
            creation_function_name,
            _default_create(name, type_, process_type),
        )
    creation_functions.append(creation_function_name)

    creation_function_impl_name = f"{CREATE_PREFIX}{name}{IMPL_SUFFIX}"
    if not hasattr(some_class, creation_function_impl_name):
        setattr(
            some_class,
            creation_function_impl_name,
            _default_create_impl(name, type_, process_type),
        )


def remove_unused_components(dict_: DictConfig) -> None:
    """
    Assuming dict_ represents the state of a configurable,
    modify it to remove all the portions corresponding to
    pluggable parts which are not in use.
    For example, if renderer_class_type is SignedDistanceFunctionRenderer,
    the renderer_MultiPassEmissionAbsorptionRenderer_args will be
    removed. Also, if chocolate_enabled is False, then chocolate_args will
    be removed.

    Args:
        dict_: (MODIFIED IN PLACE) a DictConfig instance
    """
    keys = [key for key in dict_ if isinstance(key, str)]
    suffix_length = len(TYPE_SUFFIX)
    replaceables = [key[:-suffix_length] for key in keys if key.endswith(TYPE_SUFFIX)]
    args_keys = [key for key in keys if key.endswith(ARGS_SUFFIX)]
    for replaceable in replaceables:
        selected_type = dict_[replaceable + TYPE_SUFFIX]
        if selected_type is None:
            expect = ""
        else:
            expect = replaceable + "_" + selected_type + ARGS_SUFFIX
        with open_dict(dict_):
            for key in args_keys:
                if key.startswith(replaceable + "_") and key != expect:
                    del dict_[key]

    suffix_length = len(ENABLED_SUFFIX)
    enableables = [key[:-suffix_length] for key in keys if key.endswith(ENABLED_SUFFIX)]
    for enableable in enableables:
        enabled = dict_[enableable + ENABLED_SUFFIX]
        if not enabled:
            with open_dict(dict_):
                dict_.pop(enableable + ARGS_SUFFIX, None)

    for key in dict_:
        if isinstance(dict_.get(key), DictConfig):
            remove_unused_components(dict_[key])
