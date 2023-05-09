
# coding: utf-8

# In[ ]:


# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.


# # Implicitron's config system

# Implicitron's components are all based on a unified hierarchical configuration system. 
# This allows configurable variables and all defaults to be defined separately for each new component.
# All configs relevant to an experiment are then automatically composed into a single configuration file that fully specifies the experiment.
# An especially important feature is extension points where users can insert their own sub-classes of Implicitron's base components.
# 
# The file which defines this system is [here](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/implicitron/tools/config.py) in the PyTorch3D repo.
# The Implicitron volumes tutorial contains a simple example of using the config system.
# This tutorial provides detailed hands-on experience in using and modifying Implicitron's configurable components.
# 

# ## 0. Install and import modules
# 
# Ensure `torch` and `torchvision` are installed. If `pytorch3d` is not installed, install it using the following cell:

# In[ ]:


import os
import sys
import torch
need_pytorch3d=False
try:
    import pytorch3d
except ModuleNotFoundError:
    need_pytorch3d=True
if need_pytorch3d:
    if torch.__version__.startswith(("1.13.", "2.0.")) and sys.platform.startswith("linux"):
        # We try to install PyTorch3D via a released wheel.
        pyt_version_str=torch.__version__.split("+")[0].replace(".", "")
        version_str="".join([
            f"py3{sys.version_info.minor}_cu",
            torch.version.cuda.replace(".",""),
            f"_pyt{pyt_version_str}"
        ])
        get_ipython().system('pip install fvcore iopath')
        get_ipython().system('pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html')
    else:
        # We try to install PyTorch3D from source.
        get_ipython().system("pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'")


# Ensure omegaconf is installed. If not, run this cell. (It should not be necessary to restart the runtime.)

# In[ ]:


get_ipython().system('pip install omegaconf')


# In[ ]:


from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch3d.implicitron.tools.config import (
    Configurable,
    ReplaceableBase,
    expand_args_fields,
    get_default_args,
    registry,
    run_auto_creation,
)


# ## 1. Introducing dataclasses 
# 
# [Type hints](https://docs.python.org/3/library/typing.html) give a taxonomy of types in Python. [Dataclasses](https://docs.python.org/3/library/dataclasses.html) let you create a class based on a list of members which have names, types and possibly default values. The `__init__` function is created automatically, and calls a `__post_init__` function if present as a final step. For example

# In[ ]:


@dataclass
class MyDataclass:
    a: int
    b: int = 8
    c: Optional[Tuple[int, ...]] = None

    def __post_init__(self):
        print(f"created with a = {self.a}")
        self.d = 2 * self.b


# In[ ]:


my_dataclass_instance = MyDataclass(a=18)
assert my_dataclass_instance.d == 16


# 👷 Note that the `dataclass` decorator here is function which modifies the definition of the class itself.
# It runs immediately after the definition.
# Our config system requires that implicitron library code contains classes whose modified versions need to be aware of user-defined implementations.
# Therefore we need the modification of the class to be delayed. We don't use a decorator.
# 

# ## 2. Introducing omegaconf and OmegaConf.structured
# 
# The [omegaconf](https://github.com/omry/omegaconf/) library provides a DictConfig class which is like a `dict` with str keys, but with extra features for ease-of-use as a configuration system.

# In[ ]:


dc = DictConfig({"a": 2, "b": True, "c": None, "d": "hello"})
assert dc.a == dc["a"] == 2


# OmegaConf has a serialization to and from yaml. The [Hydra](https://hydra.cc/) library relies on this for its configuration files.

# In[ ]:


print(OmegaConf.to_yaml(dc))
assert OmegaConf.create(OmegaConf.to_yaml(dc)) == dc


# OmegaConf.structured provides a DictConfig from a dataclass or instance of a dataclass. Unlike a normal DictConfig, it is type-checked and only known keys can be added.

# In[ ]:


structured = OmegaConf.structured(MyDataclass)
assert isinstance(structured, DictConfig)
print(structured)
print()
print(OmegaConf.to_yaml(structured))


# `structured` knows it is missing a value for `a`.

# Such an object has members compatible with the dataclass, so an initialisation can be performed as follows.

# In[ ]:


structured.a = 21
my_dataclass_instance2 = MyDataclass(**structured)
print(my_dataclass_instance2)


# You can also call OmegaConf.structured on an instance.

# In[ ]:


structured_from_instance = OmegaConf.structured(my_dataclass_instance)
my_dataclass_instance3 = MyDataclass(**structured_from_instance)
print(my_dataclass_instance3)


# ## 3. Our approach to OmegaConf.structured
# 
# We provide functions which are equivalent to `OmegaConf.structured` but support more features. 
# To achieve the above using our functions, the following is used.
# Note that we indicate configurable classes using a special base class `Configurable`, not a decorator.

# In[ ]:


class MyConfigurable(Configurable):
    a: int
    b: int = 8
    c: Optional[Tuple[int, ...]] = None

    def __post_init__(self):
        print(f"created with a = {self.a}")
        self.d = 2 * self.b


# In[ ]:


# The expand_args_fields function modifies the class like @dataclasses.dataclass.
# If it has not been called on a Configurable object before it has been instantiated, it will
# be called automatically.
expand_args_fields(MyConfigurable)
my_configurable_instance = MyConfigurable(a=18)
assert my_configurable_instance.d == 16


# In[ ]:


# get_default_args also calls expand_args_fields automatically
our_structured = get_default_args(MyConfigurable)
assert isinstance(our_structured, DictConfig)
print(OmegaConf.to_yaml(our_structured))


# In[ ]:


our_structured.a = 21
print(MyConfigurable(**our_structured))


# ## 4. First enhancement: nested types 🪺
# 
# Our system allows Configurable classes to contain each other. 
# One thing to remember: add a call to `run_auto_creation` in `__post_init__`.

# In[ ]:


class Inner(Configurable):
    a: int = 8
    b: bool = True
    c: Tuple[int, ...] = (2, 3, 4, 6)


class Outer(Configurable):
    inner: Inner
    x: str = "hello"
    xx: bool = False

    def __post_init__(self):
        run_auto_creation(self)


# In[ ]:


outer_dc = get_default_args(Outer)
print(OmegaConf.to_yaml(outer_dc))


# In[ ]:


outer = Outer(**outer_dc)
assert isinstance(outer, Outer)
assert isinstance(outer.inner, Inner)
print(vars(outer))
print(outer.inner)


# Note how inner_args is an extra member of outer. `run_auto_creation(self)` is equivalent to
# ```
#     self.inner = Inner(**self.inner_args)
# ```

# ## 5. Second enhancement: pluggable/replaceable components 🔌
# 
# If a class uses `ReplaceableBase` as a base class instead of `Configurable`, we call it a replaceable.
# It indicates that it is designed for child classes to use in its place.
# We might use `NotImplementedError` to indicate functionality which subclasses are expected to implement.
# The system maintains a global `registry` containing subclasses of each ReplaceableBase.
# The subclasses register themselves with it with a decorator.
# 
# A configurable class (i.e. a class which uses our system, i.e. a child of `Configurable` or `ReplaceableBase`) which contains a ReplaceableBase must also 
# contain a corresponding class_type field of type `str` which indicates which concrete child class to use.

# In[ ]:


class InnerBase(ReplaceableBase):
    def say_something(self):
        raise NotImplementedError


@registry.register
class Inner1(InnerBase):
    a: int = 1
    b: str = "h"

    def say_something(self):
        print("hello from an Inner1")


@registry.register
class Inner2(InnerBase):
    a: int = 2

    def say_something(self):
        print("hello from an Inner2")


# In[ ]:


class Out(Configurable):
    inner: InnerBase
    inner_class_type: str = "Inner1"
    x: int = 19

    def __post_init__(self):
        run_auto_creation(self)

    def talk(self):
        self.inner.say_something()


# In[ ]:


Out_dc = get_default_args(Out)
print(OmegaConf.to_yaml(Out_dc))


# In[ ]:


Out_dc.inner_class_type = "Inner2"
out = Out(**Out_dc)
print(out.inner)


# In[ ]:


out.talk()


# Note in this case there are many `args` members. It is usually fine to ignore them in the code. They are needed for the config.

# In[ ]:


print(vars(out))


# ## 6. Example with torch.nn.Module  🔥
# Typically in implicitron, we use this system in combination with [`Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)s. 
# Note in this case it is necessary to call `Module.__init__` explicitly in `__post_init__`.

# In[ ]:


class MyLinear(torch.nn.Module, Configurable):
    d_in: int = 2
    d_out: int = 200

    def __post_init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=self.d_in, out_features=self.d_out)

    def forward(self, x):
        return self.linear.forward(x)


# In[ ]:


my_linear = MyLinear()
input = torch.zeros(2)
output = my_linear(input)
print("output shape:", output.shape)


# `my_linear` has all the usual features of a Module.
# E.g. it can be saved and loaded with `torch.save` and `torch.load`.
# It has parameters:

# In[ ]:


for name, value in my_linear.named_parameters():
    print(name, value.shape)


# ## 7. Example of implementing your own pluggable component 
# Let's say I am using a library with `Out` like in section **5** but I want to implement my own child of InnerBase. 
# All I need to do is register its definition, but I need to do this before expand_args_fields is explicitly or implicitly called on Out.

# In[ ]:


@registry.register
class UserImplementedInner(InnerBase):
    a: int = 200

    def say_something(self):
        print("hello from the user")


# At this point, we need to redefine the class Out. 
# Otherwise if it has already been expanded without UserImplementedInner, then the following would not work,
# because the implementations known to a class are fixed when it is expanded.
# 
# If you are running experiments from a script, the thing to remember here is that you must import your own modules, which register your own implementations,
# before you *use* the library classes.

# In[ ]:


class Out(Configurable):
    inner: InnerBase
    inner_class_type: str = "Inner1"
    x: int = 19

    def __post_init__(self):
        run_auto_creation(self)

    def talk(self):
        self.inner.say_something()


# In[ ]:


out2 = Out(inner_class_type="UserImplementedInner")
print(out2.inner)


# ## 8: Example of making a subcomponent pluggable
# 
# Let's look what needs to happen if we have a subcomponent which we make pluggable, to allow users to supply their own.

# In[ ]:


class SubComponent(Configurable):
    x: float = 0.25

    def apply(self, a: float) -> float:
        return a + self.x


class LargeComponent(Configurable):
    repeats: int = 4
    subcomponent: SubComponent

    def __post_init__(self):
        run_auto_creation(self)

    def apply(self, a: float) -> float:
        for _ in range(self.repeats):
            a = self.subcomponent.apply(a)
        return a


# In[ ]:


large_component = LargeComponent()
assert large_component.apply(3) == 4
print(OmegaConf.to_yaml(LargeComponent))


# Made generic:

# In[ ]:


class SubComponentBase(ReplaceableBase):
    def apply(self, a: float) -> float:
        raise NotImplementedError


@registry.register
class SubComponent(SubComponentBase):
    x: float = 0.25

    def apply(self, a: float) -> float:
        return a + self.x


class LargeComponent(Configurable):
    repeats: int = 4
    subcomponent: SubComponentBase
    subcomponent_class_type: str = "SubComponent"

    def __post_init__(self):
        run_auto_creation(self)

    def apply(self, a: float) -> float:
        for _ in range(self.repeats):
            a = self.subcomponent.apply(a)
        return a


# In[ ]:


large_component = LargeComponent()
assert large_component.apply(3) == 4
print(OmegaConf.to_yaml(LargeComponent))


# The following things had to change:
# * The base class SubComponentBase was defined.
# * SubComponent gained a `@registry.register` decoration and had its base class changed to the new one.
# * `subcomponent_class_type` was added as a member of the outer class.
# * In any saved configuration yaml files, the key `subcomponent_args` had to be changed to `subcomponent_SubComponent_args`.

# ## Appendix: gotchas ⚠️
# 
# * Omitting to define `__post_init__` or not calling `run_auto_creation` in it.
# * Omitting a type annotation on a field. For example, writing 
# ```
#     subcomponent_class_type = "SubComponent"
# ```
# instead of 
# ```
#     subcomponent_class_type: str = "SubComponent"
# ```
# 
# 
