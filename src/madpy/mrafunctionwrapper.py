from ._madpy_impl import (real_function_2d, real_function_3d, save_2D_function,
                          save_3D_function)
from .madworld import MadWorld2D, MadWorld3D

# simple wrapper classes for real_function_2d and real_function_3d with some very basic functionality
# could actually be one class in theory but for consistency I feel like this is better


class MRAFunction3D:
    impl = None  # real_function_3d
    type = "unspecified"  # if the function is an orbital the usual options are "frozen_occ", "active" and "frozen_virt"
    info = ""

    def __init__(self, impl: real_function_3d, type="unspecified", info=""):
        self.impl = impl
        self.type = type
        self.info = info

    # save a function to a .00000 file
    def save_to_file(self, filename):
        save_3D_function(self.impl, filename)

    # load a function from a .00000 file
    @staticmethod
    def load_from_file(world: MadWorld3D, filename, type="unspecified", info=""):
        return MRAFunction3D(world.mraf_from_file(filename), type=type, info=info)


class MRAFunction2D:
    impl = None  # real_function_2d
    type = "unspecified"  # if the function is an orbital the usual options are "frozen_occ", "active" and "frozen_virt"
    info = ""

    def __init__(self, impl: real_function_2d, type="unspecified", info=""):
        self.impl = impl
        self.type = type
        self.info = info

    # save a function to a .00000 file
    def save_to_file(self, filename):
        save_2D_function(self.impl, filename)

    # load a function from a .00000 file
    @staticmethod
    def load_from_file(world: MadWorld2D, filename, type="unspecified", info=""):
        return MRAFunction2D(world.mraf_from_file(filename), type=type, info=info)
