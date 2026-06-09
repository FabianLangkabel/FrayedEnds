from ._frayedends_impl import PyFuncFactory2D as FuncFactory2D
from ._frayedends_impl import PyFuncFactory3D as FuncFactory3D
from .madworld import MadWorld


class MRAFunctionFactory:
    impl = None
    test = 0

    def __init__(self, madworld: MadWorld, pyfunc):
        # pyfunc is a python function defined as [](x: float, y: float, z: float) -> float
        # (or ...(x: float, y: float) for 2 dimensions)
        if madworld.dimensions == 3:
            self.impl = FuncFactory3D(madworld.impl, pyfunc)
        elif madworld.dimensions == 2:
            self.impl = FuncFactory2D(madworld.impl, pyfunc)

    def get_function(self, info=None):
        tmp = self.impl.get_mra_function()
        if info is not None:
            tmp.info = info
        return tmp
