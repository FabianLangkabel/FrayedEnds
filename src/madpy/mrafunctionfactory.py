from ._madpy_impl import PyFuncFactory2D as FuncFactory2D
from ._madpy_impl import PyFuncFactory3D as FuncFactory3D
from .mrafunctionwrapper import MRAFunction2D, MRAFunction3D


class MRAFunctionFactory3D:
    impl = None
    test = 0

    def __init__(self, madworld, pyfunc, *args, **kwargs):
        self.impl = FuncFactory3D(madworld.impl, pyfunc)

    def get_function(self, type=None, info=None):
        tmp = MRAFunction3D(self.impl.get_mra_function())
        if type is not None:
            tmp.type = type
        if info is not None:
            tmp.info = info
        return tmp


class MRAFunctionFactory2D:
    impl = None
    test = 0

    def __init__(self, madworld, pyfunc, *args, **kwargs):
        self.impl = FuncFactory2D(madworld.impl, pyfunc)

    def get_function(self, type=None, info=None):
        tmp = MRAFunction2D(self.impl.get_mra_function())
        if type is not None:
            tmp.type = type
        if info is not None:
            tmp.info = info
        return tmp
