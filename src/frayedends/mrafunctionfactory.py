from ._frayedends_impl import PyFuncFactory2D as FuncFactory2D
from ._frayedends_impl import PyFuncFactory3D as FuncFactory3D


class MRAFunctionFactory3D:
    impl = None
    test = 0

    def __init__(self, madworld, pyfunc):
        self.impl = FuncFactory3D(madworld.impl, pyfunc)

    def get_function(self, info=None):
        tmp = self.impl.get_mra_function()
        if info is not None:
            tmp.info = info
        return tmp


class MRAFunctionFactory2D:
    impl = None
    test = 0

    def __init__(self, madworld, pyfunc, *args, **kwargs):
        self.impl = FuncFactory2D(madworld.impl, pyfunc)

    def get_function(self, info=None):
        tmp = self.impl.get_mra_function()
        if info is not None:
            tmp.info = info
        return tmp
