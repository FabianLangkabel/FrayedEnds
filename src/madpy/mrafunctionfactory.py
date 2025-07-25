from ._madpy_impl import PyFuncFactory as FuncFactory


class MRAFunctionFactory:
    impl = None
    test = 0

    def __init__(self, madworld, pyfunc, *args, **kwargs):
        self.impl = FuncFactory(madworld._impl, pyfunc)

    def get_function(self):
        return self.impl.get_mra_function()
