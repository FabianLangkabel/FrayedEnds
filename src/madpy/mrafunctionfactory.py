from ._madpy_impl import PyFuncFactory as FuncFactory

class MRAFunctionFactory:
    impl = None

    def __init__(self, madworld, pyfunc, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._world = madworld
        self._world.add_instance(self)
        self.impl = FuncFactory(self._world._impl, pyfunc)

    def GetFunction(self):
        return self.impl.GetMRAFunction()

