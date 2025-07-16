from ._madpy_impl import PyFuncFactory as FuncFactory
from .baseclass import MadPyBase

class MRAFunctionFactory(MadPyBase):
    
    def __init__(self, pyfunc, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.impl = FuncFactory(pyfunc, self.madness_parameters.L, self.madness_parameters.k, self.madness_parameters.thresh, self.madness_parameters.initial_level, self.madness_parameters.truncate_mode, self.madness_parameters.refine)

    def GetFunction(self):
        return self.impl.GetMRAFunction()

