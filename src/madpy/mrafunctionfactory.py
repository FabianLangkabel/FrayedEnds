from dataclasses import dataclass
from ._madpy_impl import PyFuncFactory as FuncFactory
from .baseclass import MadPyBase

@dataclass
class MadnessParameters:
    k: int=7 # wavelet order
    L: float=50.0 # simulation box size is (L*2)^3
    thresh: float=1.e-5 # MRA threshold
    initial_level: int=5
    truncate_mode: int=1
    refine: bool=True
    n_threads: int=-1 # use all available threads by default

class MRAFunctionFactory(MadPyBase):
    impl = None

    def __init__(self, pyfunc, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.impl = FuncFactory(pyfunc, self.madness_parameters.L, self.madness_parameters.k, self.madness_parameters.thresh, self.madness_parameters.initial_level, self.madness_parameters.truncate_mode, self.madness_parameters.refine)

    def GetFunction(self):
        return self.impl.GetMRAFunction()

