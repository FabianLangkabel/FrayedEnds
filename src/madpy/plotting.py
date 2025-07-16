from ._madpy_impl import MadnessProcess
from .baseclass import MadPyBase

class Plotter(MadPyBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.impl = MadnessProcess(self.madness_parameters.L, self.madness_parameters.k, self.madness_parameters.thresh, self.madness_parameters.initial_level, self.madness_parameters.truncate_mode, self.madness_parameters.refine, self.madness_parameters.n_threads)
    
    def line_plot(self, filename, mra_function, axis="z", datapoints=2001):
        self.impl.plot(filename, mra_function, axis, datapoints)
    
    def plane_plot(self, filename, mra_function, plane="yz", zoom=1.0, datapoints=81, origin=[0.0, 0.0, 0.0]):
        self.impl.plane_plot(filename, mra_function, plane, zoom, datapoints, origin)