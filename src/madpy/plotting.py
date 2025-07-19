from ._madpy_impl import MadnessProcess
from .baseclass import MadPyBase

def plot_lines(functions, name=None):
    plt = Plotter()
    for i in range(len(functions)):
        if name is None:
            x = "function_"+functions[i].type + " " + functions[i].info
            plt.line_plot(f"{x}{i}.dat", functions[i])
        else:
            plt.line_plot(f"{name}{i}.dat", functions[i])
    del plt

class Plotter(MadPyBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.impl = MadnessProcess(self.madness_parameters.L, self.madness_parameters.k, self.madness_parameters.thresh, self.madness_parameters.initial_level, self.madness_parameters.truncate_mode, self.madness_parameters.refine, self.madness_parameters.n_threads)
    
    def line_plot(self, filename, mra_function, axis="z", datapoints=2001):
        if hasattr(mra_function, "data"):
            self.impl.plot(filename, mra_function.data, axis, datapoints)
        else:
            self.impl.plot(filename, mra_function, axis, datapoints)

    def plane_plot(self, filename, mra_function, plane="yz", zoom=1.0, datapoints=81, origin=[0.0, 0.0, 0.0]):
        if hasattr(mra_function, "data"):
            self.impl.plane_plot(filename, mra_function.data, plane, zoom, datapoints, origin)
        else:
            self.impl.plane_plot(filename, mra_function, plane, zoom, datapoints, origin)

    def cube_plot(self, filename, mra_function, molecule, zoom=1.0, datapoints=81, origin=[0.0, 0.0, 0.0]):
        if hasattr(mra_function, "data"):
            self.impl.cube_plot(filename, mra_function.data, molecule, zoom, datapoints, origin)
        else:
            self.impl.cube_plot(filename, mra_function, molecule, zoom, datapoints, origin)