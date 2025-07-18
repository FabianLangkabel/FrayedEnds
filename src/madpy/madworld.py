from ._madpy_impl import MadnessProcess
from ._madpy_impl import RedirectOutput
from functools import wraps

def redirect_output(filename="madness.out"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Redirect stdout to a file
            red = RedirectOutput(filename)
            result = func(*args, **kwargs)
            del red
            return result
        return wrapper
    return decorator

def get_function_info(orbitals):
    result = []
    for x in orbitals:
        info = {}
        for kv in x.info.strip().split(" "):
            kv = kv.split("=")
            info[kv[0]]=eval(kv[1])
        result.append({"type": x.type, **info})
    return result


class MadWorld:
    _impl = None

    madness_parameters = {
        "L": 50.0,
        "k": 7,
        "thresh": 1.e-5,
        "initial_level": 5,
        "truncate_mode": 1,
        "refine": True,
        "n_threads": -1
    }

    def __init__(self, **kwargs):

        self.madness_parameters = dict(self.madness_parameters)

        for k, v in kwargs.items():
            if k in self.madness_parameters:
                self.madness_parameters[k] = v
            else:
                raise ValueError(f"Unknown parameter: {k}")

        self._impl = MadnessProcess(self.madness_parameters["L"],
                                    self.madness_parameters["k"],
                                    self.madness_parameters["thresh"],
                                    self.madness_parameters["initial_level"],
                                    self.madness_parameters["truncate_mode"],
                                    self.madness_parameters["refine"],
                                    self.madness_parameters["n_threads"])

    def __getattr__(self, name):
        if name in self.madness_parameters:
            return getattr(self._impl, name)
        raise AttributeError(f"'MadWorld' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if hasattr(self, '_impl') and hasattr(self, 'madness_parameters') and name in self.madness_parameters:
            raise AttributeError(f"Cannot modify read-only attribute '{name}'")
        super().__setattr__(name, value)

    def get_params(self):
        return dict(self.madness_parameters)

    def line_plot(self, filename, mra_function, axis="z", datapoints=2001):
        self._impl.plot(filename, mra_function, axis, datapoints)

    def plane_plot(self, filename, mra_function, plane="yz", zoom=1.0, datapoints=81, origin=[0.0, 0.0, 0.0]):
        self._impl.plane_plot(filename, mra_function, plane, zoom, datapoints, origin)

