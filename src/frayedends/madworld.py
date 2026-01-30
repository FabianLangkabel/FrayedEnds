import inspect
from functools import wraps

from ._frayedends_impl import (MadnessProcess2D, MadnessProcess3D,
                               RedirectOutput)


def cleanup(globals):
    for name, obj in list(globals.items()):
        if (
            not name.startswith("__")
            and not callable(obj)
            and not inspect.ismodule(obj)
            and not isinstance(obj, type)
            and hasattr(obj, "__class__")
        ):
            # Check if the object is not the World
            if not isinstance(obj, MadWorld3D) and not isinstance(obj, MadWorld2D):
                del globals[name]


def redirect_output(filename="madness.out"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Allow per-call override using kwarg `redirect_filename`
            target = kwargs.pop("redirect_filename", filename)
            # Redirect stdout to a file
            red = RedirectOutput(target)
            try:
                result = func(*args, **kwargs)
            finally:
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
            info[kv[0]] = eval(kv[1])
        result.append({"type": x.type, **info})
    return result


class MadWorld3D:
    impl = None

    madness_parameters = {
        "L": 50.0,  # half the box length, units: bohr
        "k": 7,
        "thresh": 1.0e-5,
        "initial_level": 3,
        "truncate_mode": 1,
        "refine": True,
        "n_threads": -1,
    }

    def __init__(self, **kwargs):

        self.madness_parameters = dict(self.madness_parameters)

        for k, v in kwargs.items():
            if k in self.madness_parameters:
                self.madness_parameters[k] = v
            else:
                raise ValueError(f"Unknown parameter: {k}")

        self.impl = MadnessProcess3D(
            self.madness_parameters["L"],
            self.madness_parameters["k"],
            self.madness_parameters["thresh"],
            self.madness_parameters["initial_level"],
            self.madness_parameters["truncate_mode"],
            self.madness_parameters["refine"],
            self.madness_parameters["n_threads"],
        )

    def get_params(self):
        return dict(self.madness_parameters)

    def get_function_defaults(self):
        res = self.impl.get_function_defaults()
        return {
            "cell_width": res[0],
            "k": res[1],
            "thresh": res[2],
            "initial_level": res[3],
            "truncate_mode": res[4],
            "refine": res[5],
            "n_threads": res[6],
        }

    def set_function_defaults(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.madness_parameters.keys():
                self.madness_parameters[k] = v

        self.impl.L = self.madness_parameters["L"]
        self.impl.k = self.madness_parameters["k"]
        self.impl.thresh = self.madness_parameters["thresh"]
        self.impl.initial_level = self.madness_parameters["initial_level"]
        self.impl.truncate_mode = self.madness_parameters["truncate_mode"]
        self.impl.refine = self.madness_parameters["refine"]
        if "n_threads" in kwargs.keys():
            self.change_nthreads(self.madness_parameters["n_threads"])

        self.impl.update_function_defaults()

    def change_nthreads(self, nthreads):
        self.impl.change_nthreads(nthreads)

    def line_plot(self, filename, mra_function, axis="z", datapoints=2001):
        if hasattr(mra_function, "data"):
            self.impl.plot(filename, mra_function.data, axis, datapoints)
        else:
            self.impl.plot(filename, mra_function, axis, datapoints)

    def plot_lines(self, functions, name=None):
        for i in range(len(functions)):
            if name is None:
                x = "function_" + functions[i].type + " " + functions[i].info
                self.line_plot(f"{x}{i}.dat", functions[i])
            else:
                self.line_plot(f"{name}{i}.dat", functions[i])

    def plane_plot(
        self,
        filename,
        mra_function,
        plane="yz",
        zoom=1.0,
        datapoints=81,
        origin=[0.0, 0.0, 0.0],
    ):
        if hasattr(mra_function, "data"):
            self.impl.plane_plot(
                filename, mra_function.data, plane, zoom, datapoints, origin
            )
        else:
            self.impl.plane_plot(
                filename, mra_function, plane, zoom, datapoints, origin
            )

    def cube_plot(
        self,
        filename,
        mra_function,
        molecule,
        zoom=1.0,
        datapoints=81,
        origin=[0.0, 0.0, 0.0],
    ):
        if hasattr(mra_function, "data"):
            self.impl.cube_plot(
                filename, mra_function.data, molecule.impl, zoom, datapoints, origin
            )
        else:
            self.impl.cube_plot(
                filename, mra_function, molecule.impl, zoom, datapoints, origin
            )


class MadWorld2D:
    impl = None

    madness_parameters = {
        "L": 50.0,  # half the box length, units: bohr
        "k": 7,
        "thresh": 1.0e-5,
        "initial_level": 3,
        "truncate_mode": 1,
        "refine": True,
        "n_threads": -1,
    }

    def __init__(self, **kwargs):

        self.madness_parameters = dict(self.madness_parameters)

        for k, v in kwargs.items():
            if k in self.madness_parameters:
                self.madness_parameters[k] = v
            else:
                raise ValueError(f"Unknown parameter: {k}")

        self.impl = MadnessProcess2D(
            self.madness_parameters["L"],
            self.madness_parameters["k"],
            self.madness_parameters["thresh"],
            self.madness_parameters["initial_level"],
            self.madness_parameters["truncate_mode"],
            self.madness_parameters["refine"],
            self.madness_parameters["n_threads"],
        )

    def get_params(self):
        return dict(self.madness_parameters)

    def get_function_defaults(self):
        res = self.impl.get_function_defaults()
        return {
            "cell_width": res[0],
            "k": res[1],
            "thresh": res[2],
            "initial_level": res[3],
            "truncate_mode": res[4],
            "refine": res[5],
            "n_threads": res[6],
        }

    def set_function_defaults(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.madness_parameters.keys():
                self.madness_parameters[k] = v

        self.impl.L = self.madness_parameters["L"]
        self.impl.k = self.madness_parameters["k"]
        self.impl.thresh = self.madness_parameters["thresh"]
        self.impl.initial_level = self.madness_parameters["initial_level"]
        self.impl.truncate_mode = self.madness_parameters["truncate_mode"]
        self.impl.refine = self.madness_parameters["refine"]
        if "n_threads" in kwargs.keys():
            self.change_nthreads(self.madness_parameters["n_threads"])

        self.impl.update_function_defaults()

    def change_nthreads(self, nthreads):
        self.impl.change_nthreads(nthreads)

    def line_plot(self, filename, mra_function, axis="y", datapoints=2001):
        if hasattr(mra_function, "data"):
            self.impl.plot(filename, mra_function.data, axis, datapoints)
        else:
            self.impl.plot(filename, mra_function, axis, datapoints)

    def plot_lines(self, functions, name=None):
        for i in range(len(functions)):
            if name is None:
                x = "function_" + functions[i].type + " " + functions[i].info
                self.line_plot(f"{x}{i}.dat", functions[i])
            else:
                self.line_plot(f"{name}{i}.dat", functions[i])

    def plane_plot(
        self,
        filename,
        mra_function,
        plane="xy",
        zoom=1.0,
        datapoints=81,
        origin=[0.0, 0.0, 0.0],
    ):
        if hasattr(mra_function, "data"):
            self.impl.plane_plot(
                filename, mra_function.data, plane, zoom, datapoints, origin
            )
        else:
            self.impl.plane_plot(
                filename, mra_function, plane, zoom, datapoints, origin
            )
