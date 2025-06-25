from dataclasses import dataclass
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

@dataclass
class MadnessParameters:
    k: int=7 # wavelet order
    L: float=50.0 # simulation box size is (L*2)^3
    thresh: float=1.e-5 # MRA threshold
    #todo initial level, truncate_mode, refine, threads
