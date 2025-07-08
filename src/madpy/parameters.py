from dataclasses import dataclass
from ._madpy_impl import RedirectOutput
from functools import wraps

def redirect_output(filename="madness.out"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Redirect stdout to a file
            if filename is not None:
                red = RedirectOutput(filename)
            else:
                red = None
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
    initial_level: int=5 
    truncate_mode: int=1
    refine: bool=True
    n_threads: int=-1 # use all available threads by default

@dataclass
class MRAFunction:
    info: str = None # a string or datastructure with more information
    data: bin = None
    more: dict = None # a dictionary like structure that can carry more customized information

@dataclass
class Orbital:
    """
    Dataclass that holds binaries to MRA an MRA orbital and additional information
    """
    idx: int = None # an index associated to the orbital (most methods have one)
    occ: float = None # an occupation number associated to the orbital
    name: str = None # allows orbital to be named (e.g. "HF")
    info: str = None # a string or datastructure with more information
    more: dict = None # a dictionary like structure that can carry more customized information

    data: bin = None # the actual data

def unpack_madness_data(data, *args, **kwargs):
    """
    Takes an orbital, a list of orbitals or raw madness data
    :return: raw madness data or a list of raw madness data
    """

    if hasattr(data, "__len__"):
        return [unpack_madness_data(x) for x in data]
    elif hasattr(data, "data"):
        return data.data
    else:
        return data
