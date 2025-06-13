from .parameters import MadnessParameters
class MadPyBase:
    madness_parameters: MadnessParameters = None
    # the nanobind interface (madness runtime starts when initialized)
    impl = None

    def __init__(self, parameters=None, *args, **kwargs):
        if parameters is None:
            parameters = MadnessParameters()
            for k,v in kwargs.items():
                if k in parameters.__dict__:
                    parameters[k] = v
        self.madness_parameters=parameters