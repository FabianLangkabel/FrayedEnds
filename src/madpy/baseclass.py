from .mrafunctionfactory import MadnessParameters

class MadPyBase:
    madness_parameters: MadnessParameters = None
    # the nanobind interface (madness runtime starts when initialized)
    impl = None

    def __init__(self, parameters=None, *args, **kwargs):
        if parameters is None:
            keyvals={}
            for k,v in kwargs.items():
                if k in MadnessParameters.__dict__.keys():
                    keyvals[k] = v
            parameters = MadnessParameters(**keyvals)
        self.madness_parameters=parameters

