from .parameters import MadnessParameters
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

def get_function_info(orbitals):
    result = []
    for x in orbitals:
        info = {}
        for kv in x.info.strip().split(" "):
            kv = kv.split("=")
            info[kv[0]]=eval(kv[1])
        result.append({"type": x.type, **info})
    return result
