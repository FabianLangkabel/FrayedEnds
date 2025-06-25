from ._madpy_impl import Optimization as OptInterface
from .baseclass import MadPyBase
from .parameters import redirect_output

class Optimization(MadPyBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.impl = OptInterface(self.madness_parameters.L, self.madness_parameters.k, self.madness_parameters.thresh)

    @redirect_output("madopt.log")
    def __call__(self, orbitals, rdm1, rdm2, opt_thresh=1.e-4, occ_thresh=1.e-5, *args, **kwargs):
        rdm1_list = rdm1.reshape(-1).tolist()
        rdm2_list = rdm2.reshape(-1).tolist()
        self.impl.CreateNuclearPotentialAndRepulsion("molecule")
        self.impl.GiveInitialOrbitals(orbitals)
        self.impl.GiveRDMsAndRotateOrbitals(rdm1_list, rdm2_list)
        self.impl.CalculateAllIntegrals()
        self.impl.CalculateCoreEnergy()
        self.impl.CalculateEnergies()

        self.impl.OptimizeOrbitals(opt_thresh, occ_thresh)
        self.impl.RotateOrbitalsBackAndUpdateIntegrals()

        return self.impl.GetOrbitals()
    
    def get_integrals(self):
        c = self.impl.GetC()
        h = self.impl.GetHTensor()
        g = self.impl.GetGTensor()
        return c, h, g
