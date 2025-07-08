from ._madpy_impl import Optimization as OptInterface
from .baseclass import MadPyBase
from .parameters import redirect_output, unpack_madness_data

class Optimization(MadPyBase):

    _orbitals = None
    _h = None # one-body tensor
    _g = None # two-body tensor
    _c = 0.0 # constant term
    _Vnuc = None  # nuclear potential
    _nuclear_repulsion = None 

    @property
    def orbitals(self, *args, **kwargs):
        return self.get_orbitals(*args, **kwargs)

    def __init__(self, Vnuc, nuc_repulsion, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.impl = OptInterface(self.madness_parameters.L, self.madness_parameters.k, self.madness_parameters.thresh, self.madness_parameters.initial_level, self.madness_parameters.truncate_mode, self.madness_parameters.refine, self.madness_parameters.n_threads)
        self._Vnuc = Vnuc
        self._nuclear_repulsion = nuc_repulsion

    @redirect_output("madopt.log")
    def optimize_orbs(self, orbitals, rdm1, rdm2, opt_thresh=1.e-4, occ_thresh=1.e-5, *args, **kwargs):
        orbitals = unpack_madness_data(orbitals)
        rdm1_list = rdm1.reshape(-1).tolist()
        rdm2_list = rdm2.reshape(-1).tolist()
        self.impl.GivePotentialAndRepulsion(self._Vnuc, self._nuclear_repulsion)
        self.impl.GiveInitialOrbitals(orbitals)
        self.impl.GiveRDMsAndRotateOrbitals(rdm1_list, rdm2_list)
        self.impl.CalculateAllIntegrals()
        self.impl.CalculateCoreEnergy()
        self.impl.CalculateEnergies()

        self.impl.OptimizeOrbitals(opt_thresh, occ_thresh)
        self.impl.RotateOrbitalsBack()

        self._orbitals = self.impl.GetOrbitals()
        return self._orbitals
    
    def get_orbitals(self, *args, **kwargs):
        if self._orbitals is None:
            self._orbitals=self.optimize_orbs(*args, **kwargs)
            assert self._orbitals is not None
        return self._orbitals

    def get_integrals(self, *args, **kwargs):
        if self._orbitals is None:
            self.optimize_orbs(*args, **kwargs)
        self.impl.CalculateAllIntegrals()
        self._c = self.impl.GetC()
        self._h = self.impl.GetHTensor()
        self._g = self.impl.GetGTensor()
        return self._c, self._h, self._g
    
    def get_c(self, *args, **kwargs): #this is the sum of the energy of the frozen core electrons and the nuclear repulsion
        self._c=self.impl.GetC()
        return self._c
