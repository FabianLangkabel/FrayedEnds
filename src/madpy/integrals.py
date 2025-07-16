from ._madpy_impl import Integrals as IntegralsInterface
from .baseclass import MadPyBase

from tequila.quantumchemistry import NBodyTensor

class Integrals(MadPyBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.impl = IntegralsInterface(self.madness_parameters.L, self.madness_parameters.k, self.madness_parameters.thresh, self.madness_parameters.initial_level, self.madness_parameters.truncate_mode, self.madness_parameters.refine, self.madness_parameters.n_threads)
    def compute_two_body_integrals(self, orbitals, ordering="phys", *args, **kwargs):
        g_elems = self.impl.compute_two_body_integrals(orbitals)
        g = NBodyTensor(elems=g_elems, ordering="phys")
        if ordering != "phys":
            return g.reorder(to=ordering)
        else:
            return g

    def compute_kinetic_integrals(self, orbitals, *args, **kwargs):
        return self.impl.compute_kinetic_integrals(orbitals)

    def compute_potential_integrals(self, orbitals, Vnuc, *args, **kwargs):
        return self.impl.compute_potential_integrals(orbitals, Vnuc)

    def compute_overlap_integrals(self, orbitals, *args, **kwargs):
        return self.impl.compute_overlap_integrals(orbitals)

    def hello(self):
        self.impl.hello()