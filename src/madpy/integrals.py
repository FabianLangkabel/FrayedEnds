from ._madpy_impl import Integrals as IntegralsInterface
from tequila.quantumchemistry import NBodyTensor

class Integrals:

    _world = None
    impl = None

    def __init__(self, madworld, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._world = madworld
        self.impl = IntegralsInterface(self._world._impl)

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