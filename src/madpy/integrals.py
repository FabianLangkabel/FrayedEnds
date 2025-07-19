from ._madpy_impl import Integrals as IntegralsInterface
from tequila.quantumchemistry import NBodyTensor

class Integrals:

    _world = None
    impl = None

    def __init__(self, madworld, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._world = madworld
        self._world.add_instance(self)
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

    def compute_potential_integrals(self, orbitals, V, *args, **kwargs):
        return self.impl.compute_potential_integrals(orbitals, V)

    def compute_overlap_integrals(self, orbitals, other=None, *args, **kwargs):
        if other is None:
            other = orbitals
        return self.impl.compute_overlap_integrals(orbitals,other)

    def orthonormalize(self, orbitals, method="symmetric", *args, **kwargs):
        return self.impl.orthonormalize(orbitals, method, *args, **kwargs)

    def project_out(self, kernel, target, *args, **kwargs):
        return self.impl.project_out(kernel, target)

    def transform(self, orbitals, matrix, *args, **kwargs):
        return self.impl.transform(orbitals, matrix)

    def hello(self):
        self.impl.hello()
