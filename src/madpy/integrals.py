from tequila.quantumchemistry import NBodyTensor

from ._madpy_impl import Integrals3D as IntegralsInterface3D
from ._madpy_impl import Integrals2D as IntegralsInterface2D
from ._madpy_impl import Integrals_open_shell_3D as IntegralsInterface_open_shell_3D


class Integrals3D:

    impl = None

    def __init__(self, madworld, *args, **kwargs):
        self.impl = IntegralsInterface3D(madworld._impl)

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
        return self.impl.compute_overlap_integrals(orbitals, other)

    def orthonormalize(self, orbitals, method="symmetric", rr_thresh=0.0, *args, **kwargs):
        return self.normalize(self.impl.orthonormalize(orbitals, method, rr_thresh, *args, **kwargs))

    def project_out(self, kernel, target, *args, **kwargs):
        return self.impl.project_out(kernel, target)

    def project_on(self, kernel, target, *args, **kwargs):
        return self.impl.project_on(kernel, target)

    def normalize(self, orbitals, *args, **kwargs):
        return self.impl.normalize(orbitals, *args, **kwargs)

    def transform(self, orbitals, matrix, *args, **kwargs):
        return self.impl.transform(orbitals, matrix)

    def compute_nuclear_derivative(
        self,
        molecule,
    ):
        pass

class Integrals2D:

    impl = None

    def __init__(self, madworld, *args, **kwargs):
        self.impl = IntegralsInterface2D(madworld._impl)

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
        return self.impl.compute_overlap_integrals(orbitals, other)

    def orthonormalize(self, orbitals, method="symmetric", rr_thresh=0.0, *args, **kwargs):
        return self.normalize(self.impl.orthonormalize(orbitals, method, rr_thresh, *args, **kwargs))

    def project_out(self, kernel, target, *args, **kwargs):
        return self.impl.project_out(kernel, target)

    def project_on(self, kernel, target, *args, **kwargs):
        return self.impl.project_on(kernel, target)

    def normalize(self, orbitals, *args, **kwargs):
        return self.impl.normalize(orbitals, *args, **kwargs)

    def transform(self, orbitals, matrix, *args, **kwargs):
        return self.impl.transform(orbitals, matrix)

    def compute_nuclear_derivative(
        self,
        molecule,
    ):
        pass

class Integrals_open_shell_3D:

    impl = None

    def __init__(self, madworld, *args, **kwargs):
        self.impl = IntegralsInterface_open_shell_3D(madworld._impl)

    def compute_two_body_integrals(self, alpha_orbitals, beta_orbitals, *args, **kwargs):
        G = self.impl.compute_two_body_integrals(alpha_orbitals, beta_orbitals)
        return G[0], G[1], G[2]

    def compute_kinetic_integrals(self, alpha_orbitals, beta_orbitals, *args, **kwargs):
        T = self.impl.compute_kinetic_integrals(alpha_orbitals, beta_orbitals)
        return T[0], T[1]

    def compute_potential_integrals(self, alpha_orbitals, beta_orbitals, V, *args, **kwargs):
        Pot = self.impl.compute_potential_integrals(alpha_orbitals, beta_orbitals, V)
        return Pot[0], Pot[1]

    def compute_nuclear_derivative(
        self,
        molecule,
    ):
        pass