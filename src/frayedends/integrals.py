import numpy as np
from tequila.quantumchemistry import NBodyTensor
from ._frayedends_impl import Integrals2D as IntegralsInterface2D
from ._frayedends_impl import Integrals3D as IntegralsInterface3D
class Integrals3D:
    impl = None
    def __init__(self, madworld, *args, **kwargs):
        self.impl = IntegralsInterface3D(madworld.impl)

    # computes the g-tensor: the coulomb interaction between the provided orbitals
    def compute_two_body_integrals(
        self,
        orbitals,  # active space orbitals
        ordering="phys",  # ordering of the tensor, possible choices: "phys" (1212), "chem" (1122), "openfermion" (1221)
        truncation_tol=1e-6,
        coulomb_lo=0.001,
        coulomb_eps=1e-6,
        nocc=2,
    ):
        g_elems = self.impl.compute_two_body_integrals(orbitals, truncation_tol, coulomb_lo, coulomb_eps, nocc)
        g = NBodyTensor(elems=g_elems, ordering="phys")
        if ordering != "phys":
            return g.reorder(to=ordering)
        else:
            return g

    # computes coulomb interaction between frozen core orbitals and active space orbitals
    def compute_frozen_core_interaction(
        self,
        frozen_core_orbs,
        active_orbs,
        truncation_tol=1e-6,
        coulomb_lo=0.001,
        coulomb_eps=1e-6,
        nocc=2,
    ):
        return self.impl.compute_frozen_core_interaction(
            frozen_core_orbs, active_orbs, truncation_tol, coulomb_lo, coulomb_eps, nocc
        )
    def compute_kinetic_integrals(self, orbitals, *args, **kwargs):
        return self.impl.compute_kinetic_integrals(orbitals)
    def compute_potential_integrals(self, orbitals, V, *args, **kwargs):
        return self.impl.compute_potential_integrals(orbitals, V)
    def compute_overlap_integrals(self, orbitals, other=None, *args, **kwargs):
        if other is None:
            other = orbitals
        return self.impl.compute_overlap_integrals(orbitals, other)
    def orthonormalize(
                self, orbitals, method="symmetric", rr_thresh=0.0, rdm1=None, degeneracy_tol=1e-6, *args, **kwargs
            ):
                if method == "mixed":
                    # Get occupations either from rdm1 or from orbitals
                    if rdm1 is not None:
                        # Extract diagonal of rdm1 as occupations
                        rdm1_array = np.asarray(rdm1, dtype=np.float64)
                        if rdm1_array.ndim == 2:
                            occupations = np.diag(rdm1_array)
                        elif rdm1_array.ndim == 1:
                            occupations = rdm1_array.copy()
                        else:
                            raise ValueError("rdm1 must be 1D (occupations) or 2D (density matrix)")
                    else:
                        # Try to get occupations from orbitals
                        occupations = np.array([orb.occupation for orb in orbitals], dtype=np.float64)
                        if np.all(occupations == 0.0):
                            raise ValueError("For mixed orthonormalization, either provide rdm1 or set orbital.occupation values")

                    # CRITICAL: Ensure it's a contiguous C-style array (required for C++ binding)
                    occupations = np.ascontiguousarray(occupations, dtype=np.float64)

                    return self.normalize(
                        self.impl.orthonormalize(orbitals, method, rr_thresh, occupations, degeneracy_tol)
                    )
                else:
                    # For other methods, pass empty array
                    occupations_empty = np.array([], dtype=np.float64)
                    return self.normalize(
                        self.impl.orthonormalize(orbitals, method, rr_thresh, occupations_empty, degeneracy_tol)
                    )

    def project_out(self, kernel, target, *args, **kwargs):
        return self.impl.project_out(kernel, target)
    def project_on(self, kernel, target, *args, **kwargs):
        return self.impl.project_on(kernel, target)
    def normalize(self, orbitals, *args, **kwargs):
        return self.impl.normalize(orbitals, *args, **kwargs)
    def transform(self, orbitals, matrix, *args, **kwargs):
        return self.impl.transform(orbitals, matrix)

    def transform_to_natural_orbitals(self, orbitals, rdm1):
        values, vectors = np.linalg.eigh(rdm1)  # diagonalize the 1-RDM (the eigenvalues are ordered ascendingly)
        val = values[::-1]  # reverse the order of eigenvalues
        vec = vectors[:, ::-1]  # reverse the order of eigenvectors accordingly
        return self.transform(orbitals, vec), val  # transform the orbitals to the natural orbitals


class Integrals2D:
    impl = None
    def __init__(self, madworld, *args, **kwargs):
        self.impl = IntegralsInterface2D(madworld.impl)
    def compute_two_body_integrals(
        self,
        orbitals,
        ordering="phys",
        truncation_tol=1e-6,
        coulomb_lo=0.001,
        coulomb_eps=1e-6,
        nocc=2,
    ):
        g_elems = self.impl.compute_two_body_integrals(orbitals, truncation_tol, coulomb_lo, coulomb_eps, nocc)
        g = NBodyTensor(elems=g_elems, ordering="phys")
        if ordering != "phys":
            return g.reorder(to=ordering)
        else:
            return g
    def compute_frozen_core_interaction(
        self,
        frozen_core_orbs,
        active_orbs,
        truncation_tol=1e-6,
        coulomb_lo=0.001,
        coulomb_eps=1e-6,
        nocc=2,
    ):
        return self.impl.compute_frozen_core_interaction(
            frozen_core_orbs, active_orbs, truncation_tol, coulomb_lo, coulomb_eps, nocc
        )
    def compute_kinetic_integrals(self, orbitals, *args, **kwargs):
        return self.impl.compute_kinetic_integrals(orbitals)
    def compute_potential_integrals(self, orbitals, V, *args, **kwargs):
        return self.impl.compute_potential_integrals(orbitals, V)
    def compute_overlap_integrals(self, orbitals, other=None, *args, **kwargs):
        if other is None:
            other = orbitals
        return self.impl.compute_overlap_integrals(orbitals, other)
    def orthonormalize(
        self, orbitals, method="symmetric", rr_thresh=0.0, rdm1=None, degeneracy_tol=1e-6, *args, **kwargs
    ):
        if method == "mixed":
            # Get occupations either from rdm1 or from orbitals
            if rdm1 is not None:
                # Extract diagonal of rdm1 as occupations
                rdm1_array = np.asarray(rdm1, dtype=np.float64)
                if rdm1_array.ndim == 2:
                    occupations = np.diag(rdm1_array)
                elif rdm1_array.ndim == 1:
                    occupations = rdm1_array
                else:
                    raise ValueError("rdm1 must be 1D (occupations) or 2D (density matrix)")
            else:
                # Try to get occupations from orbitals
                occupations = np.array([orb.occupation for orb in orbitals], dtype=np.float64)
                if np.all(occupations == 0.0):
                    raise ValueError("For mixed orthonormalization, either provide rdm1 or set orbital.occupation values")

            return self.normalize(
                self.impl.orthonormalize(orbitals, method, rr_thresh, occupations, degeneracy_tol)
            )
        else:
            # For other methods, pass empty array
            occupations_empty = np.array([], dtype=np.float64)
            return self.normalize(
                self.impl.orthonormalize(orbitals, method, rr_thresh, occupations_empty, degeneracy_tol)
            )

    def project_out(self, kernel, target, *args, **kwargs):
        return self.impl.project_out(kernel, target)
    def project_on(self, kernel, target, *args, **kwargs):
        return self.impl.project_on(kernel, target)
    def normalize(self, orbitals, *args, **kwargs):
        return self.impl.normalize(orbitals, *args, **kwargs)
    def transform(self, orbitals, matrix, *args, **kwargs):
        return self.impl.transform(orbitals, matrix)

    def transform_to_natural_orbitals(self, orbitals, rdm1):
        values, vectors = np.linalg.eigh(rdm1)  # diagonalize the 1-RDM (the eigenvalues are ordered ascendingly)
        val = values[::-1]  # reverse the order of eigenvalues
        vec = vectors[:, ::-1]  # reverse the order of eigenvectors accordingly
        return self.transform(orbitals, vec), val  # transform the orbitals to the natural orbitals
