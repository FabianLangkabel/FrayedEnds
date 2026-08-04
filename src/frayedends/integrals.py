import numpy as np
from tequila.quantumchemistry import NBodyTensor

from ._frayedends_impl import Integrals2D as IntegralsInterface2D
from ._frayedends_impl import Integrals3D as IntegralsInterface3D
from ._frayedends_impl import Integrals_open_shell_2D as IntegralsInterface_open_shell_2D
from ._frayedends_impl import Integrals_open_shell_3D as IntegralsInterface_open_shell_3D
from ._frayedends_impl import SavedFct2D, SavedFct3D
from .madworld import MadWorld


class Integrals:
    impl = None
    dimensions = None

    def __init__(self, madworld, **kwargs):
        if madworld.dimensions == 3:
            self.dimensions = 3
            self.impl = IntegralsInterface3D(madworld.impl)
        elif madworld.dimensions == 2:
            self.dimensions = 2
            self.impl = IntegralsInterface2D(madworld.impl)
        self.override_numerical_parameters(**kwargs)

    def override_numerical_parameters(self, truncation_tol=1e-6, coulomb_lo=0.001, coulomb_eps=1e-6):
        self.impl.override_numerical_parameters(truncation_tol, coulomb_lo, coulomb_eps)

    def get_numerical_parameters(self):
        params = self.impl.get_numerical_parameters()
        p_dict = {}
        for i in params:
            p_dict[i[0]] = i[1]
        return p_dict

    # computes the g-tensor: the coulomb interaction between the provided orbitals
    def compute_two_body_integrals(
        self,
        orbitals,  # active space orbitals
        ordering="phys",  # ordering of the tensor, possible choices: "phys" (1212), "chem" (1122), "openfermion" (1221)
    ) -> np.ndarray:
        g_elems = self.impl.compute_two_body_integrals(orbitals)
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
    ) -> np.ndarray:
        if len(frozen_core_orbs) == 0:
            print("Warning: No frozen core orbitals provided for frozen core interaction.")
            return np.zeros((len(active_orbs), len(active_orbs)))
        else:
            return self.impl.compute_frozen_core_interaction(frozen_core_orbs, active_orbs)

    def compute_kinetic_integrals(self, orbitals) -> np.ndarray:
        return self.impl.compute_kinetic_integrals(orbitals)

    def compute_potential_integrals(self, orbitals, V) -> np.ndarray:
        return self.impl.compute_potential_integrals(orbitals, V)

    def compute_overlap_integrals(self, orbitals, other=None) -> np.ndarray:
        if other is None:
            other = orbitals
        return self.impl.compute_overlap_integrals(orbitals, other)

    def compute_effective_hamiltonian(
        self, core_orbitals, active_orbitals, V, energy_offset, g_ordering="phys"
    ) -> tuple[float, np.ndarray, np.ndarray]:
        H_eff = self.impl.compute_effective_hamiltonian(core_orbitals, active_orbitals, V, energy_offset)
        g = NBodyTensor(elems=H_eff[2], ordering="phys")  # todo: remove tequila here and write custom function
        if g_ordering != "phys":
            return H_eff[0], H_eff[1], g.reorder(to=g_ordering).elems
        else:
            return H_eff

    def orthonormalize(
        self, orbitals, method="symmetric", rr_thresh=0.0, rdm1=None, degeneracy_tol=1e-6, *args, **kwargs
    ):
        if method == "mixed":
            if rdm1 is not None:
                rdm1_array = np.asarray(rdm1, dtype=np.float64)
                if (
                    rdm1_array.ndim == 1
                ):  # if rdm1 is already a vector of occupation numbers, we can directly pass it to the orthonormalization routine
                    occupations = rdm1_array.copy()
                    occupations = np.ascontiguousarray(occupations, dtype=np.float64)
                    return self.impl.orthonormalize(orbitals, method, rr_thresh, occupations, degeneracy_tol)
                elif rdm1_array.ndim == 2:
                    # if rdm1 is a density matrix, we need to transform to natural orbitals
                    orbitals, occupations, transformation_M = self.transform_to_natural_orbitals(orbitals, rdm1)
                    occupations = np.ascontiguousarray(occupations, dtype=np.float64)
                    # then orthnormalize
                    orbitals = self.impl.orthonormalize(orbitals, method, rr_thresh, occupations, degeneracy_tol)
                    # and transform back to original basis
                    orbitals = self.transform(orbitals, transformation_M.T)
                    return orbitals

                else:
                    raise ValueError("rdm1 must be 1D (occupations) or 2D (density matrix)")
            else:
                raise ValueError("For method 'mixed', rdm1 (occupations or density matrix) must be provided")
        else:
            # For other methods, pass empty array
            occupations_empty = np.array([], dtype=np.float64)
            return self.impl.orthonormalize(orbitals, method, rr_thresh, occupations_empty, degeneracy_tol)

    def project_out(self, kernel, target, *args, **kwargs):
        return self.impl.project_out(kernel, target)

    def project_on(self, kernel, target, *args, **kwargs):
        return self.impl.project_on(kernel, target)

    def normalize(self, orbitals, *args, **kwargs):
        return self.impl.normalize(orbitals)

    def transform(self, orbitals, matrix, *args, **kwargs):
        return self.impl.transform(
            orbitals, matrix
        )  # transforms orbitals according to: new[i] = sum[j] old[j]*matrix[j,i]

    def transform_to_natural_orbitals(self, orbitals, rdm1):
        values, vectors = np.linalg.eigh(rdm1)  # diagonalize the 1-RDM (the eigenvalues are ordered ascendingly)
        val = values[::-1]  # reverse the order of eigenvalues
        vec = vectors[:, ::-1]  # reverse the order of eigenvectors accordingly
        return self.transform(orbitals, vec), val, vec  # transform the orbitals to the natural orbitals

    def compute_electron_density(self, core_orbitals: list[SavedFct2D] | list[SavedFct3D], active_orbitals: list[SavedFct2D] | list[SavedFct3D], rdm1: np.ndarray) -> SavedFct2D | SavedFct3D:
        if len(active_orbitals) != rdm1.shape[0] or len(active_orbitals) != rdm1.shape[1]:
            raise ValueError(
                f"Number of active orbitals ({len(active_orbitals)}) does not match the shape of the 1-RDM ({rdm1.shape})."
            )
        return self.impl.compute_electron_density(core_orbitals, active_orbitals, rdm1)

class Integrals_open_shell:
    impl = None
    dimensions = None

    def __init__(self, madworld, *args, **kwargs):
        if madworld.dimensions == 3:
            self.dimensions = 3
            self.impl = IntegralsInterface_open_shell_3D(madworld.impl)
        elif madworld.dimensions == 2:
            self.dimensions = 2
            self.impl = IntegralsInterface_open_shell_2D(madworld.impl)
        self.override_numerical_parameters(**kwargs)

    def override_numerical_parameters(
        self, truncation_tol=1e-6, coulomb_lo=0.001, coulomb_eps=1e-6, BSH_lo=0.001, BSH_eps=1e-6, *args, **kwargs
    ):
        self.impl.override_numerical_parameters(truncation_tol, coulomb_lo, coulomb_eps, BSH_lo, BSH_eps)

    def get_numerical_parameters(self):
        params = self.impl.get_numerical_parameters()
        p_dict = {}
        for i in params:
            p_dict[i[0]] = i[1]
        return p_dict

    def compute_two_body_integrals(self, alpha_orbitals, beta_orbitals, *args, **kwargs):
        G = self.impl.compute_two_body_integrals(alpha_orbitals, beta_orbitals)
        return G[0], G[1], G[2]

    def compute_kinetic_integrals(self, alpha_orbitals, beta_orbitals, *args, **kwargs):
        T = self.impl.compute_kinetic_integrals(alpha_orbitals, beta_orbitals)
        return T[0], T[1]

    def compute_potential_integrals(self, alpha_orbitals, beta_orbitals, V, *args, **kwargs):
        Pot = self.impl.compute_potential_integrals(alpha_orbitals, beta_orbitals, V)
        return Pot[0], Pot[1]

    def compute_effective_hamiltonian(
        self,
        core_alpha_orbitals,
        core_beta_orbitals,
        active_alpha_orbitals,
        active_beta_orbitals,
        V,
        energy_offset,
        *args,
        **kwargs,
    ):
        H_eff = self.impl.compute_effective_hamiltonian(
            core_alpha_orbitals, core_beta_orbitals, active_alpha_orbitals, active_beta_orbitals, V, energy_offset
        )
        return H_eff

    def compute_electron_density(
        self,
        core_alpha_orbitals: list[SavedFct2D] | list[SavedFct3D],
        core_beta_orbitals: list[SavedFct2D] | list[SavedFct3D],
        active_alpha_orbitals: list[SavedFct2D] | list[SavedFct3D],
        active_beta_orbitals: list[SavedFct2D] | list[SavedFct3D],
        rdm1: list[np.ndarray],
    ) -> list[SavedFct2D] | list[SavedFct3D]:
        r"""input:
                - orbitals as 4 individual lists of MRA functions
                - rdm1 = [rdm1_alpha, rdm1_beta] with:
                  rdm1_\sigma[i,j] = \langle a_{i,\sigma}^\dagger a_{j,\sigma} \rangle
            output:
                - [rho_\alpha, rho_beta, rho_alpha + rho_beta] as 3 MRA functions
              """
        spin_data = (
            ("alpha", active_alpha_orbitals, rdm1[0]),
            ("beta", active_beta_orbitals, rdm1[1]),
        )
        for spin, active_orbitals, rdm in spin_data:
            if (
                rdm.ndim != 2
                or len(active_orbitals) != rdm.shape[0]
                or len(active_orbitals) != rdm.shape[1]
            ):
                raise ValueError(
                    f"Number of active {spin} orbitals ({len(active_orbitals)}) "
                    f"does not match the shape of the {spin} 1-RDM ({rdm.shape})."
                )

        return self.impl.compute_electron_density(
            core_alpha_orbitals,
            core_beta_orbitals,
            active_alpha_orbitals,
            active_beta_orbitals,
            rdm1
        )
