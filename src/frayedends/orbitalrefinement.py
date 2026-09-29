import numpy as np
from tequila.quantumchemistry import NBodyTensor

from ._frayedends_impl import Optimization2D as OptInterface2D
from ._frayedends_impl import Optimization3D as OptInterface3D
from ._frayedends_impl import Optimization_open_shell_2D as OptInterface_open_shell_2D
from ._frayedends_impl import Optimization_open_shell_3D as OptInterface_open_shell_3D
from ._frayedends_impl import SavedFct2D, SavedFct3D
from .madworld import MadWorld, redirect_output


def transform_rdms(TransformationMatrix, rdm1, rdm2):
    new_rdm1 = np.dot(np.dot(TransformationMatrix.transpose(), rdm1), TransformationMatrix)
    n = rdm2.shape[0]

    temp1 = np.zeros(shape=(n, n, n, n))
    for i in range(n):
        for j in range(n):
            for k2 in range(n):
                for l in range(n):
                    k_value = 0
                    for k in range(n):
                        k_value += TransformationMatrix[k][k2] * rdm2[i][j][k][l]
                    temp1[i][j][k2][l] = k_value

    temp2 = np.zeros(shape=(n, n, n, n))
    for i2 in range(n):
        for j in range(n):
            for k2 in range(n):
                for l in range(n):
                    i_value = 0
                    for i in range(n):
                        i_value += TransformationMatrix[i][i2] * temp1[i][j][k2][l]
                    temp2[i2][j][k2][l] = i_value

    temp3 = np.zeros(shape=(n, n, n, n))
    for i2 in range(n):
        for j in range(n):
            for k2 in range(n):
                for l2 in range(n):
                    l_value = 0
                    for l in range(n):
                        l_value += TransformationMatrix[l][l2] * temp2[i2][j][k2][l]
                    temp3[i2][j][k2][l2] = l_value

    new_rdm2 = np.zeros(shape=(n, n, n, n))
    for i2 in range(n):
        for j2 in range(n):
            for k2 in range(n):
                for l2 in range(n):
                    j_value = 0
                    for j in range(n):
                        j_value += TransformationMatrix[j][j2] * temp3[i2][j][k2][l2]
                    new_rdm2[i2][j2][k2][l2] = j_value

    return new_rdm1, new_rdm2


class OrbitalRefinement:
    _fr_core_orbitals = None
    _active_orbitals = None
    _Vnuc = None  # nuclear potential
    _nuclear_repulsion = None
    dimensions = None
    impl = None
    converged = None  # indicates if the last call converged

    @property
    def orbitals(self, *args, **kwargs):
        return self.get_orbitals(*args, **kwargs)

    def __init__(
        self,
        madworld: MadWorld,
        Vnuc: SavedFct3D | SavedFct2D,
        nuc_repulsion: float,
        orthonormalization_method="symmetric",
        degeneracy_tol=1e-3,
        **kwargs,
    ):
        # setup the numerical environment for orbital refinement.
        if madworld.dimensions == 3:
            self.dimensions = 3
            self.impl = OptInterface3D(madworld.impl)
        if madworld.dimensions == 2:
            self.dimensions = 2
            self.impl = OptInterface2D(madworld.impl)
        self._Vnuc = Vnuc  # nuclear potential
        self._nuclear_repulsion = nuc_repulsion
        self.override_numerical_parameters(**kwargs)

        if orthonormalization_method not in ["symmetric", "cholesky", "cd", "mixed"]:
            raise ValueError(
                f"Invalid orthonormalization method '{orthonormalization_method}'. Must be 'symmetric', 'cd'/'cholesky', or 'mixed'."
            )
        elif orthonormalization_method != "symmetric":
            self.set_orthonormalization_method(
                orthonormalization_method, degeneracy_tol
            )  # after each refinement iteration the orbitals are orthonormalized using this method

    def override_numerical_parameters(
        self, truncation_tol=1e-6, coulomb_lo=0.001, coulomb_eps=1e-6, BSH_lo=0.001, BSH_eps=1e-6
    ):
        # truncation_tol: truncation tolerance for MRA representation of orbitals
        # coulomb_lo and coulomb_eps govern the accuracy of the representation of the Coulomb kernel
        # BSH_lo and BSH_eps govern the accuracy of the representation of the BSH kernel
        self.impl.override_numerical_parameters(truncation_tol, coulomb_lo, coulomb_eps, BSH_lo, BSH_eps)

    def get_numerical_parameters(self):
        params = self.impl.get_numerical_parameters()
        p_dict = {}
        for i in params:
            p_dict[i[0]] = i[1]
        return p_dict

    def set_orthonormalization_method(self, method="symmetric", degeneracy_tol=1e-3):
        """
        Set the orthonormalization method for orbital refinement.

        Args:
            method: Orthonormalization method - "symmetric", "cholesky", or "mixed"
                   - "symmetric": Standard symmetric orthonormalization (default)
                   - "cholesky": Cholesky decomposition orthonormalization
                   - "mixed": Use symmetric for degenerate orbitals, Cholesky for others
            degeneracy_tol: Tolerance for determining if two orbital occupations
                           are degenerate (only used for "mixed" method, default: 1e-3)
        """
        self.impl.set_orthonormalization_method(method, degeneracy_tol)

    @redirect_output("madopt.log")
    def refine_orbitals(
        self,
        orbitals: list,
        rdm1: np.ndarray,
        rdm2: np.ndarray,
        opt_thresh=1.0e-4,
        occ_thresh=1.0e-5,
        maxiter=3,
        refine_core=False,
        use_nonlinear_solver=False,
        *args,
        **kwargs,
    ):
        r"""
        this function performs the orbital refinement
        input:
         - one body reduced density matrix (rdm1) and two body reduced density matrix (rdm2) as 2 and 4 dimensional numpy arrays, respectively
           expects ordering of the form:
              rdm1[i,j] = \sum_\sigma \langle a_{i,\sigma}^\dagger a_{j,\sigma} \rangle
              rdm2[i,j,k,l] = \sum_{\sigma,\tau} \langle a_{i,\sigma}^\dagger a_{j,\tau}^\dagger a_{l,\tau} a_{k,\sigma} \rangle
         - orbitals is either a list of SavedFct3D objects (if all orbitals are active) or a list/tuple of [frozen_core_orbs, active_orbs], where frozen_core_orbs and active_orbs are lists of SavedFct3D objects.
         - opt_thresh is the threshold for convergence of the orbital refinement (based on the largest residual of the refined orbitals)
         - occ_thresh is the occupation threshold, if orbitals have occupation numbers < occ_thresh, they are skipped and not refined
         - maxiter is the maximum number of iterations for the orbital refinement
         - refine_core is a boolean flag indicating whether to refine the frozen core orbitals (default: False)
         - use_nonlinear_solver determines how orbitals are updated. Default (False): phi_{n+1} = phi_n - update; if True: updated using Krylov subspace accelerated inexact Newton method
        output:
         - list of frozen core orbitals, list of refined active orbitals and convergence flag
        """

        # Check if orbitals is a list of SavedFct objects or a list of [frozen_core, active] lists
        if isinstance(orbitals[0], SavedFct3D) or isinstance(orbitals[0], SavedFct2D):
            frozen_core_orbs = []
            active_orbs = orbitals
        else:
            frozen_core_orbs = orbitals[0]
            active_orbs = orbitals[1]

        if (len(active_orbs) != np.shape(rdm1)[0]) or (len(active_orbs) != np.shape(rdm2)[0]):
            raise ValueError(
                f"Number of active orbitals ({len(active_orbs)}) does not match the rdms dimensions ({np.shape(rdm1)} and {np.shape(rdm2)})."
            )

        self.impl.give_potential_and_repulsion(self._Vnuc, self._nuclear_repulsion)
        self.impl.give_initial_orbitals(frozen_core_orbs, active_orbs)
        self.impl.give_rdm_and_rotate_orbitals(rdm1, rdm2)
        self.converged = self.impl.optimize_orbitals(opt_thresh, occ_thresh, maxiter, refine_core, use_nonlinear_solver)
        self.impl.rotate_orbitals_back()

        self._fr_core_orbitals, self._active_orbitals = self.impl.get_orbitals()
        return self._fr_core_orbitals, self._active_orbitals, self.converged

    def get_orbitals(self, *args, **kwargs):
        if self._active_orbitals is None:
            self.refine_orbitals(*args, **kwargs)
            assert self._active_orbitals is not None
        if len(self._fr_core_orbitals) == 0:
            return self._active_orbitals
        else:
            return self._fr_core_orbitals, self._active_orbitals

    def get_effective_hamiltonian(self, g_ordering="phys", *args, **kwargs):
        # this returns (c, h, g) where c = nuclear repulsion + frozen core energy,
        # h is the effective one-body tensor (active space one body integrals plus active space - frozen core interaction) and
        # g the two-body tensor (active space Coulomb interaction). g is in phys ordering by default, can be reordered to chem or openfermion
        if self._active_orbitals is None:
            self.refine_orbitals(*args, **kwargs)
        H_eff = self.impl.get_effective_hamiltonian()
        g = NBodyTensor(elems=H_eff[2], ordering="phys")  # todo: remove tequila here and write custom function
        if g_ordering != "phys":
            return H_eff[0], H_eff[1], g.reorder(to=g_ordering).elems
        else:
            return H_eff

    def get_c(
        self, *args, **kwargs
    ):  # this is the sum of the energy of the frozen core electrons and the nuclear repulsion
        self._c = self.impl.get_c()
        return self._c


class OrbitalRefinement_open_shell:
    _orbitals = None
    _Vnuc = None  # nuclear potential
    _nuclear_repulsion = None
    impl = None
    converged = None  # indicates if the last call converged
    dimensions = None
    # @property
    # def orbitals(self, *args, **kwargs):
    #    return self.get_orbitals(*args, **kwargs)

    def __init__(
        self, madworld, Vnuc, nuc_repulsion, orthonormalization_method="symmetric", degeneracy_tol=1e-3, *args, **kwargs
    ):
        if madworld.dimensions == 3:
            self.dimensions = 3
            self.impl = OptInterface_open_shell_3D(madworld.impl)
        if madworld.dimensions == 2:
            self.dimensions = 2
            self.impl = OptInterface_open_shell_2D(madworld.impl)
        self._Vnuc = Vnuc
        self._nuclear_repulsion = nuc_repulsion
        self.override_numerical_parameters(**kwargs)

        if orthonormalization_method not in ["symmetric", "cd", "cholesky", "mixed"]:
            raise ValueError(
                f"Invalid orthonormalization method '{orthonormalization_method}'. Must be 'symmetric', 'cd'/'cholesky', or 'mixed'."
            )
        elif orthonormalization_method != "symmetric":
            self.set_orthonormalization_method(orthonormalization_method, degeneracy_tol)

    @redirect_output("madopt.log")
    def refine_orbitals(
        self,
        orbitals,
        rdm1,
        rdm2,
        opt_thresh=1.0e-4,
        occ_thresh=1.0e-5,
        maxiter=3,
        refine_core=False,
        use_nonlinear_solver=False,
        *args,
        **kwargs,
    ):
        r"""
        this function does orbital refinement for open shell system
        input:
         - orbitals is a list of [frozen_core_orbs_alpha, frozen_core_orbs_beta, active_orbs_alpha, active_orbs_beta], where each of the four elements is a list of SavedFct3D objects.
         - rdm1 is a list of two 2D numpy arrays [rdm1_alpha, rdm1_beta] with:
              rdm1_\sigma[i,j] = \langle a_{i,\sigma}^\dagger a_{j,\sigma} \rangle
         - rdm2 is a list of three 4D numpy arrays [rdm2_aa, rdm2_ab, rdm2_bb] with:
              rdm2_{\sigma\tau}[i,j,k,l] = \langle a_{i,\sigma}^\dagger a_{j,\tau}^\dagger a_{l,\tau} a_{k,\sigma} \rangle
         - opt_thresh is the threshold for convergence of the orbital refinement (based on the largest residual of the refined orbitals)
         - occ_thresh is the occupation threshold, if orbitals have occupation numbers < occ_thresh, they are skipped and not refined
         - maxiter is the maximum number of iterations for the orbital refinement
         - refine_core is a boolean flag indicating whether to refine the frozen core orbitals (default: False)
         - use_nonlinear_solver determines how orbitals are updated. Default (False): phi_{n+1} = phi_n - update; if True: updated using Krylov subspace accelerated inexact Newton method
        output:
         - list of frozen core orbitals as [frozen_core_orbs_alpha, frozen_core_orbs_beta], list of refined active orbitals as [active_orbs_alpha, active_orbs_beta] and convergence flag
        """

        self.impl.give_potential_and_repulsion(self._Vnuc, self._nuclear_repulsion)
        self.impl.give_initial_orbitals(orbitals[0], orbitals[1], orbitals[2], orbitals[3])
        self.impl.give_rdm_and_rotate_orbitals(rdm1, rdm2)
        converged = self.impl.optimize_orbitals(opt_thresh, occ_thresh, maxiter, refine_core, use_nonlinear_solver)
        self.impl.rotate_orbitals_back()
        self._orbitals = self.impl.get_orbitals()
        core_orbs = self._orbitals[:2]
        as_orbs = self._orbitals[2:]
        return core_orbs, as_orbs, converged

    def set_orthonormalization_method(self, method="symmetric", degeneracy_tol=1e-3):
        self.impl.set_orthonormalization_method(method, degeneracy_tol)

    def get_effective_hamiltonian(self, *args, **kwargs):
        H_eff = self.impl.get_effective_hamiltonian()
        return H_eff

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
