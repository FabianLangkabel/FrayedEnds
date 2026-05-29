import numpy as np

from ._frayedends_impl import Optimization2D as OptInterface2D
from ._frayedends_impl import Optimization3D as OptInterface3D
from ._frayedends_impl import Optimization_open_shell_3D as OptInterface_open_shell_3D
from ._frayedends_impl import SavedFct2D, SavedFct3D
from .madworld import MadWorld2D, MadWorld3D, redirect_output


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


class Optimization3D:
    _fr_core_orbitals = None
    _active_orbitals = None
    _h = None  # one-body tensor
    _g = None  # two-body tensor
    _c = 0.0  # constant term
    _Vnuc = None  # nuclear potential
    _nuclear_repulsion = None
    impl = None
    converged = None  # indicates if the last call converged
    opt_parameters = {
        "truncation_tol": 1e-6,  # truncation tolerance for MRA representation of orbitals
        "coulomb_lo": 0.001,  # lower cutoff for representation of Coulomb kernel
        "coulomb_eps": 1e-6,
        "BSH_lo": 0.001,  # lower cutoff for representation of BSH kernel
        "BSH_eps": 1e-6,
    }

    @property
    def orbitals(self, *args, **kwargs):
        return self.get_orbitals(*args, **kwargs)

    def __init__(self, madworld: MadWorld3D, Vnuc: SavedFct3D, nuc_repulsion: float, **kwargs):
        # setup the numerical environment for orbital refinement.
        self.impl = OptInterface3D(madworld.impl)
        self._Vnuc = Vnuc
        self._nuclear_repulsion = nuc_repulsion
        self.override_numerical_parameters(**kwargs)

    def override_numerical_parameters(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.opt_parameters:
                self.opt_parameters[k] = v
            else:
                raise ValueError(f"Unknown parameter: {k}")
        self.impl.override_numerical_parameters(
            self.opt_parameters["truncation_tol"],
            self.opt_parameters["coulomb_lo"],
            self.opt_parameters["coulomb_eps"],
            self.opt_parameters["BSH_lo"],
            self.opt_parameters["BSH_eps"],
        )

    def set_orthonormalization_method(self, method="symmetric", degeneracy_tol=1e-3):
        """
        Set the orthonormalization method for orbital optimization.

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
    def optimize_orbs(
        self,
        orbitals: list,
        rdm1: np.ndarray,
        rdm2: np.ndarray,
        opt_thresh=1.0e-4,
        occ_thresh=1.0e-5,
        maxiter=3,
        refine_core=False,
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
         - opt_thresh is the threshold for convergence of the orbital refinement (based on the change of the energy)
         - occ_thresh is the occupation threshold, if orbitals have occupation numbers < occ_thresh, they are skipped and not refined
         - maxiter is the maximum number of iterations for the orbital refinement
        output:
         - list of frozen core orbitals, list of refined active orbitals and convergence flag
        """

        # Check if orbitals is a list of SavedFct3D or a list of [frozen_core, active] lists
        if isinstance(orbitals[0], SavedFct3D):
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
        self.converged = self.impl.optimize_orbitals(opt_thresh, occ_thresh, maxiter, refine_core)
        self.impl.rotate_orbitals_back()

        self._fr_core_orbitals, self._active_orbitals = self.impl.get_orbitals()
        return self._fr_core_orbitals, self._active_orbitals, self.converged

    @redirect_output("madopt.log")
    def optimize_orbs_old(
        self,
        orbitals: list,
        rdm1: np.ndarray,
        rdm2: np.ndarray,
        opt_thresh=1.0e-4,
        occ_thresh=1.0e-5,
        maxiter=3,
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
         - opt_thresh is the threshold for convergence of the orbital refinement (based on the change of the energy)
         - occ_thresh is the occupation threshold, if orbitals have occupation numbers < occ_thresh, they are skipped and not refined
         - maxiter is the maximum number of iterations for the orbital refinement
        output:
         - list of frozen core orbitals, list of refined active orbitals and convergence flag
        """

        # Check if orbitals is a list of SavedFct3D or a list of [frozen_core, active] lists
        if isinstance(orbitals[0], SavedFct3D):
            frozen_core_orbs = []
            active_orbs = orbitals
        else:
            frozen_core_orbs = orbitals[0]
            active_orbs = orbitals[1]

        if (len(active_orbs) != np.shape(rdm1)[0]) or (len(active_orbs) != np.shape(rdm2)[0]):
            raise ValueError(
                f"Number of active orbitals ({len(active_orbs)}) does not match the rdms dimensions ({np.shape(rdm1)} and {np.shape(rdm2)})."
            )

        rdm1_list = rdm1.reshape(-1).tolist()
        rdm2_list = rdm2.reshape(-1).tolist()
        self.impl.give_potential_and_repulsion(self._Vnuc, self._nuclear_repulsion)
        self.impl.give_initial_orbitals(frozen_core_orbs, active_orbs)
        self.impl.give_rdm_and_rotate_orbitals(rdm1_list, rdm2_list)
        self.impl.calculate_all_integrals_old()
        self.impl.calculate_core_energy_old()
        self.impl.calculate_energies_old()

        self.converged = self.impl.optimize_orbitals_old(opt_thresh, occ_thresh, maxiter)
        self.impl.rotate_orbitals_back()

        self._fr_core_orbitals, self._active_orbitals = self.impl.get_orbitals()
        return self._fr_core_orbitals, self._active_orbitals, self.converged

    def get_orbitals(self, *args, **kwargs):
        if self._active_orbitals is None:
            self.optimize_orbs(*args, **kwargs)
            assert self._active_orbitals is not None
        if len(self._fr_core_orbitals) == 0:
            return self._active_orbitals
        else:
            return self._fr_core_orbitals, self._active_orbitals

    def get_integrals(self, *args, **kwargs):
        if self._active_orbitals is None:
            self.optimize_orbs(*args, **kwargs)
        self.impl.calculate_all_integrals()
        self._c = self.impl.get_c()
        self._h = self.impl.get_h_tensor()
        self._g = self.impl.get_g_tensor()
        return self._c, self._h, self._g
    
    def get_effective_hamiltonian(self, *args, **kwargs):
        return self.impl.get_effective_hamiltonian()

    def get_c(
        self, *args, **kwargs
    ):  # this is the sum of the energy of the frozen core electrons and the nuclear repulsion
        self._c = self.impl.get_c()
        return self._c

    def get_numerical_parameters(self):
        params = self.impl.get_numerical_parameters()
        p_dict = {}
        for i in params:
            p_dict[i[0]] = i[1]
        return p_dict


class Optimization2D:
    _fr_core_orbitals = None
    _active_orbitals = None
    _h = None  # one-body tensor
    _g = None  # two-body tensor
    _c = 0.0  # constant term
    _Vnuc = None  # nuclear potential
    _nuclear_repulsion = None
    impl = None
    converged = None  # indicates if the last call converged
    opt_parameters = {
        "truncation_tol": 1e-6,  # truncation tolerance for MRA representation of orbitals
        "coulomb_lo": 0.001,  # lower cutoff for representation of Coulomb kernel
        "coulomb_eps": 1e-6,
        "BSH_lo": 0.001,  # lower cutoff for representation of BSH kernel
        "BSH_eps": 1e-6,
    }

    @property
    def orbitals(self, *args, **kwargs):
        return self.get_orbitals(*args, **kwargs)

    def __init__(self, madworld: MadWorld2D, Vnuc: SavedFct2D, nuc_repulsion: float, *args, **kwargs):
        # setup the numerical environment for orbital refinement.
        self.impl = OptInterface2D(madworld.impl)
        self._Vnuc = Vnuc
        self._nuclear_repulsion = nuc_repulsion
        for k, v in kwargs.items():
            if k in self.opt_parameters:
                self.opt_parameters[k] = v
            else:
                raise ValueError(f"Unknown parameter: {k}")

        self.impl.nocc = self.opt_parameters["nocc"]
        self.impl.truncation_tol = self.opt_parameters["truncation_tol"]
        self.impl.coulomb_lo = self.opt_parameters["coulomb_lo"]
        self.impl.coulomb_eps = self.opt_parameters["coulomb_eps"]
        self.impl.BSH_lo = self.opt_parameters["BSH_lo"]
        self.impl.BSH_eps = self.opt_parameters["BSH_eps"]

    def set_orthonormalization_method(self, method="symmetric", degeneracy_tol=1e-3):
        """
        Set the orthonormalization method for orbital optimization.

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
    def optimize_orbs(
        self,
        orbitals: list,
        rdm1: np.ndarray,
        rdm2: np.ndarray,
        opt_thresh=1.0e-4,
        occ_thresh=1.0e-5,
        maxiter=3,
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
         - orbitals is either a list of SavedFct2D objects (if all orbitals are active) or a list of [frozen_core_orbs, active_orbs], where frozen_core_orbs and active_orbs are lists of SavedFct2D objects.
         - opt_thresh is the threshold for convergence of the orbital refinement (based on the change of the energy)
         - occ_thresh is the occupation threshold, if orbitals have occupation numbers < occ_thresh, they are skipped and not refined
         - maxiter is the maximum number of iterations for the orbital refinement
        output:
         - list of frozen core orbitals, list of refined active orbitals and convergence flag
        """

        # Check if orbitals is a list of SavedFct2D or a list of [frozen_core, active] lists
        if isinstance(orbitals[0], SavedFct2D):
            frozen_core_orbs = []
            active_orbs = orbitals
        else:
            frozen_core_orbs = orbitals[0]
            active_orbs = orbitals[1]

        if (len(active_orbs) != np.shape(rdm1)[0]) or (len(active_orbs) != np.shape(rdm2)[0]):
            raise ValueError(
                f"Number of active orbitals ({len(active_orbs)}) does not match the rdms dimensions ({np.shape(rdm1)} and {np.shape(rdm2)})."
            )

        rdm1_list = rdm1.reshape(-1).tolist()
        rdm2_list = rdm2.reshape(-1).tolist()
        self.impl.give_potential_and_repulsion(self._Vnuc, self._nuclear_repulsion)
        self.impl.give_initial_orbitals(frozen_core_orbs, active_orbs)
        self.impl.give_rdm_and_rotate_orbitals(rdm1_list, rdm2_list)
        self.impl.calculate_all_integrals()
        self.impl.calculate_core_energy()
        self.impl.calculate_energies()

        self.converged = self.impl.optimize_orbitals(opt_thresh, occ_thresh, maxiter)
        self.impl.rotate_orbitals_back()

        self._fr_core_orbitals, self._active_orbitals = self.impl.get_orbitals()
        return self._fr_core_orbitals, self._active_orbitals, self.converged

    def get_orbitals(self, *args, **kwargs):
        if self._active_orbitals is None:
            self.optimize_orbs(*args, **kwargs)
            assert self._active_orbitals is not None
        if len(self._fr_core_orbitals) == 0:
            return self._active_orbitals
        else:
            return self._fr_core_orbitals, self._active_orbitals

    def get_integrals(self, *args, **kwargs):
        if self._active_orbitals is None:
            self.optimize_orbs(*args, **kwargs)
        self.impl.calculate_all_integrals()
        self._c = self.impl.get_c()
        self._h = self.impl.get_h_tensor()
        self._g = self.impl.get_g_tensor()
        return self._c, self._h, self._g

    def get_c(
        self, *args, **kwargs
    ):  # this is the sum of the energy of the frozen core electrons and the nuclear repulsion
        self._c = self.impl.get_c()
        return self._c

    def get_opt_parameters(self):
        return {
            "nocc": self.impl.nocc,
            "truncation_tol": self.impl.truncation_tol,
            "coulomb_lo": self.impl.coulomb_lo,
            "coulomb_eps": self.impl.coulomb_eps,
            "BSH_lo": self.impl.BSH_lo,
            "BSH_eps": self.impl.BSH_eps,
        }


class Optimization_open_shell_3D:
    _orbitals = None
    _Vnuc = None  # nuclear potential
    _nuclear_repulsion = None
    impl = None
    converged = None  # indicates if the last call converged

    # @property
    # def orbitals(self, *args, **kwargs):
    #    return self.get_orbitals(*args, **kwargs)

    def __init__(self, madworld, Vnuc, nuc_repulsion, *args, **kwargs):
        self.impl = OptInterface_open_shell_3D(madworld.impl)
        self._Vnuc = Vnuc
        self._nuclear_repulsion = nuc_repulsion

    @redirect_output("madopt.log")
    def optimize_orbs(
        self,
        orbitals,
        rdm1,
        rdm2,
        opt_thresh=1.0e-4,
        occ_thresh=1.0e-5,
        maxiter=3,
        orthonormalization_method="symmetric",
        refine_core=False,
        *args,
        **kwargs,
    ):
        self.impl.give_potential_and_repulsion(self._Vnuc, self._nuclear_repulsion)
        self.impl.give_initial_orbitals(orbitals[0], orbitals[1], orbitals[2], orbitals[3])
        self.impl.give_rdm_and_rotate_orbitals(rdm1, rdm2)
        converged = self.impl.optimize_orbitals(opt_thresh, occ_thresh, maxiter, orthonormalization_method, refine_core)
        self.impl.rotate_orbitals_back()
        self._orbitals = self.impl.get_orbitals()
        core_orbs = self._orbitals[:2]
        as_orbs = self._orbitals[2:]
        return core_orbs, as_orbs, converged

    def get_effective_hamiltonian(self, *args, **kwargs):
        H_eff = self.impl.get_effective_hamiltonian()
        return H_eff

    def override_numerical_parameters(
        self, truncation_tol=1e-6, coulomb_lo=0.001, coulomb_eps=1e-6, BSH_lo=0.001, BSH_eps=1e-6, *args, **kwargs
    ):
        self.impl.override_numerical_parameters(truncation_tol, coulomb_lo, coulomb_eps, BSH_lo, BSH_eps)
