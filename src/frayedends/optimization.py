import numpy as np

from ._frayedends_impl import Optimization2D as OptInterface2D
from ._frayedends_impl import Optimization3D as OptInterface3D
from .madworld import redirect_output


def transform_rdms(TransformationMatrix, rdm1, rdm2):
    new_rdm1 = np.dot(
        np.dot(TransformationMatrix.transpose(), rdm1), TransformationMatrix
    )
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

    _orbitals = None
    _h = None  # one-body tensor
    _g = None  # two-body tensor
    _c = 0.0  # constant term
    _Vnuc = None  # nuclear potential
    _nuclear_repulsion = None
    impl = None
    converged = None  # indicates if the last call converged
    opt_parameters = {
        "nocc": 2, #occupation number of orbitals (at this point only closed shell is supported, so nocc=2)
        "truncation_tol": 1e-6, #truncation tolerance for MRA representation of orbitals
        "coulomb_lo": 0.001, #lower cutoff for representation of Coulomb kernel
        "coulomb_eps": 1e-6,
        "BSH_lo": 0.001, #lower cutoff for representation of BSH kernel
        "BSH_eps": 1e-6,
    }

    @property
    def orbitals(self, *args, **kwargs):
        return self.get_orbitals(*args, **kwargs)

    def __init__(self, madworld, Vnuc, nuc_repulsion, *args, **kwargs):
        self.impl = OptInterface3D(madworld.impl)
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

    @redirect_output("madopt.log")
    def optimize_orbs(
        self,
        orbitals,
        rdm1,
        rdm2,
        opt_thresh=1.0e-4,
        occ_thresh=1.0e-5,
        maxiter=3,
        *args,
        **kwargs,
    ):
        rdm1_list = rdm1.reshape(-1).tolist()
        rdm2_list = rdm2.reshape(-1).tolist()
        self.impl.give_potential_and_repulsion(self._Vnuc, self._nuclear_repulsion)
        self.impl.give_initial_orbitals(orbitals)
        self.impl.give_rdm_and_rotate_orbitals(rdm1_list, rdm2_list)
        self.impl.calculate_all_integrals()
        self.impl.calculate_core_energy()
        self.impl.calculate_energies()

        converged = self.impl.optimize_orbitals(opt_thresh, occ_thresh, maxiter)
        self.impl.rotate_orbitals_back()

        self._orbitals = self.impl.get_orbitals()
        return self._orbitals, converged

    def get_orbitals(self, *args, **kwargs):
        if self._orbitals is None:
            self._orbitals, self.converged = self.optimize_orbs(*args, **kwargs)
            assert self._orbitals is not None
        return self._orbitals

    def get_integrals(self, *args, **kwargs):
        if self._orbitals is None:
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
        return {"nocc": self.impl.nocc, "truncation_tol": self.impl.truncation_tol, "coulomb_lo": self.impl.coulomb_lo, "coulomb_eps": self.impl.coulomb_eps, "BSH_lo": self.impl.BSH_lo, "BSH_eps": self.impl.BSH_eps}

class Optimization2D:

    _orbitals = None
    _h = None  # one-body tensor
    _g = None  # two-body tensor
    _c = 0.0  # constant term
    _Vnuc = None  # nuclear potential
    _nuclear_repulsion = None
    impl = None
    converged = None  # indicates if the last call converged
    opt_parameters = {
        "nocc": 2,
        "truncation_tol": 1e-6,
        "coulomb_lo": 0.001,
        "coulomb_eps": 1e-6,
        "BSH_lo": 0.001,
        "BSH_eps": 1e-6,
    }

    @property
    def orbitals(self, *args, **kwargs):
        return self.get_orbitals(*args, **kwargs)

    def __init__(self, madworld, Vnuc, nuc_repulsion, *args, **kwargs):
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

    @redirect_output("madopt.log")
    def optimize_orbs(
        self,
        orbitals,
        rdm1,
        rdm2,
        opt_thresh=1.0e-4,
        occ_thresh=1.0e-5,
        maxiter=3,
        *args,
        **kwargs,
    ):
        rdm1_list = rdm1.reshape(-1).tolist()
        rdm2_list = rdm2.reshape(-1).tolist()
        self.impl.give_potential_and_repulsion(self._Vnuc, self._nuclear_repulsion)
        self.impl.give_initial_orbitals(orbitals)
        self.impl.give_rdm_and_rotate_orbitals(rdm1_list, rdm2_list)
        self.impl.calculate_all_integrals()
        self.impl.calculate_core_energy()
        self.impl.calculate_energies()

        converged = self.impl.optimize_orbitals(opt_thresh, occ_thresh, maxiter)
        self.impl.rotate_orbitals_back()

        self._orbitals = self.impl.get_orbitals()
        return self._orbitals, converged

    def get_orbitals(self, *args, **kwargs):
        if self._orbitals is None:
            self._orbitals, self.converged = self.optimize_orbs(*args, **kwargs)
            assert self._orbitals is not None
        return self._orbitals

    def get_integrals(self, *args, **kwargs):
        if self._orbitals is None:
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
        return {"nocc": self.impl.nocc, "truncation_tol": self.impl.truncation_tol, "coulomb_lo": self.impl.coulomb_lo, "coulomb_eps": self.impl.coulomb_eps, "BSH_lo": self.impl.BSH_lo, "BSH_eps": self.impl.BSH_eps}
