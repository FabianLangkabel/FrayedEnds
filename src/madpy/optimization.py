import numpy as np

from ._madpy_impl import Optimization as OptInterface
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


class Optimization:

    _orbitals = None
    _h = None  # one-body tensor
    _g = None  # two-body tensor
    _c = 0.0  # constant term
    _Vnuc = None  # nuclear potential
    _nuclear_repulsion = None
    impl = None

    @property
    def orbitals(self, *args, **kwargs):
        return self.get_orbitals(*args, **kwargs)

    def __init__(self, madworld, Vnuc, nuc_repulsion, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.impl = OptInterface(madworld._impl)
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
        *args,
        **kwargs,
    ):
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
            self._orbitals = self.optimize_orbs(*args, **kwargs)
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

    def get_c(
        self, *args, **kwargs
    ):  # this is the sum of the energy of the frozen core electrons and the nuclear repulsion
        self._c = self.impl.GetC()
        return self._c
