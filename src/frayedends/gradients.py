import numpy as np

from ._frayedends_impl import Gradients3D


class Gradients:
    

    def __init__(
        self,
        madworld,
        truncation_tol=1e-6,
        coulomb_lo=1e-3,
        coulomb_eps=1e-6,
        BSH_lo=1e-3,
        BSH_eps=1e-6,
    ):
        self.impl = Gradients3D(madworld.impl)
        self.impl.override_numerical_parameters(truncation_tol, coulomb_lo, coulomb_eps, BSH_lo, BSH_eps)

    def solve_h2(
        self,
        optimized_orbitals,
        rdm1,
        rdm2,
        lagrange_multipliers,
        molecular_geometry,
        d_rdm1_dz,
        d_rdm2_dz,
        d2_rdm1_dz2,
        d2_rdm2_dz2,
        max_iterations=20
    ):
        
        return self.impl.solve_h2(
            optimized_orbitals,
            rdm1,
            rdm2,
            lagrange_multipliers,
            molecular_geometry.impl,
            d_rdm1_dz,
            d_rdm2_dz,
            d2_rdm1_dz2,
            d2_rdm2_dz2,
            max_iterations
        )
