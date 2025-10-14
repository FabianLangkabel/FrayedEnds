from ._madpy_impl import Eigensolver2D as EigenInterface2D
from ._madpy_impl import Eigensolver3D as EigenInterface3D
from .madworld import redirect_output
from .mrafunctionwrapper import MRAFunction2D, MRAFunction3D


class Eigensolver3D:
    _orbitals = None  # Placeholder for orbitals
    _potential = None
    impl = None

    def __init__(self, madworld, potential, *args, **kwargs):
        self.impl = EigenInterface3D(madworld.impl)
        self._potential = potential

    @redirect_output("mad_eigensolver.log")
    def solve(
        self, frozen_occ_dim, active_dim, frozen_virt_dim, n_states=10, max_iterations=5
    ):
        if frozen_occ_dim + active_dim + frozen_virt_dim > n_states:
            raise ValueError(
                "The sum of frozen occupied, active, and frozen virtual dimensions must be less than or equal to the number of states created by the eigensolver."
            )

        self.impl.solve(self._potential.impl, n_states, max_iterations)
        all_orbs_impl = self.impl.get_orbitals()  # all calculated states as real_function_3d
        result_orbs = []  # save all required states as MRAFunction3D and set types
        for i in range(frozen_occ_dim + active_dim + frozen_virt_dim):
            if i < frozen_occ_dim:
                orb_type = "frozen_occ"
            elif i < frozen_occ_dim + active_dim:
                orb_type = "active"
            else:
                orb_type = "frozen_virt"
            result_orbs.append(MRAFunction3D(all_orbs_impl[i], type=orb_type))
        self._orbitals = result_orbs
        return self._orbitals

    def get_orbitals(self, *args, **kwargs):
        if self._orbitals is not None:
            return self._orbitals
        else:
            return self.solve(*args, **kwargs)


class Eigensolver2D:
    _orbitals = None  # Placeholder for orbitals
    _potential = None
    impl = None

    def __init__(self, madworld, potential, *args, **kwargs):
        self.impl = EigenInterface2D(madworld.impl)
        self._potential = potential

    @redirect_output("mad_eigensolver.log")
    def solve(
        self, frozen_occ_dim, active_dim, frozen_virt_dim, n_states=10, max_iterations=5
    ):
        if frozen_occ_dim + active_dim + frozen_virt_dim > n_states:
            raise ValueError(
                "The sum of frozen occupied, active, and frozen virtual dimensions must be less than or equal to the number of states created by the eigensolver."
            )

        self.impl.solve(self._potential.impl, n_states, max_iterations)
        all_orbs_impl = self.impl.get_orbitals() # all calculated states as real_function_2d
        result_orbs = []  # save all required states as MRAFunction2D and set orb_types
        for i in range(frozen_occ_dim + active_dim + frozen_virt_dim):
            if i < frozen_occ_dim:
                orb_type = "frozen_occ"
            elif i < frozen_occ_dim + active_dim:
                orb_type = "active"
            else:
                orb_type = "frozen_virt"
            result_orbs.append(MRAFunction2D(all_orbs_impl[i], type=orb_type))
        self._orbitals = result_orbs
        return self._orbitals

    def get_orbitals(self, *args, **kwargs):
        if self._orbitals is not None:
            return self._orbitals
        else:
            return self.solve(*args, **kwargs)
