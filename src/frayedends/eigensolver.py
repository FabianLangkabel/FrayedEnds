from ._frayedends_impl import Eigensolver2D as EigenInterface2D
from ._frayedends_impl import Eigensolver3D as EigenInterface3D
from .madworld import redirect_output


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

        self.impl.solve(self._potential, n_states, max_iterations)
        self._orbitals = self.impl.get_orbitals(
            frozen_occ_dim, active_dim, frozen_virt_dim
        )
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

        self.impl.solve(self._potential, n_states, max_iterations)
        self._orbitals = self.impl.get_orbitals(
            frozen_occ_dim, active_dim, frozen_virt_dim
        )
        return self._orbitals

    def get_orbitals(self, *args, **kwargs):
        if self._orbitals is not None:
            return self._orbitals
        else:
            return self.solve(*args, **kwargs)
