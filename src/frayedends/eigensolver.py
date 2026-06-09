from ._frayedends_impl import Eigensolver2D as EigenInterface2D
from ._frayedends_impl import Eigensolver3D as EigenInterface3D
from .madworld import redirect_output
from .madworld import MadWorld


class Eigensolver:
    _orbitals = None  # Placeholder for orbitals
    _potential = None
    impl = None
    dimensions = None

    def __init__(self, madworld, potential, *args, **kwargs):
        if madworld.dimensions == 3:
            self.dimensions = 3
            self.impl = EigenInterface3D(madworld.impl)
        elif madworld.dimensions == 2:
            self.dimensions = 2
            self.impl = EigenInterface2D(madworld.impl)
        self._potential = potential

    @redirect_output("mad_eigensolver.log")
    def solve(self, n_orbitals, n_guess_orbs=10, max_iterations=5):
        if n_orbitals > n_guess_orbs:
            n_guess_orbs = n_orbitals + 5

        self.impl.solve(self._potential, n_guess_orbs, max_iterations)
        self._orbitals = self.impl.get_orbitals(n_orbitals)
        return self._orbitals

    def get_orbitals(self, *args, **kwargs):
        if self._orbitals is not None:
            return self._orbitals
        else:
            return self.solve(*args, **kwargs)
