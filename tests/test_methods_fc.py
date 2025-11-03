import numpy
import pytest

import frayedends


# long test
def test_methods(orbitals="pno"):
    data = ("Li 0.0 0.0 0.0\nH 0.0 0.0 1.5", -8.001)
    geom, test_energy = data
    geom = geom.lower()
    world = frayedends.MadWorld3D(thresh=1.0e-4)
    for method in ["spa", "fci"]:
        energy, orbitals, rdm1, rdm2 = frayedends.optimize_basis_3D(
            world=world,
            many_body_method=method,
            geometry=geom,
            econv=1.0e-2,
            orbitals=orbitals,
        )
        assert numpy.isclose(energy, test_energy, atol=1.0e-3)
    del world
