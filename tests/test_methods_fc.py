import numpy
import pytest
import madpy

# long test
@pytest.mark.parametrize("method", ["spa","fci"])
def test_methods(method, orbitals="pno"):
    data = ("Li 0.0 0.0 0.0\nH 0.0 0.0 1.5", -8.007)
    geom, test_energy = data
    geom = geom.lower()
    world = madpy.MadWorld(thresh=1.e-4)
    energy, orbitals, rdm1, rdm2 = madpy.optimize_basis(world=world, many_body_method=method, geometry=geom, econv=1.e-3, orbitals=orbitals)
    assert numpy.isclose(energy, test_energy, atol=1.e-3)
    del world
