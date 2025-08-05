import numpy
import pytest
import tequila as tq
import madpy

# long test
@pytest.mark.parametrize("method", ["spa","fci"])
@pytest.mark.parametrize("orbitals", ["pno","sto3g"])
@pytest.mark.parametrize("data", [("H 0.0 0.0 0.0\nH 0.0 0.0 5.0",-1.0), ("Li 0.0 0.0 0.0\nH 0.0 0.0 1.5",-8.007)]) # values are for maxiter=1
def test_methods(data, method, orbitals):
    if method=="spa" and orbitals!="pno": return
    geom, test_energy = data
    geom = geom.lower()
    world = madpy.MadWorld(thresh=1.e-4)
    energy, orbitals, rdm1, rdm2 = madpy.optimize_basis(world=world, many_body_method=method, geometry=geom, econv=1.e-3, orbitals=orbitals)
    assert numpy.isclose(energy, test_energy, atol=1.e-3)
    del world
