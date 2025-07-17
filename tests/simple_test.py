import pytest
import madpy
import numpy

@pytest.mark.parametrize("geom", ["he 0.0 0.0 0.0", "Be 0.0 0.0 0.0"])
def test_pno_execution(geom):
    madpno = madpy.MadPNO(geom, n_orbitals=2)
    orbitals = madpno.get_orbitals()
    param = madpno.madness_parameters
    nuc_repulsion = madpno.get_nuclear_repulsion()
    Vnuc = madpno.get_nuclear_potential()
    del madpno
    integrals = madpy.Integrals(param)
    orbitals = integrals.orthonormalize(orbitals=orbitals)
    V = integrals.compute_potential_integrals(orbitals, V=Vnuc)
    S = integrals.compute_overlap_integrals(orbitals)
    T = integrals.compute_kinetic_integrals(orbitals)
    G = integrals.compute_two_body_integrals(orbitals)
    del integrals