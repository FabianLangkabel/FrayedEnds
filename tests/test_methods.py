import numpy
import pytest
import madpy

def test_methods_from_pno():
    orbitals = "pno"
    world = madpy.MadWorld(thresh=1.e-4)
    data = ("H 0.0 0.0 0.0\nH 0.0 0.0 5.0", -1.0)
    for method in ["spa", "hcb-upccgd", "upccgsd", "fci"]:
        print(method)
        geom, test_energy = data
        geom = geom.lower()
        energy, orbitals, rdm1, rdm2 = madpy.optimize_basis(world=world, many_body_method=method, geometry=geom, econv=1.e-3, orbitals=orbitals)
        assert numpy.isclose(energy, test_energy, atol=1.e-3)
    del world

def test_methods_from_minbas():
    orbitals = "sto-3g"
    kwargs = {}
    world = madpy.MadWorld(thresh=1.e-4)
    data = ("H 0.0 0.0 0.0\nH 0.0 0.0 5.0", -1.0)
    for method in ["upccgsd", "fci"]:
        print(method)
        if method != "fci": kwargs["optimizer_arguments"] = {"initial_values":"random"}
        geom, test_energy = data
        geom = geom.lower()
        energy, orbitals, rdm1, rdm2 = madpy.optimize_basis(world=world, many_body_method=method, geometry=geom, econv=1.e-3, orbitals=orbitals, **kwargs)
        assert numpy.isclose(energy, test_energy, atol=1.e-3)
    del world

def test_dmrg():
    geom = "H 0.0 0.0 0.0\nH 0.0 0.0 1.0\nH 0.0 0.0 2.0\nH 0.0 0.0 3.0"

    # initialize madness
    thresh = 1.e-4
    world = madpy.MadWorld(thresh=thresh)
    energies = []
    for method in ["dmrg", "fci"]:
        energy, orbitals, rdm1, rdm2 = madpy.optimize_basis(world=world, many_body_method=method, geometry=geom,
                                                            econv=thresh, orbitals="sto3g", maxiter=1)
        energies.append(energy)

    assert numpy.isclose(energies[0], energies[1])
    del world


