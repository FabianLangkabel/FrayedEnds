import numpy
import pytest

import frayedends


def test_methods_from_pno():
    orbitals = "pno"
    world = frayedends.MadWorld3D(thresh=1.0e-4)
    data = ("H 0.0 0.0 0.0\nH 0.0 0.0 5.0", -1.0)
    for method in ["spa", "hcb-upccgd", "upccgsd", "fci"]:
        print(method)
        geom, test_energy = data
        geom = geom.lower()
        energy, orbitals, rdm1, rdm2 = frayedends.optimize_basis_3D(
            world=world,
            many_body_method=method,
            geometry=geom,
            econv=1.0e-3,
            orbitals=orbitals,
        )
        assert numpy.isclose(energy, test_energy, atol=1.0e-3)
    del world


def test_methods_from_minbas():
    orbitals = "sto-3g"
    kwargs = {}
    world = frayedends.MadWorld3D(thresh=1.0e-4)
    data = ("H 0.0 0.0 0.0\nH 0.0 0.0 5.0", -1.0)
    for method in ["upccgsd", "fci"]:
        print(method)
        if method != "fci":
            kwargs["optimizer_arguments"] = {"initial_values": "random"}
        geom, test_energy = data
        geom = geom.lower()
        energy, orbitals, rdm1, rdm2 = frayedends.optimize_basis_3D(
            world=world,
            many_body_method=method,
            geometry=geom,
            econv=1.0e-3,
            orbitals=orbitals,
            **kwargs,
        )
        assert numpy.isclose(energy, test_energy, atol=1.0e-3)
    del world
