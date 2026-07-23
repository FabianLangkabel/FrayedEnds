import numpy as np
import pytest

import frayedends


def test_methods_from_pno():
    orbitals = "pno"
    world = frayedends.MadWorld(ndims=3, thresh=1.0e-4)
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
        assert np.isclose(energy, test_energy, atol=1.0e-3)
    del world


def test_methods_from_minbas():
    orbitals = "sto-3g"
    kwargs = {}
    world = frayedends.MadWorld(ndims=3, thresh=1.0e-4)
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
        assert np.isclose(energy, test_energy, atol=1.0e-3)
    del world

def test_2D_methods_from_ES():
    def potential(x: float, y: float) -> float:  # Qdot potential
        r = np.array([x, y, 1e-10])
        return -2 / np.linalg.norm(r)
    world = frayedends.MadWorld(ndims=2, L=100, thresh=1e-4)
    factory = frayedends.MRAFunctionFactory(world, potential) 
    mra_pot = factory.get_function()
    energy, orbitals, rdm1, rdm2 = frayedends.optimize_basis_2D(
        world,
        Vnuc=mra_pot,
        n_electrons=2,
        n_orbitals=4,
        orbitals="eigen",
        many_body_method="fci",
        maxiter=4,
        econv=1.0e-8,
    )
    assert np.isclose(energy, -11.85, atol=1.0e-1)
    del factory
    del world

