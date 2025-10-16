import numpy
import pytest
import tequila as tq

import frayedends


@pytest.mark.parametrize("method", frayedends.pyscf_interface.SUPPORTED_RDM_METHODS)
@pytest.mark.parametrize(
    "geom", ["h 0.0 0.0 0.0\nh 0.0 0.0 0.75", "Li 0.0 0.0 0.0\nH 0.0 0.0 1.5"]
)
def test_pyscf_methods(geom, method):
    geom = geom.lower()
    world = frayedends.MadWorld3D()
    minbas = frayedends.AtomicBasisProjector(world, geom)
    orbitals = minbas.orbitals
    print(len(orbitals))
    c = minbas.get_nuclear_repulsion()
    Vnuc = minbas.get_nuclear_potential()
    del minbas

    integrals = frayedends.Integrals3D(world)
    orbitals = integrals.orthonormalize(orbitals=orbitals)
    V = integrals.compute_potential_integrals(orbitals, V=Vnuc)
    T = integrals.compute_kinetic_integrals(orbitals)
    G = integrals.compute_two_body_integrals(orbitals, ordering="chem")
    del integrals

    mol = frayedends.PySCFInterface(
        geometry=geom,
        one_body_integrals=T + V,
        two_body_integrals=G,
        constant_term=c,
        frozen_core=False,
    )
    rdm1, rdm2, energy = mol.compute_rdms(method=method, return_energy=True)

    if not "slow" in method:
        mol = tq.Molecule(geometry=geom, basis_set="sto-3g", frozen_core=False)
        if "fci" in method:
            method = "fci"
        test_energy = mol.compute_energy(method=method)
        assert numpy.isclose(energy, test_energy)

    del world


# this test tests a lot of stuff
# good for consistency check
# not the best test for individual debugging
@pytest.mark.parametrize("geom", ["Li 0.0 0.0 0.0\nH 0.0 0.0 1.5"])
def test_pyscf_methods_with_frozen_core(geom, method="fci"):
    geom = geom.lower()
    world = frayedends.MadWorld3D(thresh=1.0e-7)
    minbas = frayedends.AtomicBasisProjector(world, geom)
    sto3g = minbas.orbitals
    hf_orbitals = minbas.solve_scf()
    core_orbitals = [hf_orbitals[0]]
    c = minbas.get_nuclear_repulsion()
    Vnuc = minbas.get_nuclear_potential()
    del minbas

    integrals = frayedends.Integrals3D(world)
    sto3g = integrals.orthonormalize(orbitals=sto3g)
    # the core orbital is currently at the CBS (so it will be better than sto-3g)
    # need to project back, so that we can compare to sto-3g
    core_orbitals = integrals.project_on(kernel=sto3g, target=core_orbitals)
    core_orbitals = integrals.normalize(core_orbitals)

    rest = integrals.project_out(kernel=core_orbitals, target=sto3g)
    rest = integrals.normalize(rest)
    print("before rr_cholesky: ", len(rest))
    rest = integrals.orthonormalize(rest, method="rr_cholesky", rr_thresh=1.0e-3)
    print("after rr_cholesky: ", len(rest))
    orbitals = core_orbitals + rest
    orbitals = integrals.orthonormalize(orbitals)
    S = integrals.compute_overlap_integrals(orbitals=orbitals)
    print(S)
    V = integrals.compute_potential_integrals(orbitals, V=Vnuc)
    T = integrals.compute_kinetic_integrals(orbitals)
    G = integrals.compute_two_body_integrals(orbitals, ordering="chem")
    del integrals

    mol = frayedends.PySCFInterface(
        geometry=geom,
        one_body_integrals=T + V,
        two_body_integrals=G,
        constant_term=c,
        frozen_core=True,
    )
    rdm1, rdm2, energy = mol.compute_rdms(method=method, return_energy=True)

    mol = tq.Molecule(geometry=geom, basis_set="sto-3g", frozen_core=True)
    test_energy = mol.compute_energy(method=method)
    # we are projecting the CBS core orbital to sto-3g, not necessarility the same as the sto-3g core
    assert numpy.isclose(energy, test_energy, atol=1.0e-3)

    del world
