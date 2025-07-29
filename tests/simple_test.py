import numpy
import pytest
import tequila as tq
import madpy


@pytest.mark.parametrize("geom", ["he 0.0 0.0 0.0", "Be 0.0 0.0 0.0"])
def test_pno_execution(geom):
    world = madpy.MadWorld()

    madpno = madpy.MadPNO(world, geom, n_orbitals=2)
    orbitals = madpno.get_orbitals()

    nuc_repulsion = madpno.get_nuclear_repulsion()
    Vnuc = madpno.get_nuclear_potential()

    integrals = madpy.Integrals(world)
    orbitals = integrals.orthonormalize(orbitals=orbitals)
    V = integrals.compute_potential_integrals(orbitals, V=Vnuc)
    S = integrals.compute_overlap_integrals(orbitals)
    T = integrals.compute_kinetic_integrals(orbitals)
    G = integrals.compute_two_body_integrals(orbitals)

    del madpno
    del integrals
    del world

@pytest.mark.parametrize("data", [("he 0.0 0.0 0.0",-2.87799), ("be 0.0 0.0 0.0",-14.59067977), ("h 0.0 0.0 0.0\nh 0.0 0.0 10.0", -1.0)])
def test_spa(data):
    geom, test_energy = data
    geom = geom.lower()
    world = madpy.MadWorld()
    n = 2
    if "be" in geom:
        n = 3
    madpno = madpy.MadPNO(world, geom, n_orbitals=n)
    orbitals = madpno.get_orbitals()
    edges = madpno.get_spa_edges()
    c = madpno.get_nuclear_repulsion()
    Vnuc = madpno.get_nuclear_potential()
    del madpno

    energy = 0.0
    for iteration in range(3):
        integrals = madpy.Integrals(world)
        orbitals = integrals.orthonormalize(orbitals=orbitals)
        V = integrals.compute_potential_integrals(orbitals, V=Vnuc)
        T = integrals.compute_kinetic_integrals(orbitals)
        G = integrals.compute_two_body_integrals(orbitals)
        del integrals

        mol = tq.Molecule(
            geom, one_body_integrals=T + V, two_body_integrals=G, nuclear_repulsion=c
        )
        U = mol.make_ansatz(name="SPA", edges=edges)
        H = mol.make_hamiltonian()
        E = tq.ExpectationValue(H=H, U=U)
        result = tq.minimize(E, silent=True)
        energy = result.energy
        print(result.energy)
        rdm1, rdm2 = mol.compute_rdms(U, variables=result.variables)

        opti = madpy.Optimization(world, Vnuc, c)
        orbitals = opti.get_orbitals(
            orbitals=orbitals, rdm1=rdm1, rdm2=rdm2, opt_thresh=0.001, occ_thresh=0.001
        )
        del opti


    assert numpy.isclose(energy, test_energy, atol=1.e-4)

    del world

@pytest.mark.parametrize("data", [("he 0.0 0.0 0.0",-2.87799), ("be 0.0 0.0 0.0",-14.59067977), ("h 0.0 0.0 0.0\nh 0.0 0.0 10.0", -1.0)])
def test_fci(data):
    geom, test_energy = data
    geom = geom.lower()
    world = madpy.MadWorld()
    n = 2
    if "be" in geom:
        n = 3
    madpno = madpy.MadPNO(world, geom, n_orbitals=n)
    orbitals = madpno.get_orbitals()
    print(len(orbitals))
    c = madpno.get_nuclear_repulsion()
    Vnuc = madpno.get_nuclear_potential()
    del madpno

    energy = 0.0
    for iteration in range(3):
        integrals = madpy.Integrals(world)
        orbitals = integrals.orthonormalize(orbitals=orbitals)
        V = integrals.compute_potential_integrals(orbitals, V=Vnuc)
        T = integrals.compute_kinetic_integrals(orbitals)
        G = integrals.compute_two_body_integrals(orbitals, ordering="chem")
        del integrals

        print(V.shape)
        mol = madpy.PySCFInterface(
            geometry=geom, one_body_integrals=T + V, two_body_integrals=G, constant_term=c
        )
        rdm1, rdm2, energy = mol.compute_rdms(method="fci", return_energy=True)
        print("iteration {} energy {:+2.5f}".format(iteration, energy))

        opti = madpy.Optimization(world, Vnuc, c)
        orbitals = opti.get_orbitals(
            orbitals=orbitals, rdm1=rdm1, rdm2=rdm2, opt_thresh=0.001, occ_thresh=0.001
        )
        del opti


    assert numpy.isclose(energy, test_energy, atol=1.e-4)

    del world

if __name__ == "__main__":
    test_spa("he 0.0 0.0 0.0")
