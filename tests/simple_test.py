import numpy
import pytest
import tequila as tq
import madpy

# all tests run on the same thread
# not ideal ... 
world = None

def test_startup():
    world = madpy.MadWorld()


@pytest.mark.parametrize("geom", ["he 0.0 0.0 0.0", "Be 0.0 0.0 0.0"])
def test_pno_execution(geom):
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

@pytest.mark.parametrize("data", [("he 0.0 0.0 0.0",-2.8776), ("be 0.0 0.0 0.0",-14.602), ("h 0.0 0.0 0.0\nh 0.0 0.0 10.0", -1.0)])
def test_spa(data):
    geom, test_energy = data
    geom = geom.lower()
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
    for iteration in range(1):
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


    assert numpy.isclose(energy, test_energy, atol=1.e-3)

@pytest.mark.parametrize("data", [("he 0.0 0.0 0.0",-2.8776), ("be 0.0 0.0 0.0",-14.602), ("h 0.0 0.0 0.0\nh 0.0 0.0 10.0", -1.0)])
def test_fci(data):
    geom, test_energy = data
    geom = geom.lower()
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
    for iteration in range(1):
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


    assert numpy.isclose(energy, test_energy, atol=1.e-3)

@pytest.mark.parametrize("method", madpy.pyscf_interface.SUPPORTED_RDM_METHODS)
@pytest.mark.parametrize("geom", ["h 0.0 0.0 0.0\nh 0.0 0.0 0.75", "Li 0.0 0.0 0.0\nH 0.0 0.0 1.5"])
def test_pyscf_methods(geom, method):
    geom = geom.lower()
    minbas = madpy.AtomicBasisProjector(world, geom)
    orbitals = minbas.orbitals
    print(len(orbitals))
    c = minbas.get_nuclear_repulsion()
    Vnuc = minbas.get_nuclear_potential()
    del minbas


    integrals = madpy.Integrals(world)
    orbitals = integrals.orthonormalize(orbitals=orbitals)
    V = integrals.compute_potential_integrals(orbitals, V=Vnuc)
    T = integrals.compute_kinetic_integrals(orbitals)
    G = integrals.compute_two_body_integrals(orbitals, ordering="chem")
    del integrals


    mol = madpy.PySCFInterface(
        geometry=geom, one_body_integrals=T + V, two_body_integrals=G, constant_term=c, frozen_core=False
    )
    rdm1, rdm2, energy = mol.compute_rdms(method=method, return_energy=True)

    mol = tq.Molecule(geometry=geom, basis_set="sto-3g", frozen_core=False)
    test_energy = mol.compute_energy(method=method)
    assert numpy.isclose(energy, test_energy)

# this test tests a lot of stuff
# good for consistency check
# not the best test for individual debugging
@pytest.mark.parametrize("geom", ["Li 0.0 0.0 0.0\nH 0.0 0.0 1.5"])
def test_pyscf_methods_with_frozen_core(geom, method="fci"):
    geom = geom.lower()
    minbas = madpy.AtomicBasisProjector(world, geom)
    sto3g = minbas.orbitals
    hf_orbitals = minbas.solve_scf()
    core_orbitals = [hf_orbitals[0]]
    c = minbas.get_nuclear_repulsion()
    Vnuc = minbas.get_nuclear_potential()
    del minbas

    integrals = madpy.Integrals(world)
    sto3g = integrals.orthonormalize(orbitals=sto3g)
    # the core orbital is currently at the CBS (so it will be better than sto-3g)
    # need to project back, so that we can compare to sto-3g
    core_orbitals = integrals.project_on(kernel=sto3g, target=core_orbitals)
    core_orbitals = integrals.normalize(core_orbitals)

    rest = integrals.project_out(kernel=core_orbitals, target=sto3g)
    rest = integrals.normalize(rest)
    print("before rr_cholesky: ", len(rest))
    rest = integrals.orthonormalize(rest, method="rr_cholesky", rr_thresh=1.e-3)
    print("after rr_cholesky: ", len(rest))
    orbitals = core_orbitals + rest
    orbitals = integrals.orthonormalize(orbitals)
    S = integrals.compute_overlap_integrals(orbitals=orbitals)
    print(S)
    V = integrals.compute_potential_integrals(orbitals, V=Vnuc)
    T = integrals.compute_kinetic_integrals(orbitals)
    G = integrals.compute_two_body_integrals(orbitals, ordering="chem")
    del integrals


    mol = madpy.PySCFInterface(
        geometry=geom, one_body_integrals=T + V, two_body_integrals=G, constant_term=c, frozen_core=True
    )
    rdm1, rdm2, energy = mol.compute_rdms(method=method, return_energy=True)

    mol = tq.Molecule(geometry=geom, basis_set="sto-3g", frozen_core=True)
    test_energy = mol.compute_energy(method=method)
    # we are projecting the CBS core orbital to sto-3g, not necessarility the same as the sto-3g core
    assert numpy.isclose(energy, test_energy, atol=1.e-3)

# long test
@pytest.mark.parametrize("method", ["spa","fci"])
@pytest.mark.parametrize("orbitals", ["pno","sto3g"])
@pytest.mark.parametrize("data", [("H 0.0 0.0 0.0\nH 0.0 0.0 5.0",-1.0), ("Li 0.0 0.0 0.0\nH 0.0 0.0 1.5",-8.007)]) # values are for maxiter=1
def test_methods(data, method, orbitals):
    if method=="spa" and orbitals!="pno": return
    geom, test_energy = data
    geom = geom.lower()
    energy, orbitals, rdm1, rdm2 = madpy.optimize_basis(world=world, many_body_method=method, geometry=geom, econv=1.e-3, orbitals=orbitals)
    assert numpy.isclose(energy, test_energy, atol=1.e-3)

def test_teardown():
    del world

if __name__ == "__main__":
    test_spa("he 0.0 0.0 0.0")
