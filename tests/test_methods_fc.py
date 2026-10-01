import numpy
import pytest

import frayedends
import tequila as tq
from pyscf import fci


# long test
def test_methods(orbitals="pno"):
    data = ("Li 0.0 0.0 0.0\nH 0.0 0.0 1.5", -8.001)
    geom, test_energy = data
    geom = geom.lower()
    world = frayedends.MadWorld(ndims=3, thresh=1.0e-4)
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

def test_hcb_refinement():
    geometry = "H 0.0 0.0 -1.5\nBe 0.0 0.0 0.0\nH 0.0 0.0 1.5"
    n_act, n_core = 4, 1
    expected_energies = {
        "spa": -15.7887,
        "fci": -15.7903,
    }

    molgeom = frayedends.MolecularGeometry(geometry, units="angstrom")
    n_act_electrons = molgeom.n_electrons - molgeom.n_core_electrons

    world = frayedends.MadWorld(ndims=3)
    madpno = frayedends.MadPNO(world, geometry, units="angstrom", n_orbitals=n_act + n_core)
    orbitals = madpno.get_orbitals()
    edges = madpno.get_spa_edges()
    Vnuc = madpno.get_nuclear_potential()
    nuc_repulsion = madpno.get_nuclear_repulsion()
    del madpno
    
    integrals = frayedends.Integrals(world)
    orbitals = integrals.orthonormalize(orbitals=orbitals)

    for method in ["spa", "fci"]:
        core = orbitals[:n_core]
        active = orbitals[n_core:]
        current = 0.0
        for iteration in range(5):
            c, h1, g2 = integrals.compute_effective_hamiltonian(core, active, Vnuc, nuc_repulsion, g_ordering="chem")
            mol = tq.Molecule(geometry=geometry, one_body_integrals=h1, two_body_integrals=g2, nuclear_repulsion=c, transformation="reordered-jordan-wigner", frozen_core=False, n_electrons=n_act_electrons)
            
            U = mol.make_spa_ansatz(edges=edges, hcb=True)
            opt = tq.chemistry.optimize_orbitals(molecule=mol, circuit=U, use_hcb=True, silent=True, initial_guess=None)
            active = integrals.transform(active, opt.mo_coeff)

            if method == "fci":
                c, h1, g2 = integrals.compute_effective_hamiltonian(core, active, Vnuc, nuc_repulsion, g_ordering="chem")
                e, fcivec = fci.direct_spin1.kernel(h1, g2, n_act, n_act_electrons)
                energy = e + c
            else:
                energy = opt.energy
                
            if numpy.isclose(energy, current, atol=1.0e-5, rtol=0.0):
                break
            current = energy

            opti = frayedends.OrbitalRefinement(world, Vnuc, nuc_repulsion, use_hcb=True)
            if method == "fci":
                rdm1, rdm2 = opti.get_rdms_hcb(n_act, n_act_electrons, wfn=fcivec)
            else:
                rdm1, rdm2 = opti.get_rdms_hcb(n_act, n_act_electrons, U=U, molecule=opt.molecule)

            core, active = opti.get_orbitals(
                orbitals=[core, active],
                rdm1=rdm1,
                rdm2=rdm2,
                opt_thresh=0.001,
                occ_thresh=0.001,
            )
            del opti
        assert numpy.isclose(energy, expected_energies[method], atol=1.0e-3)
    del integrals
    del world