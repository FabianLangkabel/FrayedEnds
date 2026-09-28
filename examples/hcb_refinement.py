import numpy as np
import tequila as tq
from pyscf import fci

import frayedends as fe


def solve(world, geometry, n_act, n_core, name, method):
    print(f"\nSolving {name} with geometry:\n{geometry}\nActive orbitals: {n_act}, Core orbitals: {n_core}")

    molgeom = fe.MolecularGeometry(geometry, units="angstrom")
    n_act_electrons = molgeom.n_electrons - molgeom.n_core_electrons

    madpno = fe.MadPNO(world, geometry, units="angstrom", n_orbitals=n_act + n_core)
    orbitals = madpno.get_orbitals()
    edges = madpno.get_spa_edges()
    Vnuc = madpno.get_nuclear_potential()
    nuc_repulsion = madpno.get_nuclear_repulsion()
    del madpno

    integrals = fe.Integrals(world)
    orbitals = integrals.orthonormalize(orbitals=orbitals)
    core = orbitals[:n_core]
    active = orbitals[n_core:]

    current = 0.0
    for iteration in range(10):
        c, h1, g2 = integrals.compute_effective_hamiltonian(core, active, Vnuc, nuc_repulsion, g_ordering="chem")
        mol = tq.Molecule(
            geometry=geometry,
            one_body_integrals=h1,
            two_body_integrals=g2,
            nuclear_repulsion=c,
            transformation="reordered-jordan-wigner",
            frozen_core=False,
            n_electrons=n_act_electrons,
        )

        U = mol.make_spa_ansatz(edges=edges, hcb=True)
        opt = tq.chemistry.optimize_orbitals(molecule=mol, circuit=U, use_hcb=True, silent=True, initial_guess=None)
        active = integrals.transform(active, opt.mo_coeff)

        if method == "fci":
            c, h1, g2 = integrals.compute_effective_hamiltonian(core, active, Vnuc, nuc_repulsion, g_ordering="chem")
            e, fcivec = fci.direct_spin1.kernel(h1, g2, n_act, n_act_electrons)
            energy = e + c
        elif method == "spa":
            energy = opt.energy
        else:
            raise Exception(f"Supported methods are 'fci' and 'spa'. You selected '{method}'.")

        print("iteration {} energy {:+2.10f}".format(iteration, energy))
        if np.isclose(energy, current, atol=1.0e-5, rtol=0.0):
            break
        current = energy

        opti = fe.OrbitalRefinement(world, Vnuc, nuc_repulsion, use_hcb=True)
        if method == "fci":
            rdm1, rdm2 = opti.get_rdms_hcb(n_act, n_act_electrons, wfn=fcivec)
        else:
            rdm1, rdm2 = opti.get_rdms_hcb(n_act, n_act_electrons, U=U, molecule=opt.molecule)

        if n_core > 0:
            core, active = opti.get_orbitals(
                orbitals=[core, active],
                rdm1=rdm1,
                rdm2=rdm2,
                opt_thresh=0.001,
                occ_thresh=0.001,
            )
        else:
            active = opti.get_orbitals(
                orbitals=active,
                rdm1=rdm1,
                rdm2=rdm2,
                opt_thresh=0.001,
                occ_thresh=0.001,
            )
        del opti
    del integrals
    del world


test_list = [
    {"geometry": "H 0.0 0.0 0.0\nH 0.0 0.0 1.5", "n_act": 2, "n_core": 0, "name": "H2"},
    {"geometry": "H 0.0 0.0 -1.5\nBe 0.0 0.0 0.0\nH 0.0 0.0 1.5", "n_act": 4, "n_core": 1, "name": "BeH2"},
    # {
    #     "geometry": "Li 0.0 0.0 -1.5\nB 0.0 0.0 1.5",
    #     "n_act": 4,
    #     "n_core": 2,
    #     "name": "LiB",
    #     "tt": 1e-5,
    #     "clo": 0.003,
    #     "ceps": 2e-6,
    # },
    # {
    #     "geometry": "H 0.0 1.0 -1.5\nMg 1.0 0.0 0.0\nH 0.0 1.0 1.5",
    #     "n_act": 4,
    #     "n_core": 5,
    #     "name": "MgH2",
    #     "Blo": 0.005,
    #     "Beps": 3e-6,
    # },
]

world = fe.MadWorld(ndims=3)
for test in test_list:
    solve(world, method="spa", **test)
