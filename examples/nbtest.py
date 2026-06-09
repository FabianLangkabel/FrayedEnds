import frayedends as fe
from pyscf import fci
import numpy as np
from time import time


def solve(world, geometry, n_act, n_core, name, tt = 1e-6, clo = 0.001, ceps = 1e-6, Blo = 0.001, Beps = 1e-6):
    start=time()
    print(f"\nSolving {name} with geometry:\n{geometry}\nActive orbitals: {n_act}, Core orbitals: {n_core}")
    molgeom = fe.MolecularGeometry(geometry, units="angstrom")
    n_act_electrons = molgeom.n_electrons - molgeom.n_core_electrons
    
    madpno = fe.MadPNO(world, geometry, units="angstrom", n_orbitals=n_act+n_core)
    Vnuc= madpno.get_nuclear_potential()
    nuc_repulsion = madpno.get_nuclear_repulsion()
    orbitals = madpno.get_orbitals()

    integrals = fe.Integrals(world, truncation_tol=tt, coulomb_eps=ceps, coulomb_lo=clo)
    print(integrals.get_numerical_parameters())
    orbitals = integrals.orthonormalize(orbitals=orbitals)

    core=orbitals[:n_core]
    active=orbitals[n_core:]

    c, h1, g2 = integrals.get_effective_hamiltonian(core, active, Vnuc, nuc_repulsion, g_ordering="chem")
    pno_end = time()
    print(f"PNO generation time: {pno_end - start:.2f} seconds")

    for iteration in range(5):
        print("\n")
        fci_start = time()
        # FCI calculation
        e, fcivec = fci.direct_spin1.kernel(
            h1, g2, n_act, n_act_electrons
        )  # Computes the energy and the FCI vector
        rdm1, rdm2 = fci.direct_spin1.make_rdm12(
            fcivec, n_act, n_act_electrons
        )  # Computes the 1- and 2- body reduced density matrices
        rdm2 = np.swapaxes(rdm2, 1, 2)
        fci_end = time()
        print(f"FCI calculation time: {fci_end - fci_start:.2f} seconds")
        print("iteration {} energy {:+2.10f}".format(iteration, e + c))

        opti_start= time()
        opti = fe.Optimization(world, Vnuc, nuc_repulsion, truncation_tol=tt, coulomb_eps=ceps, coulomb_lo=clo, BSH_lo=Blo, BSH_eps=Beps)
        if iteration == 0:
            print(opti.get_numerical_parameters())
        core, active, converged = opti.optimize_orbs(
            orbitals=[core, active],
            rdm1=rdm1,
            rdm2=rdm2,
            opt_thresh=0.001,
            occ_thresh=0.001,
            redirect_filename=f"{name}_madopt{iteration}.log",
        )

        c, h1, g2 = opti.get_effective_hamiltonian(g_ordering = "chem")
        opti_end = time()
        print(f"Orbital optimization time: {opti_end - opti_start:.2f} seconds")
    end = time()
    print("Total time: ", end - start)

    for i in range(len(core)):
        core[i].save_to_file(f"{name}_core_orb{i}")
    for i in range(len(active)):
        active[i].save_to_file(f"{name}_active_orb{i}")
    


test_list = [
    {"geometry": "H 0.0 0.0 0.0\nH 0.0 0.0 1.5", "n_act": 2, "n_core": 0, "name": "H2"},
    {"geometry": "H 0.0 0.0 -1.5\nBe 0.0 0.0 0.0\nH 0.0 0.0 1.5", "n_act": 4, "n_core": 1, "name": "BeH2", "tt": 1e-7, "clo": 0.0001, "ceps": 1e-7, "Blo": 0.0001, "Beps": 1e-7},
    {"geometry": "Li 0.0 0.0 -1.5\nB 0.0 0.0 1.5", "n_act": 4, "n_core": 2, "name": "LiB", "tt": 1e-5, "clo": 0.003, "ceps": 2e-6},
    {"geometry": "H 0.0 1.0 -1.5\nMg 1.0 0.0 0.0\nH 0.0 1.0 1.5", "n_act": 4, "n_core": 5, "name": "MgH2","Blo": 0.005, "Beps": 3e-6}
]

world = fe.MadWorld(ndims=3)
for test in test_list:
    solve(world, **test)




