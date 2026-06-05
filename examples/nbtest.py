import frayedends as fe
from pyscf import fci
import numpy as np
from time import time


def solve(world, geometry, n_act, n_core, name, tt = 1e-6, clo = 0.001, ceps = 1e-6, Blo = 0.001, Beps = 1e-6):
    start=time()
    molgeom = fe.MolecularGeometry(geometry, units="angstrom")
    n_act_electrons = molgeom.n_electrons - molgeom.n_core_electrons
    
    madpno = fe.MadPNO(world, geometry, units="angstrom", n_orbitals=n_act+n_core)
    Vnuc= madpno.get_nuclear_potential()
    nuc_repulsion = madpno.get_nuclear_repulsion()
    orbitals = madpno.get_orbitals()

    integrals = fe.Integrals3D(world, truncation_tol=tt, coulomb_eps=ceps, coulomb_lo=clo)
    print(integrals.get_numerical_parameters())
    orbitals = integrals.orthonormalize(orbitals=orbitals)

    core=orbitals[:n_core]
    active=orbitals[n_core:]

    c, h1, g2 = integrals.get_effective_hamiltonian(core, active, Vnuc, nuc_repulsion, g_ordering="chem")

    for iteration in range(5):
        # FCI calculation
        e, fcivec = fci.direct_spin1.kernel(
            h1, g2.elems, n_act, n_act_electrons
        )  # Computes the energy and the FCI vector
        rdm1, rdm2 = fci.direct_spin1.make_rdm12(
            fcivec, n_act, n_act_electrons
        )  # Computes the 1- and 2- body reduced density matrices
        rdm2 = np.swapaxes(rdm2, 1, 2)

        print("iteration {} energy {:+2.10f}".format(iteration, e + c))

        opti = fe.Optimization3D(world, Vnuc, nuc_repulsion, truncation_tol=tt, coulomb_eps=ceps, coulomb_lo=clo, BSH_lo=Blo, BSH_eps=Beps)
        print(opti.get_numerical_parameters())
        [core, active] = opti.get_orbitals(
            orbitals=[core, active],
            rdm1=rdm1,
            rdm2=rdm2,
            opt_thresh=0.001,
            occ_thresh=0.001,
            redirect_filename=f"madopt{iteration}.log",
        )

        c, h1, g2 = opti.get_effective_hamiltonian()
    end = time()
    print("Total time: ", end - start)

    for i in range(len(core)):
        core[i].save_to_file(f"{name}_core_orb{i}")
    for i in range(len(active)):
        active[i].save_to_file(f"{name}_active_orb{i}")
    








