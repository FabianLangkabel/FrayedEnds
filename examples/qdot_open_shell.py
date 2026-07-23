from time import time

import numpy as np
from pyscf import fci

import frayedends as fe

true_start = time()
n_alpha = 2  # Number of spin up e
n_beta = 1  # Number of spin down e
n_orbitals = 4  # Number of orbitals (all active in this example)


def potential(x: float, y: float) -> float:  # Qdot potential
    r = np.array([x, y, 1e-4])
    return -5 / np.linalg.norm(r)


# Setup world and parameters
world = fe.MadWorld(
    ndims=2, L=100, thresh=1e-4
)  # This is required for any MADNESS calculation as it initializes the required environment

factory = fe.MRAFunctionFactory(
    world, potential
)  # This transform a python function into a MRA function which can be read by MADNESS
mra_pot = factory.get_function()  # Potential as MRA function

eigen = fe.Eigensolver(world, mra_pot)  # This sets up the eigensolver, which provides initial guess orbitals
orbitals = eigen.get_orbitals(n_orbitals=n_orbitals, n_guess_orbs=5)

integrals = fe.Integrals(world)
orbitals = integrals.orthonormalize(orbitals=orbitals)

nuc_repulsion = 0

orbitals_ab = [orbitals, orbitals]

integralsOS = fe.Integrals_open_shell(world)

# SCF-like loop with orbital refinement and core refinement
for iteration in range(10):
    # Get initial effective Hamiltonian
    c, h1, g2 = integralsOS.compute_effective_hamiltonian(
        [], [], orbitals_ab[0], orbitals_ab[1], mra_pot, nuc_repulsion
    )
    g2[0] = g2[0].transpose(0, 2, 1, 3)  # transform g tensors to chem ordering
    g2[1] = g2[1].transpose(0, 2, 1, 3)
    g2[2] = g2[2].transpose(0, 2, 1, 3)

    # FCI calculation on active space
    e, fcivec = fci.direct_uhf.kernel(h1, g2, n_orbitals, (n_alpha, n_beta))
    rdm1, rdm2 = fci.direct_uhf.make_rdm12s(fcivec, n_orbitals, (n_alpha, n_beta))
    rdm2 = np.swapaxes(rdm2, 1, 2)
    rdm_2_phys_aa = rdm2[0].transpose(0, 2, 1, 3)  # again reordering to fit our convention
    rdm_2_phys_ab = rdm2[1].transpose(0, 2, 1, 3)
    rdm_2_phys_bb = rdm2[2].transpose(0, 2, 1, 3)

    print(f"Iteration {iteration} - FCI energy: {e + c:+2.10f}")

    # Orbital refinement with core orbital refinement enabled
    opti = fe.OrbitalRefinement_open_shell(world, mra_pot, nuc_repulsion)
    core, orbitals_ab, converged = opti.refine_orbitals(
        orbitals=[[], [], orbitals_ab[0], orbitals_ab[1]],
        rdm1=rdm1,
        rdm2=[rdm_2_phys_aa, rdm_2_phys_ab, rdm_2_phys_bb],
        opt_thresh=0.0001,
        occ_thresh=0.0001,
        maxiter=1,
        redirect_filename=f"madopt{iteration}.log",
    )

for i in range(len(orbitals_ab[0])):
    world.line_plot(f"orb{i}.dat", orbitals_ab[0][i], axis="y")  # Plots the optimized orbitals

true_end = time()
print(f"Total time: {true_end - true_start:.2f} seconds")

fe.cleanup(globals())
