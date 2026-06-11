from time import time

import numpy as np
from pyscf import fci

import frayedends

true_start = time()

# Li2 geometry (bond distance ~2.67 Angstrom)
geom = "Li 0.0 0.0 0.0\nLi 0.0 0.0 2.67"  # geometry in Angstrom
molgeom = frayedends.MolecularGeometry(geom, units="angstrom")

# Setup world and parameters
world = frayedends.MadWorld(ndims=3, thresh=1e-6)
n_orbitals = 4  # 2 core + 2 active
n_core_orbitals = 2
n_act_orbitals = 2
n_act_electrons = molgeom.n_electrons - molgeom.n_core_electrons  # 2 electrons in active space

print(f"Active space electrons: {n_act_electrons}")

madpno = frayedends.MadPNO(world, geom, units="angstrom", n_orbitals=n_orbitals)
orbitals = madpno.get_orbitals()

integrals = frayedends.Integrals(world)
orbitals = integrals.orthonormalize(orbitals=orbitals)

# Get nuclear potential and repulsion energy
nuc_repulsion = molgeom.get_nuclear_repulsion()
Vnuc = molgeom.get_vnuc(world)

# Split into core and active orbitals
core = [orbitals[:n_core_orbitals], orbitals[:n_core_orbitals]]
active = [orbitals[n_core_orbitals:], orbitals[n_core_orbitals:]]


integrals = frayedends.Integrals_open_shell(world)


# SCF-like loop with orbital refinement and core refinement
for iteration in range(10):
    # Get initial effective Hamiltonian
    c, h1, g2 = integrals.compute_effective_hamiltonian(core[0], core[1], active[0], active[1], Vnuc, nuc_repulsion)
    g2[0] = g2[0].transpose(0, 2, 1, 3)
    g2[1] = g2[1].transpose(0, 2, 1, 3)
    g2[2] = g2[2].transpose(0, 2, 1, 3)

    # FCI calculation on active space
    e, fcivec = fci.direct_uhf.kernel(h1, g2, n_act_orbitals, (1, 1))
    rdm1, rdm2 = fci.direct_uhf.make_rdm12s(fcivec, n_act_orbitals, (1, 1))
    rdm2 = np.swapaxes(rdm2, 1, 2)
    rdm_2_phys_aa = rdm2[0].transpose(0, 2, 1, 3)
    rdm_2_phys_ab = rdm2[1].transpose(0, 2, 1, 3)
    rdm_2_phys_bb = rdm2[2].transpose(0, 2, 1, 3)

    print(f"Iteration {iteration} - FCI energy: {e + c:+2.10f} and core energy: {c:+2.10f}")

    # Orbital refinement with core orbital refinement enabled
    opti = frayedends.OrbitalRefinement_open_shell(world, Vnuc, nuc_repulsion)
    core, active, converged = opti.refine_orbitals(
        orbitals=[core[0], core[1], active[0], active[1]],
        rdm1=rdm1,
        rdm2=[rdm_2_phys_aa, rdm_2_phys_ab, rdm_2_phys_bb],
        opt_thresh=0.0001,
        occ_thresh=0.0001,
        maxiter=1,
        refine_core=True,  # Enable core orbital refinement
        redirect_filename=f"compli2_madopt{iteration}.log",
    )


orbitals = core[0] + active[0]
for i in range(len(orbitals)):
    world.cube_plot(f"oscompLi2_ref{i}_plot", orbitals[i], molgeom, zoom=5, datapoints=101)
    orbitals[i].save_to_file(f"oscompLi2_ref{i}")

true_end = time()
print(f"Total time: {true_end - true_start:.2f} seconds")

frayedends.cleanup(globals())
