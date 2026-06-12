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

print(f"Li2 calculation: {n_core_orbitals} core orbitals (refined), {n_act_orbitals} active orbitals")
print(f"Active space electrons: {n_act_electrons}")

madpno = frayedends.MadPNO(world, geom, units="angstrom", n_orbitals=n_orbitals)
orbitals = madpno.get_orbitals()

integrals = frayedends.Integrals(world)
orbitals = integrals.orthonormalize(orbitals=orbitals)

# Get nuclear potential and repulsion energy
nuc_repulsion = molgeom.get_nuclear_repulsion()
Vnuc = molgeom.get_vnuc(world)

# Split into core and active orbitals
core = orbitals[:n_core_orbitals]
active = orbitals[n_core_orbitals:]

# Get initial effective Hamiltonian
c, h1, g2 = integrals.compute_effective_hamiltonian(core, active, Vnuc, nuc_repulsion, g_ordering="chem")


print(f"Initial core energy: {c:+2.10f}")

# SCF-like loop with orbital refinement and core refinement
for iteration in range(10):
    # FCI calculation on active space
    e, fcivec = fci.direct_spin1.kernel(h1, g2, n_act_orbitals, n_act_electrons)
    rdm1, rdm2 = fci.direct_spin1.make_rdm12(fcivec, n_act_orbitals, n_act_electrons)
    rdm2 = np.swapaxes(rdm2, 1, 2)

    print(f"Iteration {iteration} - FCI energy: {e + c:+2.10f} and core energy: {c:+2.10f}")

    # Orbital refinement with core orbital refinement enabled
    opti = frayedends.OrbitalRefinement(world, Vnuc, nuc_repulsion)
    opti.set_orthonormalization_method("mixed", 0.00001)
    core, active, converged = opti.refine_orbitals(
        orbitals=[core, active],
        rdm1=rdm1,
        rdm2=rdm2,
        opt_thresh=0.0001,
        occ_thresh=0.0001,
        maxiter=1,
        refine_core=True,  # Enable core orbital refinement
        redirect_filename=f"li2_madopt{iteration}.log",
    )

    # Get updated effective Hamiltonian
    c, h1, g2 = opti.get_effective_hamiltonian(g_ordering="chem")


orbitals = core + active
for i in range(len(orbitals)):
    world.cube_plot(f"Li2_ref{i}_plot", orbitals[i], molgeom, zoom=5, datapoints=101)
    orbitals[i].save_to_file(f"Li2_ref{i}")

true_end = time()
print(f"Total time: {true_end - true_start:.2f} seconds")

frayedends.cleanup(globals())
