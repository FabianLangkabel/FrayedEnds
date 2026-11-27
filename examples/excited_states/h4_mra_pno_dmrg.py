"""
H4 linear molecule State Average (SA) DMRG calculation with Orbital Refinement
Pair Natural Orbitals (PNOs) used as initial guesses
"""

import time

import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

import frayedends as fe

# Parameter Configuration
molecule_name = "h4"
n_elec = 4  # Number of electrons
number_roots = 3  # Number of states (groundstate, 1. excited state, 2. excited state)
iterations = 20  # Number of iterations
box_size = 50.0  # Size of the simulation box
wavelet_order = 7  # Order of wavelet basis functions
madness_thresh = 1.0e-4  # Threshold for numerical precision of function representation
econv = 1.0e-4  # Energy convergence threshold (state-averaged energy)

# Create .dat files for the results
output_file = f"{molecule_name}_pno_dmrg_oo.dat"

with open(output_file, "w") as f:
    f.write("iteration iteration_time_s " + " ".join(f"energy_{i}" for i in range(number_roots)) + " energy_SA\n")

total_start = time.perf_counter()

# Define a linear H4 molecule geometry with 1.0 Angstrom spacing between adjacent atoms
geom = "H 0.0 0.0 -1.5 \nH 0.0 0.0 -0.5 \nH 0.0 0.0 0.5 \nH 0.0 0.0 1.5 \n"

# Setting up the numerical environment for the MRA calculations
world = fe.MadWorld(ndims=3, L=box_size, k=wavelet_order, thresh=madness_thresh)

# Get 8 Pair Natural Orbitals (PNOs)
madpno = fe.MadPNO(world, geom, units="angstrom", n_orbitals=8, maxrank=8)
orbs = madpno.get_orbitals()

nuclear_repulsion_energy = madpno.get_nuclear_repulsion()  # Compute nuclear repulsion energy
Vnuc = madpno.get_nuclear_potential()  # Compute nuclear potential

integrals = fe.Integrals(world)
orbs = integrals.orthonormalize(orbitals=orbs)  # Orthonormalize orbitals

n_orbitals = len(orbs)

# for i in range(n_orbitals):
#     world.line_plot(f"initial_orb{i}.dat", orbs[i], axis="z", datapoints=2001)  # Plot PNOs

# Calculate initial integrals
G = integrals.compute_two_body_integrals(orbs, ordering="chem").elems  # g-tensor (electron-electron interaction)
T = integrals.compute_kinetic_integrals(orbs)  # Kinetic energy
V = integrals.compute_potential_integrals(orbs, Vnuc)  # Potential energy
h1 = T + V

# Performe SA DMRG calculation and extract rdms
driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SU2, n_threads=8)
driver.initialize_system(n_sites=n_orbitals, n_elec=n_elec, spin=0)

idx = driver.orbital_reordering(h1, G)
h1_new = h1[idx][:, idx]
g2_new = G[idx][:, idx][:, :, idx][:, :, :, idx]

mpo = driver.get_qc_mpo(h1e=h1_new, g2e=g2_new, ecore=nuclear_repulsion_energy, iprint=0)
ket = driver.get_random_mps(tag="KET", bond_dim=100, nroots=number_roots)
energies = driver.dmrg(mpo, ket, n_sweeps=10, bond_dims=[100], noises=[1e-5] * 4 + [0], thrds=[1e-10] * 8, iprint=0)

# Extract rdms
kets = [driver.split_mps(ket, ir, tag="KET-%d" % ir) for ir in range(ket.nroots)]
sa_1pdm = np.mean([driver.get_1pdm(k) for k in kets], axis=0)  # Compute the state average 1-body rdm
sa_2pdm = np.mean([driver.get_2pdm(k) for k in kets], axis=0).transpose(
    0, 3, 1, 2
)  # Compute the state average 2-body rdm

# State-averaged energy from SA-RDMs
e_avg = (
    np.einsum("ij,ij->", sa_1pdm, h1_new) + 0.5 * np.einsum("ijkl,ijkl->", sa_2pdm, g2_new) + nuclear_repulsion_energy
)

idx_back = np.argsort(idx)  # Get the inverse of the orbital reordering
sa_1pdm = sa_1pdm[idx_back][:, idx_back]
sa_2pdm = sa_2pdm[idx_back][:, idx_back][:, :, idx_back][:, :, :, idx_back]
sa_2pdm_phys = sa_2pdm.swapaxes(1, 2)  # Change to physics Notation

with open(output_file, "a") as f:
    f.write("0 0.00 " + " ".join(f"{x:.15f}" for x in energies) + f" {e_avg:.15f}\n")

print(f"{'Iter':<6}{'Energies':<{14 * number_roots}}{'E_SA':<16}dE_SA")
print(f"{0:<6}{''.join(f'{e:<14.8f}' for e in energies)}{e_avg:<16.8f}-")

current = e_avg
converged = False

for iter in range(iterations):
    iter_start = time.perf_counter()

    # Orbital Refinement
    opti = fe.OrbitalRefinement(world, Vnuc, nuclear_repulsion_energy)
    orbs = opti.get_orbitals(orbitals=orbs, rdm1=sa_1pdm, rdm2=sa_2pdm_phys, opt_thresh=0.001, occ_thresh=0.001)

    # for i in range(len(orbs)):
    #     world.line_plot(f"orb{i}.dat", orbs[i], axis="z", datapoints=2001)  # Plot the refined orbitals

    # DMRG calculation with refined orbitals
    G = integrals.compute_two_body_integrals(orbs, ordering="chem").elems  # g-tensor (electron-electron interaction)
    T = integrals.compute_kinetic_integrals(orbs)  # Kinetic energy
    V = integrals.compute_potential_integrals(orbs, Vnuc)  # Potential energy
    h1 = T + V

    driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SU2, n_threads=8)
    driver.initialize_system(n_sites=n_orbitals, n_elec=n_elec, spin=0)

    idx = driver.orbital_reordering(h1, G)
    h1_new = h1[idx][:, idx]
    g2_new = G[idx][:, idx][:, :, idx][:, :, :, idx]

    mpo = driver.get_qc_mpo(h1e=h1_new, g2e=g2_new, ecore=nuclear_repulsion_energy, iprint=0)
    ket = driver.get_random_mps(tag="KET", bond_dim=100, nroots=number_roots)
    energies = driver.dmrg(mpo, ket, n_sweeps=10, bond_dims=[100], noises=[1e-5] * 4 + [0], thrds=[1e-10] * 8, iprint=0)

    kets = [driver.split_mps(ket, ir, tag="KET-%d" % ir) for ir in range(ket.nroots)]
    sa_1pdm = np.mean([driver.get_1pdm(k) for k in kets], axis=0)  # Compute the state average 1-body rdm
    sa_2pdm = np.mean([driver.get_2pdm(k) for k in kets], axis=0).transpose(
        0, 3, 1, 2
    )  # Compute the state average 2-body rdm

    # State-averaged energy from SA-RDMs
    e_avg = (
        np.einsum("ij,ij->", sa_1pdm, h1_new)
        + 0.5 * np.einsum("ijkl,ijkl->", sa_2pdm, g2_new)
        + nuclear_repulsion_energy
    )

    idx_back = np.argsort(idx)  # Get the inverse of the orbital reordering
    sa_1pdm = sa_1pdm[idx_back][:, idx_back]
    sa_2pdm = sa_2pdm[idx_back][:, idx_back][:, :, idx_back][:, :, :, idx_back]
    sa_2pdm_phys = sa_2pdm.swapaxes(1, 2)  # Change to physics Notation

    print(f"{iter + 1:<6}{''.join(f'{e:<14.8f}' for e in energies)}{e_avg:<16.8f}{e_avg - current:+.1e}")

    iter_time = time.perf_counter() - iter_start
    with open(output_file, "a") as f:
        f.write(f"{iter + 1} {iter_time:.2f} " + " ".join(f"{x:.15f}" for x in energies) + f" {e_avg:.15f}\n")

    if abs(e_avg - current) < econv:
        converged = True
        print(f"Converged after {iter + 1} iterations (|dE_SA| < {econv:.0e})")
        break  # stop as soon as the state-averaged energy changes less than econv
    current = e_avg

if not converged:
    print(f"Not converged after {iterations} iterations")

with open(output_file, "a") as f:
    status = "converged" if converged else "not converged"
    f.write(f"# {status}, econv = {econv:.0e}, total time = {time.perf_counter() - total_start:.2f} s\n")

fe.cleanup(globals())
