"""
H4 linear molecule FCI calculation (without Orbital Refinement)
Pair Natural Orbitals (PNOs) used as initial guesses
"""

import time

import numpy as np
from pyscf import fci

import frayedends as fe

# Parameter Configuration
molecule_name = "h4"
n_elec = 4  # Number of electrons
number_roots = 3  # Number of states (groundstate, 1. excited state, 2. excited state)
box_size = 50.0  # Size of the simulation box
wavelet_order = 7  # Order of wavelet basis functions
madness_thresh = 1.0e-4  # Threshold for numerical precision of function representation

# Create .dat files for the results
output_file = f"{molecule_name}_pno_fci.dat"
 
with open(output_file, "w") as f:
    f.write("time_s " + " ".join(f"energy_{i}" for i in range(number_roots)) + " energy_SA\n")

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
#     world.line_plot(f"orb{i}.dat", orbs[i], axis="z", datapoints=2001)  # Plot PNOs

# Calculate initial integrals
G = integrals.compute_two_body_integrals(orbs, ordering="chem").elems  # g-tensor (electron-electron interaction)
T = integrals.compute_kinetic_integrals(orbs)  # Kinetic energy
V = integrals.compute_potential_integrals(orbs, Vnuc)  # Potential energy
h1 = T + V

# Performe FCI calculation and extract rdms
cisolver = fci.direct_spin1.FCI()
cisolver = fci.addons.fix_spin_(cisolver, ss=0)
e, fcivec = cisolver.kernel(h1, G, n_orbitals, n_elec, nroots=number_roots)
energies = e + nuclear_repulsion_energy

rdms = [cisolver.make_rdm12(c, n_orbitals, n_elec) for c in fcivec]
sa_1pdm = np.mean([r[0] for r in rdms], axis=0)  # Compute the state average 1-body rdm
sa_2pdm = np.mean([r[1] for r in rdms], axis=0)  # Compute the state average 2-body rdm

# State-averaged energy from SA-RDMs
e_avg = (np.einsum("ij,ij->", sa_1pdm, h1)
        + 0.5 * np.einsum("ijkl,ijkl->", sa_2pdm, G)
        + nuclear_repulsion_energy)

total_time = time.perf_counter() - total_start
 
print(f"{'Energies':<{14 * number_roots}}{'E_SA':<16}Time / s")
print(f"{''.join(f'{e:<14.8f}' for e in energies)}{e_avg:<16.8f}{total_time:.1f}")
 
with open(output_file, "a") as f:
    f.write(f"{total_time:.2f} " + " ".join(f"{x:.15f}" for x in energies) + f" {e_avg:.15f}\n")
 
fe.cleanup(globals())