"""
H4 linear molecule DMRG calculation (without Orbital Refinement)
Pair Natural Orbitals (PNOs) used as initial guesses
"""

import frayedends as fe
import time
import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

# Parameter Configuration
molecule_name = "h4"
n_elec = 4  # Number of electrons
number_roots = 3    # Number of states (groundstate, 1. excited state, 2. excited state)
box_size = 50.0     # Size of the simulation box
wavelet_order = 7   # Order of wavelet basis functions
madness_thresh = 0.0001 # Threshold for numerical precision of function representation
basisset = '6-31g'  # Initial basis set for calculation

iteration_results = []

# Create .dat files for the results
with open("results_pno_dmrg.dat", "w") as f:
    header = "distance " + " ".join(f"energy_{i}" for i in range(number_roots))
    f.write(header + "\n")

total_start = time.perf_counter()

# Define a linear H4 molecule geometry with 1.0 Angstrom spacing between adjacent atoms
geom = ("H 0.0 0.0 -1.5 \n"
        "H 0.0 0.0 -0.5 \n"
        "H 0.0 0.0 0.5 \n"
        "H 0.0 0.0 1.5 \n")

# Setting up the numerical environment for the MRA calculations
world = fe.MadWorld3D(L=box_size, k=wavelet_order, thresh=madness_thresh)

# Get 8 Pair Natural Orbitals (PNOs)
madpno = fe.MadPNO(world, geom, n_orbitals=8)
orbs = madpno.get_orbitals()

nuc_repulsion = madpno.get_nuclear_repulsion() # Compute nuclear repulsion energy
Vnuc = madpno.get_nuclear_potential() # Compute nuclear potential

integrals = fe.Integrals3D(world)
orbs = integrals.orthonormalize(orbitals=orbs) # Orthonormalize orbitals

n_orbitals = len(orbs)

for i in range(n_orbitals):
    orbs[i].type="active"

for i in range(n_orbitals):
    world.line_plot(f"orb{i}.dat", orbs[i], axis="z", datapoints=2001) # Plot PNOs

# Calculate initial integrals
integrals = fe.Integrals3D(world)
G = integrals.compute_two_body_integrals(orbs, ordering="chem").elems # g-tensor (electron-electron interaction)
T = integrals.compute_kinetic_integrals(orbs) # Kinetic energy
V = integrals.compute_potential_integrals(orbs, Vnuc) # Potential energy
S = integrals.compute_overlap_integrals(orbs) # Overlap

# Performe DMRG calculation and extract rdms
driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SU2, n_threads=8)
driver.initialize_system(n_sites=n_orbitals, n_elec=n_elec, spin=0)
mpo = driver.get_qc_mpo(h1e=T+V, g2e=G, ecore=nuc_repulsion, iprint=0)
ket = driver.get_random_mps(tag="KET", bond_dim=100, nroots=number_roots)
energies = driver.dmrg(mpo, ket, n_sweeps=10, bond_dims=[100], noises=[1e-5] * 4 + [0], thrds=[1e-10] * 8, iprint=1)

with open("results_pno_dmrg.dat", "a") as f:
    f.write(" ".join(f"{x:.15f}" for x in energies) + "\n")

fe.cleanup(globals())