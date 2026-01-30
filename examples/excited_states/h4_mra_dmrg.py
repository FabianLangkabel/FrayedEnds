"""
H4 linear molecule State Average (SA) DMRG calculation with Orbital Refinement
NWChem molecular orbitals used as initial guesses
"""

import subprocess as sp
import frayedends as fe
import time
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
import numpy as np

# Parameter Configuration
molecule_name = "h4"
n_elec = 4  # Number of electrons
number_roots = 3    # Number of states (groundstate, 1. excited state, 2. excited state)
iterations = 6      # Number of iterations
box_size = 50.0     # Size of the simulation box
wavelet_order = 7   # Order of wavelet basis functions
madness_thresh = 0.0001 # Threshold for numerical precision of function representation
basisset = '6-31g'  # Initial basis set for calculation

iteration_results = []

# Create .dat files for the results
with open("iteration_nwchem_dmrg_oo.dat", "w") as f:
    header = "iteration iteration_time_s " + " ".join(f"energy_{i}" for i in range(number_roots))
    f.write(header + "\n")

with open("results_nwchem_dmrg_oo.dat", "w") as f:
    header = " ".join(f"energy_{i}" for i in range(number_roots))
    f.write(header + "\n")

total_start = time.perf_counter()

# Create NWChem Input file
# Define a linear H4 molecule geometry with 1.0 Angstrom spacing between adjacent atoms

nwchem_input = '''
title "molecule"
memory stack 1500 mb heap 100 mb global 1400 mb
charge 0  
geometry units angstrom noautosym nocenter
    H 0.0 0.0 -1.5
    H 0.0 0.0 -0.5
    H 0.0 0.0 0.5
    H 0.0 0.0 1.5
end
basis  
  * library ''' + basisset + '''
end
scf  
 maxiter 200
end   
task scf  
'''

# Run NWchem calculations
with open("nwchem", "w") as f:
    f.write(nwchem_input)
programm = sp.call("/opt/anaconda3/envs/frayedends/bin/nwchem nwchem", stdout=open('nwchem.out', 'w'), stderr=open('nwchem_err.log', 'w'), shell = True)

# Setting up the numerical environment for the MRA calculations
world = fe.MadWorld3D(L=box_size, k=wavelet_order, thresh=madness_thresh)

# Read the atomic orbitals (AOs) and the molecular orbitals (MOs) from the NWchem calculation and translate them into multiwavelets
converter = fe.NWChem_Converter(world)
converter.read_nwchem_file("nwchem")
orbs = converter.get_mos()
Vnuc = converter.get_Vnuc()
nuclear_repulsion_energy = converter.get_nuclear_repulsion_energy()

n_orbitals = len(orbs)

for i in range(n_orbitals):
    orbs[i].type="active"

for i in range(n_orbitals):
    world.line_plot(f"initial_orb{i}.dat", orbs[i], axis="z", datapoints=2001) # Plot guess orbitals

# Calculate initial integrals
integrals = fe.Integrals3D(world)
G = integrals.compute_two_body_integrals(orbs, ordering="chem").elems # g-tensor (electron-electron interaction)
T = integrals.compute_kinetic_integrals(orbs) # Kinetic energy
V = integrals.compute_potential_integrals(orbs, Vnuc) # Potential energy
S = integrals.compute_overlap_integrals(orbs) # Overlap

# Performe State Average (SA) DMRG calculation and extract rdms
driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SU2, n_threads=8)
driver.initialize_system(n_sites=n_orbitals, n_elec=n_elec, spin=0)
mpo = driver.get_qc_mpo(h1e=T+V, g2e=G, ecore=nuclear_repulsion_energy, iprint=0)
ket = driver.get_random_mps(tag="KET", bond_dim=100, nroots=number_roots)
energies = driver.dmrg(mpo, ket, n_sweeps=10, bond_dims=[100], noises=[1e-5] * 4 + [0], thrds=[1e-10] * 8, iprint=1)
print('State-averaged MPS energies = [%s]' % " ".join("%20.15f" % x for x in energies))

# Extract rdms
kets = [driver.split_mps(ket, ir, tag="KET-%d" % ir) for ir in range(ket.nroots)]
sa_1pdm = np.mean([driver.get_1pdm(k) for k in kets], axis=0)   # Compute the state average 1-body rdm
sa_2pdm = np.mean([driver.get_2pdm(k) for k in kets], axis=0).transpose(0,3,1,2) # Compute the state average 2-body rdm
print('Energy from SA-pdms = %20.15f' % (np.einsum('ij,ij->', sa_1pdm, T+V) + 0.5 * np.einsum('ijkl,ijkl->', sa_2pdm, G) + nuclear_repulsion_energy))
sa_2pdm_phys = sa_2pdm.swapaxes(1,2) #Physics Notation

with open("iteration_nwchem_dmrg_oo.dat", "a") as f:
    f.write(f"{-1} {0.00} " + " ".join(f"{x:.15f}" for x in energies) + "\n")

for iter in range(iterations):
    iter_start = time.perf_counter()

    # Orbital Refinement
    opti = fe.Optimization3D(world, Vnuc, nuclear_repulsion_energy)
    orbs = opti.get_orbitals(orbitals=orbs, rdm1=sa_1pdm, rdm2=sa_2pdm_phys, opt_thresh=0.001, occ_thresh=0.001)

    for i in range(len(orbs)):
        world.line_plot(f"orb{i}.dat", orbs[i], axis="z", datapoints=2001) # Plot the refined orbitals

    # DMRG calculation with refined orbitals
    G = integrals.compute_two_body_integrals(orbs, ordering="chem").elems # g-tensor (electron-electron interaction)
    T = integrals.compute_kinetic_integrals(orbs) # Kinetic energy
    V = integrals.compute_potential_integrals(orbs, Vnuc) # Potential energy
    S = integrals.compute_overlap_integrals(orbs) # Overlap

    driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SU2, n_threads=8)
    driver.initialize_system(n_sites=n_orbitals, n_elec=n_elec, spin=0)
    mpo = driver.get_qc_mpo(h1e=T+V, g2e=G, ecore=nuclear_repulsion_energy, iprint=0)
    ket = driver.get_random_mps(tag="KET", bond_dim=100, nroots=number_roots)
    energies = driver.dmrg(mpo, ket, n_sweeps=10, bond_dims=[100], noises=[1e-5] * 4 + [0], thrds=[1e-10] * 8, iprint=1)
    print('State-averaged MPS energies after refinement = [%s]' % " ".join("%20.15f" % x for x in energies))
    np.savetxt("energies_it_" + str(iter) + ".txt", energies)

    kets = [driver.split_mps(ket, ir, tag="KET-%d" % ir) for ir in range(ket.nroots)]
    sa_1pdm = np.mean([driver.get_1pdm(k) for k in kets], axis=0) # Compute the state average 1-body rdm
    sa_2pdm = np.mean([driver.get_2pdm(k) for k in kets], axis=0).transpose(0,3,1,2) # Compute the state average 2-body rdm
    print('Energy from SA-pdms = %20.15f' % (np.einsum('ij,ij->', sa_1pdm, T+V) + 0.5 * np.einsum('ijkl,ijkl->', sa_2pdm, G) + nuclear_repulsion_energy))
    sa_2pdm_phys = sa_2pdm.swapaxes(1,2) # Change to physics Notation

    iter_end = time.perf_counter()
    iter_time = iter_end - iter_start

    with open("iteration_nwchem_dmrg_oo.dat", "a") as f:
        f.write(f"{iter} {iter_time:.2f} " + " ".join(f"{x:.15f}" for x in energies) + "\n")

    iteration_results.append({"iteration": iter, "iteration_time": iter_time, "energies": energies})

with open("results_nwchem_dmrg_oo.dat", "a") as f:
    f.write(" ".join(f"{x:.15f}" for x in energies) + "\n")

fe.cleanup(globals())