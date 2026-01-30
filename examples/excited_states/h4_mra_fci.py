"""
H4 linear molecule FCI calculation with Orbital Refinement
NWChem molecular orbitals used as initial guesses
"""

import frayedends as fe
import time
import subprocess as sp
import numpy as np
from pyscf import fci

# Parameter Configuration
molecule_name = "h4"
n_electrons = 4  # Number of electrons
iterations = 10  # Number of iterations
box_size = 50.0  # Size of the simulation box
wavelet_order = 7  # Order of wavelet basis functions
madness_thresh = 0.0001  # Threshold for numerical precision of function representation
econv = 1.e-6  # Energy convergence threshold
basisset = '6-31g'  # Initial basis set for calculation

iteration_results = []

# Create .dat files for the results
with open("iteration_nwchem_fci_oo.dat", "w") as f:
    header = "iteration iteration_time_s energy_0"
    f.write(header + "\n")

with open("results_nwchem_fci_oo.dat", "w") as f:
    header = "energy_0"
    f.write(header + "\n")

true_start = time.perf_counter()

# Create NWChem Input file
# Defines a linear H4 molecule geometry with 1.0 Angstrom spacing between adjacent atoms
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
    world.line_plot(f"initial_orb{i}.dat", orbs[i], axis="z", datapoints=2001)  # Plot guess orbitals

current = 0.0
for iteration in range(iterations):
    iter_start = time.perf_counter()

    integrals = fe.Integrals3D(world)  # Setup for integrals
    G = integrals.compute_two_body_integrals(orbs, ordering="chem").elems # g-tensor (electron-electron interaction)
    T = integrals.compute_kinetic_integrals(orbs)  # Kinetic energy
    V = integrals.compute_potential_integrals(orbs, Vnuc)  # Potential energy

    # Full Configuration Interaction (FCI) calculation
    e, fcivec = fci.direct_spin0.kernel(T + V, G, n_orbitals,
                                        n_electrons)  # Compute the energy and the FCI vector
    # Calculate reduced density matrices needed for orbital refinement
    rdm1, rdm2 = fci.direct_spin0.make_rdm12(fcivec, n_orbitals,
                                             n_electrons)  # Compute the 1- and 2- body rdms
    rdm2 = np.swapaxes(rdm2, 1, 2) # Change to physics notation

    e_tot = e + nuclear_repulsion_energy # Computes total energy

    print("iteration {} FCI electronic energy {:+2.8f}, total energy {:+2.8f}".format(iteration, e, e_tot))

    # Orbital Refinement
    opti = fe.Optimization3D(world, Vnuc, nuclear_repulsion_energy)
    orbs = opti.get_orbitals(
        orbitals=orbs, rdm1=rdm1, rdm2=rdm2, opt_thresh=0.001, occ_thresh=0.001
    )  # Refines the orbitals and returns the new ones

    for i in range(n_orbitals):
        world.line_plot(f"orb{i}.dat", orbs[i], axis="z", datapoints=2001)  # Plot the refined orbitals

    iter_end = time.perf_counter()
    iter_time = iter_end - iter_start

    with open("iteration_nwchem_fci_oo.dat", "a") as f:
        f.write(f"{iteration} {iter_time:.2f} {e_tot: .15f}" + "\n")

    iteration_results.append({"iteration": iteration, "iteration_time": iter_time, "energy": e_tot})

    if np.isclose(e_tot, current, atol=econv, rtol=0.0):
        break  # Loop terminates as soon as the energy changes less than econv in one iteration step
    current = e_tot

with open("results_nwchem_fci_oo.dat", "a") as f:
    f.write(f"{e_tot: .15f}" + "\n")

fe.cleanup(globals())