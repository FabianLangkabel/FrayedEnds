import time
import subprocess as sp
import numpy as np
import tequila as tq
from pyscf import fci

import madpy as mad

distance = np.arange(2.5, 0.50, -0.05).tolist()
iteration_energies = []
n_electrons = 4  # Number of electrons
iterations = 6
box_size = 50.0
wavelet_order = 7
madness_thresh = 0.0001
econv = 1.e-8 # Energy convergence threshold
basisset = '6-31g'

results = []

with open("results.dat", "w") as f:
    header = "distance iteration iteration_time_s energy_0"
    f.write(header + "\n")

true_start = time.perf_counter()
for d in distance:
    dist_start = time.perf_counter()
    nwchem_input = '''
    title "molecule"
    memory stack 1500 mb heap 100 mb global 1400 mb
    charge 0  
    geometry units angstrom noautosym nocenter
        H 0.0 0.0 ''' + (-d-d/2).__str__() + '''
        H 0.0 0.0 ''' + (-d/2).__str__() + '''
        H 0.0 0.0 ''' + (d/2).__str__() + '''
        H 0.0 0.0 ''' + (d+d/2).__str__() + '''
    end
    basis  
      * library ''' + basisset + '''
    end
    scf  
     maxiter 200
    end   
    task scf  
    '''
    # two H2 molecules getting closer and closer to a H4 molecule
    # H 0.0 0.0 ''' + (-d - 2.55).__str__() + '''
    # H 0.0 0.0 ''' + (-d).__str__() + '''
    # H 0.0 0.0 ''' + d.__str__() + '''
    # H 0.0 0.0 ''' + (d + 2.55).__str__() + '''

    with open("nwchem", "w") as f:
        f.write(nwchem_input)
    programm = sp.call("/opt/anaconda3/envs/frayedends/bin/nwchem nwchem", stdout=open('nwchem.out', 'w'), stderr=open('nwchem_err.log', 'w'), shell = True)


    '''
    ### Convert NWChem AOs and MOs to MRA-Orbitals
    Read the atomic orbitals (AOs) and molecular orbitals (MOs) from a NWChem calculation and translate them into multiwavelets.
    '''

    world = mad.MadWorld3D(L=box_size, k=wavelet_order, thresh=madness_thresh)

    converter = mad.NWChem_Converter(world)
    converter.read_nwchem_file("nwchem")
    orbs = converter.get_mos()
    Vnuc = converter.get_Vnuc()
    nuclear_repulsion_energy = converter.get_nuclear_repulsion_energy()
    n_orbitals = len(orbs)
    del converter

    for i in range(len(orbs)):
        orbs[i].type="active"

    #for i in range(len(orbs)):
    #    world.line_plot(f"es_orb{i}.dat", orbs[i], axis="z", datapoints=2001)  # Plots guess orbitals

    current = 0.0
    for iteration in range(iterations):
        iter_start = time.perf_counter()

        integrals = mad.Integrals3D(world)  # Setup for integrals
        G = integrals.compute_two_body_integrals(orbs, ordering="chem") # g-tensor (electron-electron interaction)
        T = integrals.compute_kinetic_integrals(orbs)  # Kinetic energy
        V = integrals.compute_potential_integrals(orbs, Vnuc)  # Potential energy (h-tensor=T+V)

        # FCI calculation
        e, fcivec = fci.direct_spin0.kernel(T + V, G.elems, n_orbitals,
                                            n_electrons)  # Computes the energy and the FCI vector
        rdm1, rdm2 = fci.direct_spin0.make_rdm12(fcivec, n_orbitals,
                                                 n_electrons)  # Computes the 1- and 2- body reduced density matrices
        rdm2 = np.swapaxes(rdm2, 1, 2)

        e_elec = e
        e_tot = e_elec + nuclear_repulsion_energy

        iter_end = time.perf_counter()
        iter_time = iter_end - iter_start
        print("iteration {} FCI electronic energy {:+2.8f}, total energy {:+2.8f}".format(iteration, e_elec, e_tot))

        with open("iteration_times.dat", "a") as f:
            f.write(f"{d:.6f} {iteration} {iter_time:.6f}\n")

        with open("results.dat", "a") as f:
            f.write(f"{d:.6f} {iteration} {iter_time:.6f} {e_tot: .15f}" + "\n")

        results.append({"distance": d, "iteration": iteration, "iteration_time": iter_time, "energy": e})

        # Orbital optimization
        opti = mad.Optimization3D(world, Vnuc, nuclear_repulsion_energy)
        orbs = opti.get_orbitals(
            orbitals=orbs, rdm1=rdm1, rdm2=rdm2, opt_thresh=0.001, occ_thresh=0.001
        )  # Optimizes the orbitals and returns the new ones

        #for i in range(len(orbs)):
        #    world.line_plot(f"orb{i}.dat", orbs[i])  # Plots the optimized orbitals

        if np.isclose(e, current, atol=econv, rtol=0.0):
            break  # The loop terminates as soon as the energy changes less than econv in one iteration step
        current = e_tot

    dist_end = time.perf_counter()
    dist_time = dist_end - dist_start
    print(f"Distance {d:.6f} took {dist_time:.2f} s")
    with open("distance_times.dat", "a") as f:
        f.write(f"{d:.6f} {dist_time:.6f}\n")

    del integrals
    del opti
    del world

true_end = time.perf_counter()
total_time = true_end - true_start
print("Total time: ", total_time)
with open("total_time.dat", "w") as f:
    f.write(f"total_runtime_s {total_time:.6f}\n")