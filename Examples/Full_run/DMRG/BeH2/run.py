import tequila as tq
import numpy as np
import os
import shutil
import json
from pathlib import Path
import logging
import subprocess as sp
import OrbOpt_helper
from pyblock2._pyscf.ao2mo import integrals as itg
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

'''
Definitions/Parameters
'''                       
distance = 2.5
iteration_energies = [] #Stores the energies at the beginning of each iteration step after the VQE
iterations = 6
molecule_name = "beh2"
box_size = 50.0
wavelet_order = 7 #Default parameter of Orbital-generation, do not change without changing in Orbital-generation!!!
madness_thresh = 0.0001
optimization_thresh = 0.005
NO_occupation_thresh = 0.001

geometry_bohr = '''
Be 0.0 0.0 0.0
H 0.0 0.0 ''' + distance.__str__() + '''
H 0.0 0.0 ''' + (-distance).__str__()

geometry_angstrom = OrbOpt_helper.convert_geometry_from_bohr_to_angstrom(geometry_bohr)


OrbOpt_helper.create_molecule_file(geometry_bohr) # Important here geometry in Bohr
mol = tq.Molecule(geometry_angstrom, dft={"L":box_size}, name=molecule_name, n_pno=4, pno={"maxrank":2}) # Important here geometry in Angstrom
all_orbitals = [0,1,2,3,4,5,6]
frozen_occupied_orbitals = [0]
active_orbitals = [1,2,3,4,5,6]



# Copy initial orbitals and intergals to folder for first step
os.mkdir("0")
for orb in all_orbitals:
    shutil.move("mra_orbital_" + str(orb) + ".00000", "0/mra_orbital_" + str(orb) + ".00000")
shutil.move(molecule_name + "_htensor.npy", "0/htensor.npy")
shutil.move(molecule_name + "_gtensor.npy", "0/gtensor.npy")


'''
Iteration Loop
'''
for it in range(iterations):
    it_str = it.__str__()
    print("Iteration-Step: " + it_str)

    # Read molecule
    if it == 0:
        # Copy h and g tensors for VQE to current folder to read molecule
        shutil.copy(it_str + "/htensor.npy", molecule_name + "_htensor.npy")
        shutil.copy(it_str + "/gtensor.npy", molecule_name + "_gtensor.npy")
        mol = tq.Molecule(geometry_angstrom, n_pno="read", name=molecule_name) #Important here geometry in Angstrom
        c, h1, h2 = mol.get_integrals(ordering="chem")
        g2 = h2.elems

    else:
        with open(it_str + '/c.txt') as f:
            c = float(f.readline())
        h1 = np.load(it_str + "/htensor.npy")
        g2 = np.load(it_str + "/gtensor.npy")
        g2 = np.swapaxes(g2,1,2) # Integrals are in physics notation, Block2 uses chemistry notation
    
    ################### DMRG ###################
    threads = 4
    ram_gb = 4
    bd_pre = 100
    bd_reordered = 200

    #### First DMRG calculation
    driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SU2, n_threads=threads, stack_mem=int(ram_gb*1024**3), restart_dir="./restart")
    driver.initialize_system(n_sites=mol.n_orbitals, n_elec=mol.n_electrons, spin=0)
    mpo = driver.get_qc_mpo(h1e=h1, g2e=g2, ecore=c, iprint=1)
    ket = driver.get_random_mps(tag="GS", bond_dim=250, nroots=1)
    energy = driver.dmrg(mpo, ket, n_sweeps=500, bond_dims=[bd_pre], iprint=1)
    print('DMRG energy = %20.15f' % energy)

    #### Orbital reordering
    idx = driver.orbital_reordering(h1, g2)
    print(idx)
    h1_new = h1[idx][:, idx]
    g2_new = g2[idx][:, idx][:, :, idx][:, :, :, idx]

    #### Main DMRG calculation
    driver.initialize_system(n_sites=mol.n_orbitals, n_elec=mol.n_electrons, spin=0)
    mpo = driver.get_qc_mpo(h1e=h1_new, g2e=g2_new, ecore=c, iprint=1)
    ket = driver.get_random_mps(tag="GS", bond_dim=250, nroots=1)
    energy = driver.dmrg(mpo, ket, n_sweeps=500, bond_dims=[bd_reordered], iprint=1)
    print('DMRG energy = %20.15f' % energy)

    #### PDM extraction
    pdm1 = driver.get_1pdm(ket)
    pdm2 = driver.get_2pdm(ket).transpose(0, 3, 1, 2)
    print('Energy from pdms = %20.15f' % (np.einsum('ij,ij->', pdm1, h1_new) + 0.5 * np.einsum('ijkl,ijkl->', pdm2, driver.unpack_g2e(g2_new)) + c))

    idx_back = np.zeros(len(idx), dtype=int)
    for i in range(len(idx)):
        idx_back[idx[i]] = i
        
    pdm1 = pdm1[idx_back][:, idx_back]
    pdm2 = pdm2[idx_back][:, idx_back][:, :, idx_back][:, :, :, idx_back]
    pdm2 = np.swapaxes(pdm2,1,2) # chemistry to physics notation
    print('Energy from reordered pdms = %20.15f' % (np.einsum('ij,ij->', pdm1, h1) + 0.5 * np.einsum('ijkl,ikjl->', pdm2, driver.unpack_g2e(g2)) + c))
    np.save(it_str + "/" + molecule_name + "_1rdm.npy", pdm1)
    np.save(it_str + "/" + molecule_name + "_2rdm.npy", pdm2)

    dmrg_energy = np.einsum('ij,ij->', pdm1, h1) + 0.5 * np.einsum('ijkl,ikjl->', pdm2, driver.unpack_g2e(g2)) + c
    iteration_energies.append(dmrg_energy.__str__())

    ################### DMRG finished ###################

    
    # Delete h and g tensors from the current folder and create folder for next iteration step
    if it == 0:
        os.remove(molecule_name + "_htensor.npy")
        os.remove(molecule_name + "_gtensor.npy")

    os.mkdir((it + 1).__str__())
    
    # Create OrbitalOptimization input file
    OrbOptFilePath = "madness_input.json"
    OrbOpt_helper.create_orbital_opt_input(OrbOptFilePath, it, all_orbitals, frozen_occupied_orbitals, active_orbitals, 
                                 box_size, wavelet_order, madness_thresh, optimization_thresh, NO_occupation_thresh,
                                 molecule_name)
    
    shutil.copy("madness_input.json", it_str + "/madness_input.json")

    # Execute orbital optimization
    programm = sp.call("/workspaces/MRA-OrbitalOptimization/build/orbital_optimization/OrbitalOptimization madness_input.json", stdout=open(it_str + '/log', 'w'), stderr=open(it_str + '/err_log', 'w'), shell = True)

    
# Write energies to the hard disk
with open(r'Energies.txt', 'w') as fp:
    fp.write('\n'.join(iteration_energies))