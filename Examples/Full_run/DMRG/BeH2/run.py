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
optimization_thresh = 0.001
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
    # Copy h and g tensors for VQE to current folder to read molecule
    shutil.copy(it_str + "/htensor.npy", molecule_name + "_htensor.npy")
    shutil.copy(it_str + "/gtensor.npy", molecule_name + "_gtensor.npy")
    
    # Read molecule
    mol = tq.Molecule(geometry_angstrom, n_pno="read", name=molecule_name) #Important here geometry in Angstrom
    
    ################### DMRG ###################
    threads = 4
    ram_gb = 4
    bd_pre = 100
    bd_reordered = 200

    c, h1, h2 = mol.get_integrals(ordering="chem")

    #### First DMRG calculation
    driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SU2, n_threads=threads, stack_mem=int(60*1024**3), restart_dir="./restart")
    driver.initialize_system(n_sites=mol.n_orbitals, n_elec=mol.n_electrons, spin=0)
    mpo = driver.get_qc_mpo(h1e=h1, g2e=h2.elems, ecore=c, iprint=1)
    ket = driver.get_random_mps(tag="GS", bond_dim=250, nroots=1)
    energy = driver.dmrg(mpo, ket, n_sweeps=500, bond_dims=[bd_pre], iprint=1)
    print('DMRG energy = %20.15f' % energy)

    #### Orbital reordering
    idx = driver.orbital_reordering(h1, h2.elems)
    print(idx)
    h1_new = h1[idx][:, idx]
    g2_new = h2.elems[idx][:, idx][:, :, idx][:, :, :, idx]

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
    pdm2 = np.swapaxes(pdm2,1,2) #WICHTIG: UMSORTIERUNG VON RDMS IN PYSCF
    print('Energy from reordered pdms = %20.15f' % (np.einsum('ij,ij->', pdm1, h1) + 0.5 * np.einsum('ijkl,ikjl->', pdm2, driver.unpack_g2e(h2.elems)) + c))
    OrbOpt_helper.write_rdms(pdm1, pdm2, it_str + "/" + molecule_name)

    dmrg_energy = np.einsum('ij,ij->', pdm1, h1) + 0.5 * np.einsum('ijkl,ikjl->', pdm2, driver.unpack_g2e(h2.elems)) + c
    iteration_energies.append(dmrg_energy.__str__())

    ################### DMRG finished ###################

    
    # Delete h and g tensors from the current folder and create folder for next iteration step
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
    programm = sp.call("/workspaces/MRA-OrbitalOptimization/build/OrbitalOptimization madness_input.json", stdout=open(it_str + '/log', 'w'), stderr=open(it_str + '/err_log', 'w'), shell = True)

    
# Write energies to the hard disk
with open(r'Energies.txt', 'w') as fp:
    fp.write('\n'.join(iteration_energies))