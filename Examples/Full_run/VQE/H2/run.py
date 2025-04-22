import tequila as tq
import numpy as np
import os
import shutil
import json
from pathlib import Path
import logging
import subprocess as sp
import pyscf
from pyscf import fci
import OrbOpt_helper


'''
Definitions/Parameters
'''                       
distance = 2.5 # Distance between the two hydrogen atoms in Bohr 
iteration_energies = [] #Stores the energies at the beginning of each iteration step after the VQE
all_occ_number = [] #Stores the orbital occupations at the beginning of each iteration step after the VQE
iterations = 6
molecule_name = "h2"
box_size = 50.0 # the system is in a volume of dimensions (box_size*2)^3
wavelet_order = 7 #Default parameter of Orbital-generation, do not change without changing in Orbital-generation!!!
madness_thresh = 0.0001
optimization_thresh = 0.001
NO_occupation_thresh = 0.001

def get_best_initial_values(mol):
    tries = 20
    U = mol.make_ansatz(name="HCB-UpCCGD")
    best_opt = tq.quantumchemistry.optimize_orbitals(molecule=mol, circuit=U, silent=True, use_hcb=True, initial_guess="random")
    opt = tq.quantumchemistry.optimize_orbitals(molecule=mol, circuit=U, silent=True, use_hcb=True)
    if opt.energy < best_opt.energy:
        best_opt = opt
    
    for _ in range(tries):
        #opt = tq.quantumchemistry.optimize_orbitals(molecule=mol, circuit=U, silent=True, use_hcb=True, initial_guess="random")
        initial_guess = np.eye(mol.n_orbitals) + np.random.normal(scale=1.0, loc=0.0, size=mol.n_orbitals**2).reshape(mol.n_orbitals, mol.n_orbitals)
        opt = tq.quantumchemistry.optimize_orbitals(molecule=mol, circuit=U, silent=True, use_hcb=True, initial_guess=initial_guess)
        if opt.energy < best_opt.energy:
            best_opt = opt
            
    return best_opt

distance = distance/2
geometry_bohr = '''
H 0.0 0.0 ''' + distance.__str__() + '''
H 0.0 0.0 ''' + (-distance).__str__()

geometry_angstrom = OrbOpt_helper.convert_geometry_from_bohr_to_angstrom(geometry_bohr)


OrbOpt_helper.create_molecule_file(geometry_bohr) # Important here geometry in Bohr

mol = tq.Molecule(geometry_angstrom, dft={"L":box_size}, name=molecule_name) # Important here geometry in Angstrom           

print(mol.compute_energy("fci"))
print(mol.compute_energy("hf"))
 
all_orbitals = [0,1]
frozen_occupied_orbitals = []
active_orbitals = [0,1]



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
    
    if it == 0:
            # Copy h and g tensors for VQE to current folder to read molecule
            shutil.copy(it_str + "/htensor.npy", molecule_name + "_htensor.npy")
            shutil.copy(it_str + "/gtensor.npy", molecule_name + "_gtensor.npy")
            mol = tq.Molecule(geometry_angstrom, n_pno="read", name=molecule_name) #Important here geometry in Angstrom
            c, h1, h2 = mol.get_integrals(ordering="chem")
            print(c)
            print(mol.compute_energy("fci"))
            print(mol.compute_energy("hf"))
    else:
        with open(it_str + '/c.txt') as f:
            c = float(f.readline())
            print(c)
        h1 = np.load(it_str + "/htensor.npy")
        g2 = np.load(it_str + "/gtensor.npy")
        g2=tq.quantumchemistry.NBodyTensor(g2, ordering="dirac")
        g2=g2.reorder(to="openfermion")
        mol = tq.Molecule(geometry_angstrom, one_body_integrals=h1, two_body_integrals=g2, nuclear_repulsion=c, name=molecule_name) #Important here geometry in Angstrom
        print(mol.compute_energy("fci"))
        print(mol.compute_energy("hf"))

    
    #VQE
    U = mol.make_ansatz(name="HCB-UpCCGD")
    if it == 0:
        opt = get_best_initial_values(mol)
    else:
        opt = tq.quantumchemistry.optimize_orbitals(molecule=mol, circuit=U, silent=True, use_hcb=True, initial_guess=opt.mo_coeff)

    mol_new = opt.molecule
    H = mol_new.make_hardcore_boson_hamiltonian()
    U = mol_new.make_ansatz(name="HCB-UpCCGD")
    E = tq.ExpectationValue(H=H, U=U)
    result = tq.minimize(E, silent=True, use_hcb=True)
    print("VQE energy: " + (str)(result.energy))
    iteration_energies.append(result.energy.__str__())

    # Write 1rdm and 2rdm for iteration step
    rdm1, rdm2 = mol.compute_rdms(U=U, variables=result.variables, use_hcb=True)
    rdm1, rdm2 = OrbOpt_helper.transform_rdms(opt.mo_coeff.transpose(), rdm1, rdm2)
    
    #OrbOpt_helper.write_rdms(rdm1, rdm2, it_str + "/" + molecule_name)
    np.save(it_str + "/" + molecule_name + "_1rdm.npy", rdm1)
    np.save(it_str + "/" + molecule_name + "_2rdm.npy", rdm2)
    all_occ_number.append(np.sort(np.linalg.eig(rdm1)[0])[::-1])
    
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
    programm = sp.call("/Users/timo/workspace/MRA_nanobind/MRA-OrbitalOptimization/build/orbital_optimization/OrbitalOptimization madness_input.json", stdout=open(it_str + '/log', 'w'), stderr=open(it_str + '/err_log', 'w'), shell = True)

# Write energies to the hard disk
with open(r'Energies.txt', 'w') as fp:
    fp.write('\n'.join(iteration_energies))
    
# Write NO-Occupations to the hard disk
all_occ_number_matrix = np.column_stack(all_occ_number)
np.savetxt('all_occ_number.txt', all_occ_number_matrix)
