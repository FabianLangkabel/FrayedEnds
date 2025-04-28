import tequila as tq
import numpy as np
import os
import shutil
from pathlib import Path
import logging
import subprocess as sp
import pyscf
from pyscf import fci
import OrbOpt_helper
import sys
sys.path.append('/Users/timo/workspace/MRA_nanobind/MRA-OrbitalOptimization/build/madness_extension')
import MadPy as mad

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

all_orbitals = [0,1]
frozen_occupied_orbitals = []
active_orbitals = [0,1]
as_dim=len(active_orbitals)

##################
#Add PNO part here
##################



for it in range(iterations):
    print("---------------------------------------------------")
    print("---------------------------------------------------")
    print("Iteration: " + it.__str__())
    print("---------------------------------------------------")
    print("---------------------------------------------------")

    if it == 0:
        mol = tq.Molecule(geometry_angstrom, n_pno="read", name=molecule_name)
        c, h1, g2 = mol.get_integrals(ordering="chem")
    else:
        h1=np.array(h1_elements).reshape(as_dim,as_dim)
        g2=np.array(g2_elements).reshape(as_dim,as_dim,as_dim,as_dim)
        g2=tq.quantumchemistry.NBodyTensor(g2, ordering="dirac")
        g2=g2.reorder(to="openfermion")
        mol = tq.Molecule(geometry_angstrom, one_body_integrals=h1, two_body_integrals=g2, nuclear_repulsion=c, name=molecule_name) #Important here geometry in Angstrom
        

    
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
    
    rdm1_list=rdm1.reshape(-1).tolist()
    rdm2_list=rdm2.reshape(-1).tolist()
    all_occ_number.append(np.sort(np.linalg.eig(rdm1)[0])[::-1])
    
    #Orbital-Optimization
    opti = mad.Optimization(box_size, wavelet_order, madness_thresh)
    opti.nocc = 2; # spatial orbital = 2; spin orbitals = 1
    opti.truncation_tol = 1e-6
    opti.coulomb_lo = 0.001
    opti.coulomb_eps = 1e-6
    opti.BSH_lo = 0.01
    opti.BSH_eps = 1e-6

    # This only makes sense for H2
    print("Read rdms, create initial guess and calculate initial energy")
    opti.CreateNuclearPotentialAndRepulsion("molecule")
    if it==0:
        opti.ReadInitialOrbitals([], ["mra_orbital_0","mra_orbital_1"], [])
    else:
        opti.GiveInitialOrbitals(all_orbs)
    opti.GiveRDMsAndRotateOrbitals(rdm1_list, rdm2_list)
    opti.CalculateAllIntegrals()
    opti.CalculateCoreEnergy()
    opti.CalculateEnergies()

    print("---------------------------------------------------")
    print("Start orbital optimization")
    opti.OptimizeOrbitals(optimization_thresh, NO_occupation_thresh)
    
    opti.RotateOrbitalsBackAndUpdateIntegrals()

    all_orbs=opti.GetOrbitals()
    c=opti.GetC()
    h1_elements=opti.GetHTensor()
    g2_elements=opti.GetGTensor()

    del opti
    '''
    print("---------------------------------------------------")
    print("---------------------------------------------------")
    print("---------------------------------------------------")
    print("---------------------------------------------------")
    print("---------------------------------------------------")
    print(h1)
    print("\n comp:")
    h_comp=np.load("../../H2/files3/1/htensor.npy")
    print(h_comp)
    print("---------------------------------------------------")
    print(g2)
    print("\n comp:")
    g_comp=np.load("../../H2/files3/1/gtensor.npy")
    print(g_comp)

    for i in range(len(all_orbs)):
        opti.plot("new_orbital" + str(i), opti.loadfct(all_orbs[i]), box_size)
    for i in range(len(all_orbs)):
        opti.plot("old_orbital" + str(i), opti.loadfct_from_file("0/mra_orbital_"+str(i)), box_size)
    '''
# Write energies to the hard disk
with open(r'Energies.txt', 'w') as fp:
    fp.write('\n'.join(iteration_energies))
    
# Write NO-Occupations to the hard disk
all_occ_number_matrix = np.column_stack(all_occ_number)
np.savetxt('all_occ_number.txt', all_occ_number_matrix)