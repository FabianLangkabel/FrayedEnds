import tequila as tq
import numpy as np
import os
import shutil
from pathlib import Path
import logging
import subprocess as sp
import pyscf
from pyscf import fci
import time
import OrbOpt_helper
import sys
sys.path.append('/Users/timo/workspace/MRA_nanobind/MRA-OrbitalOptimization/build/madness_extension') #Adjust this path to your installation of MRA-OrbitalOptimization
import MadPy as mad


start_time = time.time()
iteration_energies = [] #Stores the energies at the beginning of each iteration step after the VQE
all_occ_number = [] #Stores the orbital occupations at the beginning of each iteration step after the VQE

iterations = 6 #Iterations of the VQE and Orbital-Optimization algorithm

#Parameters for the PNO and Orbital-Optimization calculations
box_size = 50.0 # the system is in a volume of dimensions (box_size*2)^3
wavelet_order = 7 #Default parameter of Orbital-generation, do not change without changing in Orbital-generation!!!
madness_thresh = 0.0001
optimization_thresh = 0.001
NO_occupation_thresh = 0.001

molecule_name = "h2"
distance = 2.5 # Distance between the two hydrogen atoms in Bohr 
distance = distance/2
geometry_bohr = '''
H 0.0 0.0 ''' + distance.__str__() + '''
H 0.0 0.0 ''' + (-distance).__str__()

geometry_angstrom = OrbOpt_helper.convert_geometry_from_bohr_to_angstrom(geometry_bohr)

all_orbitals = [0,1]
frozen_occupied_orbitals = []
active_orbitals = [0,1]
as_dim=len(active_orbitals)

#PNO calculation to get an initial guess for the molecular orbitals
print("Starting PNO calculation")
red=mad.RedirectOutput("PNO.log")
params=tq.quantumchemistry.ParametersQC(name=molecule_name, geometry=geometry_angstrom, basis_set=None, multiplicity=1)
OrbOpt_helper.create_molecule_file(geometry_bohr) # Important here geometry in Bohr
pno=mad.PNOInterface(OrbOpt_helper.PNO_input(params,"molecule",dft={"L":box_size}), box_size, wavelet_order, madness_thresh)
pno.DeterminePNOsAndIntegrals()
all_orbs=pno.GetPNOs(len(frozen_occupied_orbitals),as_dim,0) # input: dimensions of (frozen_occ, active, forzen_virt) space
h1=pno.GetHTensor()
g2=pno.GetGTensor()
c=pno.GetNuclearRepulsion()
del pno
del red
OrbOpt_helper.PNO_cleanup()

peak_loc=[[0.0,0.0,-distance],[0.0,0.0,distance]] #locations of the peaks
sharpness_list=[100.0,100.0] #sharpness of the peaks 
Q=2
PotMaker = mad.CoulombPotentialFromChargeDensity(box_size, wavelet_order, madness_thresh,sharpness_list,Q,peak_loc)
custom_pot=PotMaker.CreatePotential()
PotMaker.plot("custom_potential.dat", custom_pot) #Plot the custom potential
del PotMaker

print("Starting VQE and Orbital-Optimization")
for it in range(iterations):
    print("---------------------------------------------------")
    print("Iteration: " + it.__str__())

    if it == 0:
        mol = tq.Molecule(geometry_angstrom, one_body_integrals=h1, two_body_integrals=g2, nuclear_repulsion=c, name=molecule_name)
    else:
        #todo: transfer nb::ndarray objects directly
        h1=np.array(h1_elements).reshape(as_dim,as_dim)
        g2=np.array(g2_elements).reshape(as_dim,as_dim,as_dim,as_dim)
        g2=tq.quantumchemistry.NBodyTensor(g2, ordering="dirac")
        g2=g2.reorder(to="openfermion")
        mol = tq.Molecule(geometry_angstrom, one_body_integrals=h1, two_body_integrals=g2, nuclear_repulsion=c, name=molecule_name) #Important here geometry in Angstrom
        

    
    #VQE
    U = mol.make_ansatz(name="HCB-UpCCGD")
    if it == 0:
        opt = OrbOpt_helper.get_best_initial_values(mol)
    else:
        opt = tq.quantumchemistry.optimize_orbitals(molecule=mol, circuit=U, silent=True, use_hcb=True, initial_guess=opt.mo_coeff)
    
    mol_new = opt.molecule
    H = mol_new.make_hardcore_boson_hamiltonian()
    U = mol_new.make_ansatz(name="HCB-UpCCGD")
    E = tq.ExpectationValue(H=H, U=U)
    result = tq.minimize(E, silent=True, use_hcb=True)
    print("VQE energy: " + (str)(result.energy))
    iteration_energies.append(result.energy.__str__())

    # Compute 1rdm and 2rdm
    rdm1, rdm2 = mol.compute_rdms(U=U, variables=result.variables, use_hcb=True)
    rdm1, rdm2 = OrbOpt_helper.transform_rdms(opt.mo_coeff.transpose(), rdm1, rdm2)
    
    rdm1_list=rdm1.reshape(-1).tolist()
    rdm2_list=rdm2.reshape(-1).tolist()
    all_occ_number.append(np.sort(np.linalg.eig(rdm1)[0])[::-1])
    
    #Orbital-Optimization
    red=mad.RedirectOutput("OrbOpt"+str(it)+".log")
    opti = mad.Optimization(box_size, wavelet_order, madness_thresh)
    opti.nocc = 2; # spatial orbital = 2; spin orbitals = 1
    opti.truncation_tol = 1e-6
    opti.coulomb_lo = 0.001
    opti.coulomb_eps = 1e-6
    opti.BSH_lo = 0.01
    opti.BSH_eps = 1e-6

    print("Read rdms, create initial guess and calculate initial energy")
    opti.GiveCustomPotential(custom_pot)
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
    for i in range(len(all_orbs)):
        opti.plot("orbital_" + i.__str__()+".dat", all_orbs[i])
    del opti
    del red

# Write energies to the hard disk
with open(r'Energies.txt', 'w') as fp:
    fp.write('\n'.join(iteration_energies))
    
# Write NO-Occupations to the hard disk
all_occ_number_matrix = np.column_stack(all_occ_number)
np.savetxt('all_occ_number.txt', all_occ_number_matrix)
end_time = time.time()

print("Total time: " + (end_time - start_time).__str__())
