import tequila as tq
import numpy as np
import time
import OrbOpt_helper
import madpy as mad


start_time = time.time()
iteration_energies = [] #Stores the energies at the beginning of each iteration step after the VQE
all_occ_number = [] #Stores the orbital occupations at the beginning of each iteration step after the VQE

iterations = 6 #Iterations of the VQE and Orbital-Optimization algorithm

#Parameters for the PNO and Orbital-Optimization calculations
box_size = 50.0 # the system is in a volume of dimensions (box_size*2)^3
wavelet_order = 7 #Default parameter of Orbital-generation, do not change without changing in Orbital-generation!!!
madness_thresh = 0.00001
optimization_thresh = 0.001
NO_occupation_thresh = 0.001

molecule_name = "He2"
distance = 0.0 # Distance between the two hydrogen atoms in Bohr 
distance = distance/2
geometry_bohr = '''
He 0.0 0.0 ''' + distance.__str__() + '''
He 0.0 0.0 ''' + (-distance).__str__()



print(geometry_bohr)
geometry_angstrom = OrbOpt_helper.convert_geometry_from_bohr_to_angstrom(geometry_bohr)

all_orbitals = [0,1,2,3]
frozen_occupied_orbitals = []
active_orbitals = [0,1,2,3]
as_dim=len(active_orbitals)


#Custom potential is created here
class DoubleWellPot:
    a=-10.0
    mu=np.array([0.0,0.0,-2.0])
    sigma=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    b=-5.0
    mu2=np.array([0.0,0.0,2.0])
    sigma2=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    c=-7.5
    mu3=np.array([0.0,0.0,-4.0])
    sigma3=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    d=-2.5
    mu4=np.array([0.0,0.0,4.0])
    sigma4=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    def __init__(self):
        pass
    def pot(self, x, y, z): #actual potential function
        r= np.array([x, y, z])
        diff1 = np.zeros(3)
        for i in range(3):
            diff1[i] = r[i] - self.mu[i]
    
        sigma_inv = np.linalg.inv(self.sigma)  # inverse of sigma (covariance matrix)
        sum1 = 0.0

        for i in range(3): 
            for j in range(3):
                sum1 += diff1[i] * sigma_inv[i, j] * diff1[j]
        

        potential1 = self.a * np.exp(- 0.5 * sum1)

        diff2 = np.zeros(3)
        for i in range(3):
            diff2[i] = r[i] - self.mu2[i]
        
        sigma2_inv = np.linalg.inv(self.sigma2)
        sum2 = 0.0

        for i in range(3): 
            for j in range(3):
                sum2 += diff2[i] * sigma2_inv[i, j] * diff2[j]

        potential2 = self.b * np.exp(- 0.5 * sum2)

        diff3 = np.zeros(3)
        for i in range(3):
            diff3[i] = r[i] - self.mu3[i]
        sigma3_inv = np.linalg.inv(self.sigma3)
        sum3 = 0.0

        for i in range(3):
            for j in range(3):
                sum3 += diff3[i] * sigma3_inv[i, j] * diff3[j]
        
        potential3 = self.c * np.exp(- 0.5 * sum3)

        diff4 = np.zeros(3)
        for i in range(3):
            diff4[i] = r[i] - self.mu4[i]
        
        sigma4_inv = np.linalg.inv(self.sigma4)
        sum4 = 0.0

        for i in range(3):
            for j in range(3):
                sum4 += diff4[i] * sigma4_inv[i, j] * diff4[j]
        
        potential4 = self.d * np.exp(- 0.5 * sum4)
        
        return potential1 + potential2 + potential3 + potential4


MP=DoubleWellPot()
factory=mad.PyFuncFactory(box_size, wavelet_order, madness_thresh,MP.pot)
mra_pot=factory.GetMRAFunction()
del factory


opti=mad.Optimization(box_size, wavelet_order, madness_thresh)
opti.plot("custom_pot.dat",mra_pot)
opti.plane_plot("pot.dat",mra_pot,plane="yz",zoom=10.0,datapoints=71,origin=[0.0,0.0,0.0])
del opti


eigensolver = mad.Eigensolver(box_size, wavelet_order, madness_thresh)
eigensolver.solve(mra_pot, 10, 10)
all_orbs=eigensolver.GetOrbitals(len(frozen_occupied_orbitals),as_dim,0)
del eigensolver

integrals = mad.Integrals(box_size, wavelet_order, madness_thresh)
G_elems = integrals.compute_two_body_integrals(all_orbs)
G = tq.quantumchemistry.NBodyTensor(elems=G_elems, ordering="phys").elems
T = integrals.compute_kinetic_integrals(all_orbs)
V = integrals.compute_potential_integrals(all_orbs, mra_pot)
S = integrals.compute_overlap_integrals(all_orbs)
del integrals
h1=T+V
g2=G
c=0.0

"""
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
print(np.shape(h1))
print(np.shape(g2))
print(c)
"""
print("Starting VQE and Orbital-Optimization")
for it in range(iterations):
    print("---------------------------------------------------")
    print("Iteration: " + it.__str__())

    if it == 0:
        mol = tq.Molecule(geometry_angstrom, one_body_integrals=h1, two_body_integrals=g2, nuclear_repulsion=c, name=molecule_name)
        print('done mol')
    else:
        #todo: transfer nb::ndarray objects directly
        h1=np.array(h1_elements).reshape(as_dim,as_dim)
        g2=np.array(g2_elements).reshape(as_dim,as_dim,as_dim,as_dim)
        g2=tq.quantumchemistry.NBodyTensor(g2, ordering="dirac")
        g2=g2.reorder(to="openfermion")
        mol = tq.Molecule(geometry_angstrom, one_body_integrals=h1, two_body_integrals=g2, nuclear_repulsion=c, name=molecule_name) #Important here geometry in Angstrom
        

    
    #VQE
    U = mol.make_ansatz(name="HCB-UpCCGD")
    print('done with U')
    if it == 0:
        opt = OrbOpt_helper.get_best_initial_values(mol)
    else:
        opt = tq.quantumchemistry.optimize_orbitals(molecule=mol, circuit=U, silent=True, use_hcb=True, initial_guess=opt.mo_coeff)
    print('done with opt')
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
    opti.GiveCustomPotential(mra_pot)
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
        opti.plane_plot(i.__str__()+"orb.dat",all_orbs[i],plane="yz",zoom=10.0,datapoints=71,origin=[0.0,0.0,0.0])
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
