import numpy as np
import madpy as mad
import tequila as tq
from time import time
from pyscf import fci

true_start=time()
# initialize the PNO interface
dummy_molecule = "He 0.0 0.0 0.0" 

def potential(x,y,z):
    a = -5.0
    r = np.array([x,y,z])
    return a * np.exp(- 0.5 * np.linalg.norm(r)**2)

factory = mad.MRAFunctionFactory(potential)
param = factory.madness_parameters
mra_pot = factory.GetFunction()
del factory

eigen = mad.Eigensolver(mra_pot, parameters=param)
orbitals = eigen.get_orbitals(0,2,0,n_states=5) # frozen occupied, active, frozen virtual dimensions
del eigen


plt=mad.Plotter()
plt.line_plot("potential.dat", mra_pot)
for i in range(len(orbitals)):
    plt.line_plot(f"es_orb{i}.dat",orbitals[i])
del plt

c = 0.0

for iteration in range(6):
    integrals = mad.Integrals(param)
    G = integrals.compute_two_body_integrals(orbitals, ordering="chem")
    T = integrals.compute_kinetic_integrals(orbitals)
    V = integrals.compute_potential_integrals(orbitals, mra_pot)
    S = integrals.compute_overlap_integrals(orbitals)
    del integrals

    mol = tq.Molecule(dummy_molecule, one_body_integrals=T+V, two_body_integrals=G, nuclear_repulsion=c)
    e, fcivec = fci.direct_spin0.kernel(T+V, G.elems, mol.n_orbitals, mol.n_electrons)
    rdm1, rdm2 = fci.direct_spin0.make_rdm12(fcivec, mol.n_orbitals, mol.n_electrons)
    rdm2 = np.swapaxes(rdm2, 1, 2)

    rdm1_list = rdm1.reshape(-1).tolist()
    rdm2_list = rdm2.reshape(-1).tolist()


    print("iteration {} FCI energy {:+2.8f}".format(iteration, e))
    
    opti = mad.Optimization(mra_pot, nuc_repulsion=0.0, parameters=param)
    orbitals = opti.get_orbitals(orbitals=orbitals, rdm1=rdm1, rdm2=rdm2, opt_thresh=0.001, occ_thresh=0.001)
    c,h_el,g_el = opti.get_integrals()
    del opti

    

    plt=mad.Plotter()
    for i in range(len(orbitals)):
        plt.line_plot(f"orb{i}.dat",orbitals[i])
    del plt

true_end=time()
print("Total time: ", true_end-true_start)




