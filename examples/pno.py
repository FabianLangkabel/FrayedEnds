import madpy
import tequila as tq
from time import time

true_start=time()
# initialize the PNO interface
geom = "H 0.0 0.0 -1.25\nH 0.0 0.0 1.25" # geometry in Angstrom
madpno = madpy.MadPNO(geom, n_orbitals=2)
orbitals = madpno.get_orbitals()
print(madpy.get_function_info(orbitals))

param = madpno.madness_parameters
nuc_repulsion= madpno.get_nuclear_repulsion()
Vnuc = madpno.get_nuclear_potential()
del madpno
integrals = madpy.Integrals(param)
orbitals = integrals.orthonormalize(orbitals=orbitals)
del integrals

plt=madpy.Plotter()
for i in range(len(orbitals)):
    plt.line_plot(f"pnoorb{i}.dat",orbitals[i])
del plt


c=nuc_repulsion
for iteration in range(6):

    integrals = madpy.Integrals(param)
    G = integrals.compute_two_body_integrals(orbitals).elems
    T = integrals.compute_kinetic_integrals(orbitals)
    V = integrals.compute_potential_integrals(orbitals, Vnuc)
    S = integrals.compute_overlap_integrals(orbitals)
    print(S)
    del integrals

    mol = tq.Molecule(geom, one_body_integrals=T+V, two_body_integrals=G, nuclear_repulsion=c)
    
    U = mol.make_ansatz(name="UpCCGD")
    H = mol.make_hamiltonian()
    E = tq.ExpectationValue(H=H, U=U)
    result = tq.minimize(E, silent=True)
    rdm1, rdm2 = mol.compute_rdms(U, variables=result.variables)

    print("iteration {} energy {:+2.5f}".format(iteration, result.energy))
    
    opti = madpy.Optimization(Vnuc, nuc_repulsion, parameters=param)
    orbitals = opti.get_orbitals(orbitals=orbitals, rdm1=rdm1, rdm2=rdm2, opt_thresh=0.001, occ_thresh=0.001)
    c = opti.get_c() #if there are no frozen core electrons, this should always be equal to the nuclear repulsion
    print(orbitals)
    del opti

    plt=madpy.Plotter()
    for i in range(len(orbitals)):
        plt.line_plot(f"orb{i}.dat",orbitals[i])
    del plt

true_end=time()
print("Total time: ", true_end-true_start)




