import numpy
import madpy
import tequila as tq
from time import time

true_start = time()
# initialize the PNO interface
geom = "H 0.0 0.0 0.0\nH 0.0 0.0 3.5\nH 0.0 0.0 7.0\nH 0.0 0.0 10.5"  # geometry in Angstrom
madpno = madpy.MadPNO(geom, n_orbitals=4)
orbitals = madpno.get_orbitals()
edges = madpno.get_spa_edges()

param = madpno.madness_parameters
nuc_repulsion = madpno.get_nuclear_repulsion()
Vnuc = madpno.get_nuclear_potential()
del madpno

plt = madpy.Plotter()
for i in range(len(orbitals)):
    plt.line_plot(f"pnoorb{i}.dat", orbitals[i])
del plt

integrals = madpy.Integrals(param)
orbitals = integrals.orthonormalize(orbitals=orbitals)
del integrals
c = nuc_repulsion
for iteration in range(6):
    
    plt = madpy.Plotter()
    for i in range(len(orbitals)):
        plt.line_plot(f"orbital_{i}_iteration_{iteration}.dat", orbitals[i])
    del plt

    integrals = madpy.Integrals(param)
    G = integrals.compute_two_body_integrals(orbitals).elems
    T = integrals.compute_kinetic_integrals(orbitals)
    V = integrals.compute_potential_integrals(orbitals, Vnuc)
    S = integrals.compute_overlap_integrals(orbitals)
    del integrals

    mol = tq.Molecule(geom, one_body_integrals=T + V, two_body_integrals=G, nuclear_repulsion=c)
    U = mol.make_ansatz(name="SPA", edges=edges)
    H = mol.make_hamiltonian()
    E = tq.ExpectationValue(H=H, U=U)
    result = tq.minimize(E, silent=True)
    rdm1, rdm2 = mol.compute_rdms(U, variables=result.variables)

    print(c)
    print("iteration {} energy {:+2.5f}".format(iteration, result.energy))

    opti = madpy.Optimization(Vnuc, nuc_repulsion, parameters=param)
    new_orbitals = opti.get_orbitals(orbitals=orbitals, rdm1=rdm1, rdm2=rdm2, opt_thresh=0.001, occ_thresh=0.001)
    del opti

    integrals = madpy.Integrals(param)
    S = integrals.compute_overlap_integrals(orbitals, new_orbitals)
    print("overlap new and old")
    print(S)
    # permute if necessary to avoid breaking the spa ansatz
    # virtuals are sometimes flipped
    xorbitals=[x for x in orbitals]
    for i in range(S.shape[0]):
        j = numpy.argmax(S[i])
        # currently not implented
        # we would simply need to scale the function by -1.0
        # this will only affect spa if we reuse the parameters
        if S[i][j] < 0.0: print("\n\n--> phase detected <--\n\n")
        orbitals[i] = new_orbitals[j]
    S = integrals.compute_overlap_integrals(xorbitals, orbitals)
    print(S)
    del integrals

true_end = time()
print("Total time: ", true_end - true_start)




