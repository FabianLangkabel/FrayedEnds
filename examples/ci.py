from time import time

import numpy
import tequila as tq

import madpy

method = "fci"  # "cisd"

true_start = time()
# initialize the PNO interface
geom = "H 0.0 0.0 0.0\nH 0.0 0.0 3.5\nH 0.0 0.0 7.0\nH 0.0 0.0 10.5"  # geometry in Angstrom

world = madpy.MadWorld3D()
madpno = madpy.MadPNO(world, geom, n_orbitals=4)
orbitals = madpno.get_orbitals()

nuc_repulsion = madpno.get_nuclear_repulsion()
Vnuc = madpno.get_nuclear_potential()

for i in range(len(orbitals)):
    world.line_plot(f"pnoorb{i}.dat", orbitals[i])

integrals = madpy.Integrals3D(world)
orbitals = integrals.orthonormalize(orbitals=orbitals)

c = nuc_repulsion
for iteration in range(6):

    for i in range(len(orbitals)):
        world.line_plot(f"orbital_{i}_iteration_{iteration}.dat", orbitals[i])

    integrals = madpy.Integrals3D(world)
    G = integrals.compute_two_body_integrals(orbitals).elems
    T = integrals.compute_kinetic_integrals(orbitals)
    V = integrals.compute_potential_integrals(orbitals, Vnuc)
    S = integrals.compute_overlap_integrals(orbitals)

    mol = madpy.PySCFInterface(
        geometry=geom, one_body_integrals=T + V, two_body_integrals=G, constant_term=c
    )
    rdm1, rdm2, energy = mol.compute_rdms(method=method, return_energy=True)
    print("iteration {} energy {:+2.5f}".format(iteration, energy))

    opti = madpy.Optimization3D(world, Vnuc, nuc_repulsion)
    new_orbitals = opti.get_orbitals(
        orbitals=orbitals, rdm1=rdm1, rdm2=rdm2, opt_thresh=0.001, occ_thresh=0.001
    )

    integrals = madpy.Integrals3D(world)
    S = integrals.compute_overlap_integrals(orbitals, new_orbitals)
    print("overlap new and old")
    print(S)
    # permute if necessary to avoid breaking the spa ansatz
    # virtuals are sometimes flipped
    xorbitals = [x for x in orbitals]
    for i in range(S.shape[0]):
        j = numpy.argmax(S[i])
        # currently not implented
        # we would simply need to scale the function by -1.0
        # this will only affect spa if we reuse the parameters
        if S[i][j] < 0.0:
            print("\n\n--> phase detected <--\n\n")
        orbitals[i] = new_orbitals[j]
    S = integrals.compute_overlap_integrals(xorbitals, orbitals)
    print(S)

true_end = time()
print("Total time: ", true_end - true_start)

madpy.cleanup(globals())
