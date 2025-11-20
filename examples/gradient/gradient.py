import time

import numpy as np
import tequila as tq
from pyscf import fci

import frayedends as fe

world = fe.MadWorld3D()

distance_list = [1.417 + 0.02 * i for i in range(1)]
Energy_list = []
Gradient_list = []
n_orbitals = 6
n_electrons = 2  # Number of electrons

for distance in distance_list:
    true_start = time.time()
    geometry = "H 0.0 0.0 0.0\nH 0.0 0.0 " + str(distance)

    molecule = fe.MolecularGeometry(units="bohr")
    molecule.add_atom(0.0, 0.0, 0.0, "H")
    molecule.add_atom(0.0, 0.0, distance, "H")
    madpno = fe.MadPNO(world, geometry, units="bohr", n_orbitals=n_orbitals)
    orbitals = madpno.get_orbitals()

    nuc_repulsion = madpno.get_nuclear_repulsion()
    Vnuc = madpno.get_nuclear_potential()

    integrals = fe.Integrals3D(world)
    orbitals = integrals.orthonormalize(orbitals=orbitals)

    c = nuc_repulsion
    current = 0.0
    print("Distance: ", distance)
    for iteration in range(100):

        integrals = fe.Integrals3D(world)
        G = integrals.compute_two_body_integrals(orbitals, ordering="chem")
        T = integrals.compute_kinetic_integrals(orbitals)
        V = integrals.compute_potential_integrals(orbitals, Vnuc)
        S = integrals.compute_overlap_integrals(orbitals)

        # FCI calculation
        e, fcivec = fci.direct_spin0.kernel(
            T + V, G.elems, n_orbitals, n_electrons
        )  # Computes the energy and the FCI vector
        rdm1, rdm2 = fci.direct_spin0.make_rdm12(
            fcivec, n_orbitals, n_electrons
        )  # Computes the 1- and 2- body reduced density matrices
        rdm2 = np.swapaxes(rdm2, 1, 2)

        print("iteration {} energy {:+2.7f}".format(iteration, e + c))
        if abs(e + c - current) < 1e-6:
            break
        current = e + c

        opti = fe.Optimization3D(world, Vnuc, nuc_repulsion)
        orbitals = opti.get_orbitals(
            orbitals=orbitals, rdm1=rdm1, rdm2=rdm2, opt_thresh=0.001, occ_thresh=0.001
        )
        c = (
            opti.get_c()
        )  # if there are no frozen core electrons, this should always be equal to the nuclear repulsion

        for i in range(len(orbitals)):
            world.line_plot(f"orb{i}.dat", orbitals[i])
    Energy_list.append(e + c)
    # gradient calculation
    part_deriv_V = molecule.compute_nuclear_derivative(world, 1, 2)
    Deriv_tens = integrals.compute_potential_integrals(orbitals, part_deriv_V)
    part_deriv_c = molecule.nuclear_repulsion_derivative(1, 2)

    grad = 0.0
    for i in range(len(orbitals)):
        for j in range(len(orbitals)):
            grad += rdm1[i, j] * Deriv_tens[i, j]

    print("gradient: ", grad + part_deriv_c)
    Gradient_list.append(grad + part_deriv_c)

    true_end = time.time()
    print("Total time: ", true_end - true_start)

print("distance_list=", distance_list)
print("Energy_list=", Energy_list)
print("Gradient_list=", Gradient_list)

fe.cleanup(globals())
