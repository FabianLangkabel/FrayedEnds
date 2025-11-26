import time

import numpy as np
import tequila as tq
from pyscf import fci

import frayedends as fe

world = fe.MadWorld3D(thresh=1e-4)

distance_list = [0.5 + 0.1 * i for i in range(5)]
Energy_list = []
Gradient_list = []

n_orbitals = 3
n_act_orbitals = 2
n_act_electrons = 2
for distance in distance_list:
    print(
            "------------------------------------------------------------------------------"
        )
    print(
            "------------------------------------------------------------------------------"
        )
    [print("Distance:", distance)]
    true_start = time.time()
    geometry = "Li 0.0 0.0 " + str(-distance / 2) + "\nH 0.0 0.0 " + str(distance / 2)

    madpno = fe.MadPNO(
        world, geometry, units="bohr", n_orbitals=n_orbitals, dft={"localize": "canon"}
    )
    orbitals = madpno.get_orbitals()

    world.set_function_defaults()
    print(world.get_function_defaults())

    nuc_repulsion = madpno.get_nuclear_repulsion()
    Vnuc = madpno.get_nuclear_potential()

    integrals = fe.Integrals3D(world)
    orbitals = integrals.orthonormalize(orbitals=orbitals)
    
    frozen_orbitals = []
    for orb in orbitals:
        if orb.type == "frozen_occ":
            frozen_orbitals.append(orb)

    c = nuc_repulsion
    current = 0.0
    for iteration in range(10):
        print(
            "------------------------------------------------------------------------------"
        )
        integrals = fe.Integrals3D(world)
        if iteration == 0:
            G = integrals.compute_two_body_integrals(orbitals)
            T = integrals.compute_kinetic_integrals(orbitals)
            V = integrals.compute_potential_integrals(orbitals, Vnuc)
            mol = tq.Molecule(
                geometry,
                units="bohr",
                one_body_integrals=T + V,
                two_body_integrals=G,
                nuclear_repulsion=c,
            )
            params = mol.parameters
            c, h1, g2 = mol.get_integrals(ordering="chem")
        else:
            T = integrals.compute_kinetic_integrals(active_orbitals)
            V = integrals.compute_potential_integrals(active_orbitals, Vnuc)
            G = integrals.compute_two_body_integrals(active_orbitals, ordering="chem")
            FC_int= integrals.compute_frozen_core_interaction(frozen_orbitals, active_orbitals)
            # print(FC_int)
            # print(T+V+FC_int)
            h1 = T + V + FC_int
            g2 = G

        # FCI calculation
        e, fcivec = fci.direct_spin1.kernel(
            h1, g2.elems, n_act_orbitals, n_act_electrons
        )  # Computes the energy and the FCI vector
        rdm1, rdm2 = fci.direct_spin1.make_rdm12(
            fcivec, n_act_orbitals, n_act_electrons
        )  # Computes the 1- and 2- body reduced density matrices
        rdm2 = np.swapaxes(rdm2, 1, 2)
        #for i in range(len(rdm1)):
        #    print("rdm1[", i, ",", i, "]:", rdm1[i, i])

        print("iteration {} energy {:+2.7f}".format(iteration, e + c))

        if abs(e+c - current) < 1e-5:
            break
        current = e + c

        opti = fe.Optimization3D(world, Vnuc, nuc_repulsion)
        orbitals = opti.get_orbitals(
            orbitals=orbitals,
            rdm1=rdm1,
            rdm2=rdm2,
            maxiter=1,
            opt_thresh=0.0001,
            occ_thresh=0.0001,
        )
        active_orbitals = []
        for i in range(len(orbitals)):
            if orbitals[i].type == "active":
                active_orbitals.append(orbitals[i])
        c = opti.get_c()
    
    Energy_list.append(current)

    molecule = fe.MolecularGeometry(geometry= geometry, units="bohr")
    part_deriv_V = molecule.compute_nuclear_derivative(world, 1, 2)
    Deriv_tens = integrals.compute_potential_integrals(orbitals, part_deriv_V)
    part_deriv_c = molecule.nuclear_repulsion_derivative(1, 2)
    grad = 2*Deriv_tens[0,0]  #Li core orbital contribution
    for i in range(len(active_orbitals)):
        for j in range(len(active_orbitals)):
            grad += rdm1[i, j] * Deriv_tens[i+1, j+1]
    print("gradient: ", grad + part_deriv_c)
    Gradient_list.append(grad + part_deriv_c)

    true_end = time.time()
    print("Total time: ", true_end - true_start)

print("distance_list =", distance_list)
print("Energy_list =", Energy_list)
print("Gradient_list =", Gradient_list)

fe.cleanup(globals())