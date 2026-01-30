import time

import numpy as np
import tequila as tq
from pyscf import fci

import frayedends as fe

world = fe.MadWorld3D(thresh=1e-6)

distance_list = [1.0 + 0.01 * i for i in range(250,300)]
Energy_list = []
Gradient_list = []
no_convergence = []
not_symmetric_grad = []

n_orbitals = 5
n_act_orbitals = 5
n_act_electrons = 6
miter_oopt=1
for distance in distance_list:
    print(
            "------------------------------------------------------------------------------"
        )
    print(
            "------------------------------------------------------------------------------"
        )
    print("Distance:", distance)
    print("Maxiter orb opt:", miter_oopt)
    true_start = time.time()
    geometry = "H 0.0 0.0 " + str(-distance) + "\nBe 0.0 0.0 0.0" + "\nH 0.0 0.0 " + str(distance)
    
    pno_start = time.time()
    madpno = fe.MadPNO(
        world, geometry, units="bohr", n_orbitals=n_orbitals
    )
    orbitals = madpno.get_orbitals()
    for i in range(len(orbitals)):
        if orbitals[i].type == "frozen_occ":
            orbitals[i].type = "active"
    pno_end = time.time()
    print("Pno time:", pno_end-pno_start)

    world.set_function_defaults()
    print(world.get_function_defaults())

    nuc_repulsion = madpno.get_nuclear_repulsion()
    Vnuc = madpno.get_nuclear_potential()

    integrals = fe.Integrals3D(world)
    orbitals = integrals.orthonormalize(orbitals=orbitals)
    

    c = nuc_repulsion
    current = 0.0
    for iteration in range(31):   
        print( 
            "------------------------------------------------------------------------------"
        )
        integrals = fe.Integrals3D(world)
        G = integrals.compute_two_body_integrals(orbitals,ordering="chem")
        T = integrals.compute_kinetic_integrals(orbitals)
        V = integrals.compute_potential_integrals(orbitals, Vnuc)
        h1 = T + V
        g2 = G
        

        fci_start = time.time()
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
        fci_end = time.time()
        print("fci time:", fci_end-fci_start)
        print(rdm1)
        print("iteration {} energy {:+2.7f}".format(iteration, e + c))

        if abs(current - (e + c)) < 1e-6:
            print("FCI energy converged.")
            break
        elif iteration == 30:
            print("FCI energy did not converge.")
            no_convergence.append(distance)
        current = e + c

        opti_start = time.time()
        opti = fe.Optimization3D(world, Vnuc, nuc_repulsion)
        orbitals = opti.get_orbitals(
            orbitals=orbitals,
            rdm1=rdm1,
            rdm2=rdm2,
            maxiter=miter_oopt,
            opt_thresh=0.0001,
            occ_thresh=0.0001,
        )
        print("Converged?:", opti.converged)
        opti_end = time.time()
        print("orb opt time:", opti_end-opti_start)
        c = opti.get_c()

        
    Energy_list.append(current)
    
    
    molecule = fe.MolecularGeometry(geometry=geometry, units="bohr")
    part_deriv_V_0 = molecule.molecular_potential_derivative(world, 0, 2)
    part_deriv_V_2 = molecule.molecular_potential_derivative(world, 2, 2)
    Deriv_tens = integrals.compute_potential_integrals(orbitals, part_deriv_V_0)
    Deriv_tens2 = integrals.compute_potential_integrals(orbitals, part_deriv_V_2)
    part_deriv_c = molecule.nuclear_repulsion_derivative(0, 2)
    grad = 0.0
    for i in range(len(orbitals)):
        for j in range(len(orbitals)):
            grad += rdm1[i, j] * Deriv_tens[i, j]
    print("gradient0: ", grad + part_deriv_c)

    part_deriv_c2 = molecule.nuclear_repulsion_derivative(2, 2)

    grad2 = 0.0
    for i in range(len(orbitals)):
        for j in range(len(orbitals)):
            grad2 += rdm1[i, j] * Deriv_tens2[i, j]
    print("gradient2: ", grad2 + part_deriv_c2)
    Gradient_list.append(grad2 + part_deriv_c2 - grad - part_deriv_c)
    if abs(grad2 + part_deriv_c2 + grad + part_deriv_c) > 1e-6:
        not_symmetric_grad.append(distance)
    true_end = time.time()
    print("Total time: ", true_end - true_start)

print("distance_list =", distance_list)
print("Energy_list =", Energy_list)
print("Gradient_list =", Gradient_list)
print("No convergence at distances:", no_convergence)
print("Not symmetric gradient at distances:", not_symmetric_grad)

fe.cleanup(globals())
