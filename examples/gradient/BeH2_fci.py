import time

import numpy as np
import tequila as tq
from pyscf import fci

import frayedends as fe

world = fe.MadWorld3D(thresh=1e-6)

distance_list = [1.5 + 1.5 * i for i in range(2)]
Energy_list = []
Gradient_list = []

n_orbitals = 5
n_act_orbitals = 4
n_act_electrons = 4
miter=1
for distance in distance_list:
    print(
            "------------------------------------------------------------------------------"
        )
    print(
            "------------------------------------------------------------------------------"
        )
    print("Distance:", distance)
    print("Maxiter orbital optimization:", miter)
    true_start = time.time()
    geometry = "H 0.0 0.0 " + str(-distance) + "\nBe 0.0 0.0 0.0" + "\nH 0.0 0.0 " + str(distance)
    molgeom = fe.MolecularGeometry(geometry=geometry, units="bohr")
    
    pno_start = time.time()
    madpno = fe.MadPNO(
        world, geometry, units="bohr", n_orbitals=n_orbitals, dft={"localize": "boys"}
    )
    orbitals = madpno.get_orbitals()
    
    pno_end = time.time()
    print("Pno time:", pno_end-pno_start)

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
    for iteration in range(5):
        active_orbitals = []
        for i in range(len(orbitals)):
            if orbitals[i].type == "active":
                active_orbitals.append(orbitals[i])

        print( 
            "------------------------------------------------------------------------------"
        )
        integrals = fe.Integrals3D(world)
        if iteration == 0:
            G = integrals.compute_two_body_integrals(orbitals)
            T = integrals.compute_kinetic_integrals(orbitals)
            V = integrals.compute_potential_integrals(orbitals, Vnuc)
            mol = molgeom.to_tq_mol(
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
            params.frozen_core = False
            mol = tq.Molecule(parameters=params, one_body_integrals=h1, two_body_integrals=g2, nuclear_repulsion=c, n_electrons=n_act_electrons)

        H = mol.make_hamiltonian()
        res = np.linalg.eigvalsh(H.to_matrix())
        print("Diagonalization energy:", res[0])
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
        current = e + c

        opti_start = time.time()
        opti = fe.Optimization3D(world, Vnuc, nuc_repulsion)
        orbitals = opti.get_orbitals(
            orbitals=orbitals,
            rdm1=rdm1,
            rdm2=rdm2,
            maxiter=miter,
            opt_thresh=0.0001,
            occ_thresh=0.0001,
        )   
        opti_end = time.time()
        print("orb opt time:", opti_end-opti_start)
        active_orbitals = []
        for i in range(len(orbitals)):
            if orbitals[i].type == "active":
                active_orbitals.append(orbitals[i])
        c = opti.get_c()

        
    Energy_list.append(current)
    
    for i in range(len(orbitals)):
        world.cube_plot(f"d_{distance}_orbital_{i}", orbitals[i], molgeom, zoom=5.0, datapoints=81)
        world.line_plot(f"d_{distance}_orbital_{i}.dat", orbitals[i])

    part_deriv_V_0 = molgeom.molecular_potential_derivative(world, 0, 2)
    part_deriv_V_2 = molgeom.molecular_potential_derivative(world, 2, 2)
    Deriv_tens = integrals.compute_potential_integrals(orbitals, part_deriv_V_0)
    Deriv_tens2 = integrals.compute_potential_integrals(orbitals, part_deriv_V_2)
    part_deriv_c = molgeom.nuclear_repulsion_derivative(0, 2)
    grad = 2*Deriv_tens[0,0]  #Be core orbital contribution
    for i in range(len(active_orbitals)):
        for j in range(len(active_orbitals)):
            grad += rdm1[i, j] * Deriv_tens[i+1, j+1]
    print("gradient0: ", grad + part_deriv_c)

    part_deriv_c2 = molgeom.nuclear_repulsion_derivative(2, 2)
    grad2 = 2*Deriv_tens2[0,0]  #Be core orbital contribution
    for i in range(len(active_orbitals)):
        for j in range(len(active_orbitals)):
            grad2 += rdm1[i, j] * Deriv_tens2[i+1, j+1]
    print("gradient2: ", grad2 + part_deriv_c2)
    Gradient_list.append(grad2 + part_deriv_c2 - grad - part_deriv_c)
    print(molgeom.compute_energy_gradient(world, orbitals, rdm1, nocc=2))

    true_end = time.time()
    print("Total time: ", true_end - true_start)

print("distance_list =", distance_list)
print("Energy_list =", Energy_list)
print("Gradient_list =", Gradient_list)

fe.cleanup(globals())
