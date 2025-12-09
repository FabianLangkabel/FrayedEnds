import time

import numpy as np
import tequila as tq
from pyscf import fci
import subprocess as sp

import frayedends as fe

world = fe.MadWorld3D(thresh=1e-6)

distance_list = [2.4 + 0.05 * i for i in range(1)]
Energy_list = []
Gradient_list = []

n_orbitals = 5
n_act_orbitals = 4
n_fr_orbitals = n_orbitals - n_act_orbitals
n_act_electrons = 4
miter_oopt=1
for distance in distance_list:
    iteration_e=[]
    print(
            "------------------------------------------------------------------------------"
        )
    print(
            "------------------------------------------------------------------------------"
        )
    print("Distance:", distance)
    print("Maxiter orbital optimization:", miter_oopt)
    true_start = time.time()
    geometry = "H 0.0 0.0 " + str(-distance) + "\nBe 0.0 0.0 0.0" + "\nH 0.0 0.0 " + str(distance)
    
    nwchem_start = time.time()

    basisset = '6-31g'

    nwchem_input = '''
title "molecule"
memory stack 1500 mb heap 100 mb global 1400 mb
charge 0  
geometry units bohr noautosym nocenter''' + '\n' + geometry + '''
end
basis  
* library ''' + basisset + '''
end
scf  
maxiter 200
end   
task scf  '''

    with open("nwchem", "w") as f:
        f.write(nwchem_input)
    programm = sp.call("/Users/timo/miniforge3/envs/fe_test/bin/nwchem nwchem", stdout=open('nwchem.out', 'w'), stderr=open('nwchem_err.log', 'w'), shell = True)

    converter = fe.NWChem_Converter(world)
    converter.read_nwchem_file("nwchem")
    nw_orbitals = converter.get_mos()
    Vnuc = converter.get_Vnuc()
    nuc_repulsion = converter.get_nuclear_repulsion_energy()

    for i in range(len(nw_orbitals)):
        nw_orbitals[i].info=str(i)

    nwchem_end = time.time()
    print("NWChem time:", nwchem_end-nwchem_start)

    world.set_function_defaults()
    print(world.get_function_defaults())

    integrals = fe.Integrals3D(world)
    nw_orbitals = integrals.orthonormalize(orbitals=nw_orbitals)

    G = integrals.compute_two_body_integrals(nw_orbitals)
    T = integrals.compute_kinetic_integrals(nw_orbitals)
    V = integrals.compute_potential_integrals(nw_orbitals, Vnuc)
    mol = tq.Molecule(
        geometry,
        units="bohr",
        one_body_integrals=T + V,
        two_body_integrals=G,
        nuclear_repulsion=nuc_repulsion,
    )
    print(mol.active_space)
    c, h1, g2 = mol.get_integrals(ordering="chem")

    fci_start = time.time()
    # FCI calculation
    e, fcivec = fci.direct_spin1.kernel(
        h1, g2.elems, np.shape(h1)[0], n_act_electrons
    )  # Computes the energy and the FCI vector
    rdm1, rdm2 = fci.direct_spin1.make_rdm12(
        fcivec, np.shape(h1)[0], n_act_electrons
    )  # Computes the 1- and 2- body reduced density matrices
    rdm2 = np.swapaxes(rdm2, 1, 2)
    print("Initial FCI energy {:+2.7f}".format(e + c))
    fci_end = time.time()
    print("fci time:", fci_end-fci_start)

    occ_n=[]
    for i in range(len(rdm1)):
        occ_n.append(rdm1[i, i])
    print(occ_n)
    occ_n_sorted=sorted(occ_n, reverse=True)

    orbitals=[]
    frozen_orbitals = []
    for i in range(n_orbitals):
        if i < n_fr_orbitals:
            orbitals.append(nw_orbitals[i])
            orbitals[i].type="frozen_occ"
            frozen_orbitals.append(orbitals[i])
        else:
            index=occ_n.index(occ_n_sorted[i - n_fr_orbitals])
            orbitals.append(nw_orbitals[index + n_fr_orbitals])
            orbitals[i].type="active"
    
    for i in range(len(orbitals)):
        print(orbitals[i].info)
    
    #Start of the main algorithm
    current = e + c
    for iteration in range(401):
        if iteration % 5 == 0:
            for i in range(len(orbitals)):
                world.line_plot(f"it_{iteration:02d}_orb_{i:01d}_{int(distance*100):03d}.dat", orbitals[i])
            if iteration == 0:
                for i in range(len(orbitals)):
                    world.plane_plot(f"nwchem_{i}_{int(distance*100)}", orbitals[i], plane="xy", zoom=5.0, datapoints=120)
                    world.plane_plot(f"nwchem_{i}_{int(distance*100)}", orbitals[i], plane="xz", zoom=5.0, datapoints=120)
                    world.plane_plot(f"nwchem_{i}_{int(distance*100)}", orbitals[i], plane="yz", zoom=5.0, datapoints=120)
            

        opti_start = time.time()
        opti = fe.Optimization3D(world, Vnuc, nuc_repulsion)
        orbitals = opti.get_orbitals(
            orbitals=orbitals,
            rdm1=rdm1,
            rdm2=rdm2,
            maxiter=1,
            opt_thresh=0.0001,
            occ_thresh=0.0001,
        )
        print("Converged?:", opti.converged)
        opti_end = time.time()
        print("orb opt time:", opti_end-opti_start)
        active_orbitals = []
        for i in range(len(orbitals)):
            if orbitals[i].type == "active":
                active_orbitals.append(orbitals[i])
        c = opti.get_c()

        print( 
            "------------------------------------------------------------------------------"
        )
        integrals = fe.Integrals3D(world)
        T = integrals.compute_kinetic_integrals(active_orbitals)
        V = integrals.compute_potential_integrals(active_orbitals, Vnuc)
        G = integrals.compute_two_body_integrals(active_orbitals, ordering="chem")
        FC_int= integrals.compute_frozen_core_interaction(frozen_orbitals, active_orbitals)
        # print(FC_int)
        # print(T+V+FC_int)
        h1 = T + V + FC_int
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
        for i in range(np.shape(rdm1)[0]):
            print("rdm1[", i, ",", i, "]:", rdm1[i, i])
        print("iteration {} energy {:+2.7f}".format(iteration, e + c))

        
        current = e + c
        iteration_e.append(current)
        

        
    Energy_list.append(current)
    
    
    for i in range(len(orbitals)):
        world.plane_plot(f"orbital_{i}_{int(distance*100)}", orbitals[i],plane="xy",zoom=5.0,datapoints=120)
        world.plane_plot(f"orbital_{i}_{int(distance*100)}", orbitals[i],plane="xz",zoom=5.0,datapoints=120)
        world.plane_plot(f"orbital_{i}_{int(distance*100)}", orbitals[i],plane="yz",zoom=5.0,datapoints=120)
    


    molecule = fe.MolecularGeometry(geometry=geometry, units="bohr")
    part_deriv_V_0 = molecule.compute_nuclear_derivative(world, 0, 2)
    part_deriv_V_2 = molecule.compute_nuclear_derivative(world, 2, 2)
    Deriv_tens = integrals.compute_potential_integrals(orbitals, part_deriv_V_0)
    Deriv_tens2 = integrals.compute_potential_integrals(orbitals, part_deriv_V_2)
    part_deriv_c = molecule.nuclear_repulsion_derivative(0, 2)
    grad = 2*Deriv_tens[0,0]  #Be core orbital contribution
    for i in range(len(active_orbitals)):
        for j in range(len(active_orbitals)):
            grad += rdm1[i, j] * Deriv_tens[i+1, j+1]
    print("gradient0: ", grad + part_deriv_c)

    part_deriv_c2 = molecule.nuclear_repulsion_derivative(2, 2)
    grad2 = 2*Deriv_tens2[0,0]  #Be core orbital contribution
    for i in range(len(active_orbitals)):
        for j in range(len(active_orbitals)):
            grad2 += rdm1[i, j] * Deriv_tens2[i+1, j+1]
    print("gradient2: ", grad2 + part_deriv_c2)
    Gradient_list.append(grad2 + part_deriv_c2 - grad - part_deriv_c)

    true_end = time.time()
    print("Total time: ", true_end - true_start)

print("distance_list =", distance_list)
print("Energy_list =", Energy_list)
print("Gradient_list =", Gradient_list)
print("iteration_energies =", iteration_e)

fe.cleanup(globals())
