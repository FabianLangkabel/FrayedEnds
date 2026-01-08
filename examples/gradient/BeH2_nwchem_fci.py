import time

import numpy as np
import tequila as tq
from pyscf import fci
import subprocess as sp

import frayedends as fe

world = fe.MadWorld3D(thresh=1e-8)

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
    molgeom = fe.MolecularGeometry(geometry=geometry, units="bohr")

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


    nwchem_end = time.time()
    print("NWChem time:", nwchem_end-nwchem_start)

    world.set_function_defaults()
    print(world.get_function_defaults())

    integrals = fe.Integrals3D(world)
    nw_orbitals = integrals.orthonormalize(orbitals=nw_orbitals)

    G = integrals.compute_two_body_integrals(nw_orbitals, truncation_tol=1e-8, coulomb_lo=0.00001, coulomb_eps=1e-8)
    T = integrals.compute_kinetic_integrals(nw_orbitals)
    V = integrals.compute_potential_integrals(nw_orbitals, Vnuc)
    mol = tq.Molecule(
        geometry,
        units="bohr",
        one_body_integrals=T + V,
        two_body_integrals=G,
        nuclear_repulsion=nuc_repulsion,
    )
    params0 = mol.parameters
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

    ac_orbitals = []
    for i in range(n_fr_orbitals,len(nw_orbitals)):
        ac_orbitals.append(nw_orbitals[i])

    nat_orbs,occ_n=integrals.transform_to_natural_orbitals(ac_orbitals, rdm1)
    world.cube_plot(f"nwchem_raw{0}", nw_orbitals[0], molgeom, zoom=5.0)
    print("Natural occupation numbers:", occ_n)
    for i in range(len(nat_orbs)):
            world.cube_plot(f"nwchem_nat{i+1}", nat_orbs[i], molgeom, zoom=5.0)
    
    orbitals = []
    for i in range(n_orbitals):
        if i < n_fr_orbitals:
            orbitals.append(nw_orbitals[i])
            orbitals[i].type = "frozen_occ"
        else:
            orbitals.append(nat_orbs[i - n_fr_orbitals])
            orbitals[i].type = "active"
    frozen_orbitals=[]
    active_orbitals=[]
    for i in range(len(orbitals)):
        if orbitals[i].type == "frozen_occ":
            frozen_orbitals.append(orbitals[i])
        else:
            active_orbitals.append(orbitals[i])

    G = integrals.compute_two_body_integrals(orbitals, truncation_tol=1e-8, coulomb_lo=0.00001, coulomb_eps=1e-8)
    T = integrals.compute_kinetic_integrals(orbitals)
    V = integrals.compute_potential_integrals(orbitals, Vnuc)
    mol = tq.Molecule(
        geometry,
        units="bohr",
        one_body_integrals=T + V,
        two_body_integrals=G,
        nuclear_repulsion=nuc_repulsion,
    )
    params0 = mol.parameters
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
    print("Second FCI energy {:+2.7f}".format(e + c))
    fci_end = time.time()
    

    #Start of the main algorithm
    current = e + c
    for iteration in range(401):
        opti_start = time.time()
        opti = fe.Optimization3D(world, Vnuc, nuc_repulsion, BSH_lo=0.00001, BSH_eps=1e-8, coulomb_lo=0.00001, coulomb_eps=1e-8, truncation_tol=1e-8)
        orbitals = opti.get_orbitals(
            orbitals=orbitals,
            rdm1=rdm1,
            rdm2=rdm2,
            maxiter=1,
            opt_thresh=0.00001,
            occ_thresh=0.00001,
            redirect_filename=f"madopt_it{iteration}.log"
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
        G = integrals.compute_two_body_integrals(active_orbitals, ordering="chem", truncation_tol=1e-8, coulomb_lo=0.00001, coulomb_eps=1e-8)
        FC_int= integrals.compute_frozen_core_interaction(frozen_orbitals, active_orbitals)
        # print(FC_int)
        # print(T+V+FC_int)
        h1 = T + V + FC_int
        g2 = G

        params0.frozen_core = False
        mol = tq.Molecule(
            parameters=params0,
            one_body_integrals=h1,
            two_body_integrals=g2,
            nuclear_repulsion=c,
            n_electrons=4,
        )
        H=mol.make_hamiltonian()
        res = np.linalg.eigvalsh(H.to_matrix())
        print("Lowest eigenvalue of Hamiltonian:", res[0])

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
        if iteration in [5,20,250,400]:
            nat_orbs1, occ_n = integrals.transform_to_natural_orbitals(active_orbitals, rdm1)
            print("Natural occupation numbers:", occ_n)
            world.line_plot(f"it_{iteration:02d}_orb_{(0):01d}_{int(distance*100):03d}.dat", orbitals[0])
            for i in range(len(nat_orbs1)):
                world.cube_plot(f"nat1_orb{(i+1):01d}_it{iteration:03d}", nat_orbs1[i], molgeom, zoom=5.0)
                world.line_plot(f"it_{iteration:02d}_orb_{(i+1):01d}_{int(distance*100):03d}.dat", nat_orbs1[i])
        
        current = e + c
        iteration_e.append(current)
        

        
    Energy_list.append(current)
    
    
    

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
