import time

import numpy as np
import tequila as tq

import frayedends as fe
from pyscf import fci


world = fe.MadWorld3D(thresh=1e-8)

distance_list = [70 + 0.5 * i for i in range(1)]
Energy_list = []
iteration_energy=[]
lowest_ev=[]
Gradient_list = []

print(world.get_function_defaults())
#def s_orb_Li(x, y, z):
#    r = np.array([x, y, z + 70 / 2])
#    return np.exp(-np.sqrt(np.linalg.norm(r))) / np.sqrt(np.pi)

n_orbitals = 3
n_act_orbitals= 2
n_act_electrons = 2
for distance in distance_list:
    true_start = time.time()
    geometry = "Li 0.0 0.0 " + str(-distance / 2) + "\nH 0.0 0.0 " + str(
        distance / 2
    )

    madpno = fe.MadPNO(world, geometry, units="bohr", n_orbitals=n_orbitals, dft={"localize": "canon"})
    pno_orbitals = madpno.get_orbitals()

    print(world.get_function_defaults())
    world.set_function_defaults()
    print(world.get_function_defaults())
    orbitals = pno_orbitals
    #orbitals[0].type = "active"
    #orb0 = pno_orbitals[0]
    #orb1 = pno_orbitals[1]

    #factory = fe.MRAFunctionFactory3D(world, s_orb_Li)
    #orb2 = factory.get_function()
    #orb2.type = "active"
    #world.line_plot(f"pno3.dat", orb2)

    nuc_repulsion = madpno.get_nuclear_repulsion()
    Vnuc = madpno.get_nuclear_potential()

    integrals = fe.Integrals3D(world)
    #orb2 = integrals.project_out(kernel=[orb0], target=[orb2])[0]
    #orbitals.append(orb0)
    #orbitals.append(orb1)
    #orbitals.append(orb2)
    orbitals = integrals.orthonormalize(orbitals=orbitals)
    for i in range(len(orbitals)):
        world.line_plot(f"pno{i}.dat", orbitals[i])

    frozen_orbitals = []
    for orb in orbitals:
        if orb.type == "frozen_occ":
            frozen_orbitals.append(orb)

    c = nuc_repulsion
    current = 0.0
    for iteration in range(50):
        print("------------------------------------------------------------------------------")
        print(world.get_function_defaults())
        print("nuclear repulsion:", c)
        integrals = fe.Integrals3D(world)
        if iteration == 0:
            G = integrals.compute_two_body_integrals(orbitals, coulomb_lo=0.0001, coulomb_eps=1e-7, truncation_tol=1e-7)
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
            #T = integrals.compute_kinetic_integrals(active_orbitals)
            #V = integrals.compute_potential_integrals(active_orbitals, Vnuc)
            #FC_int= integrals.compute_frozen_core_interaction(frozen_orbitals, active_orbitals, coulomb_lo=0.0001, coulomb_eps=1e-7, truncation_tol=1e-7)
            #print(T)
            #print(V)
            #print(FC_int)
            #print(T+V+FC_int)
            h1 = np.array(h1_elements).reshape(n_act_orbitals, n_act_orbitals)
            g2 = np.array(g2_elements).reshape(n_act_orbitals, n_act_orbitals, n_act_orbitals, n_act_orbitals)
            g2 = tq.quantumchemistry.NBodyTensor(g2, ordering="dirac")
            g2.reorder(to="chem")
            print(h1)
            print(g2.elems)
            params.frozen_core = False
            mol = tq.Molecule(
                parameters=params,
                one_body_integrals=h1,
                two_body_integrals=g2,
                nuclear_repulsion=c,
                n_electrons=2
            )
            # print("h1:",h1)
            # print("g2:",g2.elems)

        U = mol.make_ansatz(name="UpCCGSD")
        H = mol.make_hamiltonian()
        # FCI calculation
        e, fcivec = fci.direct_spin1.kernel(
        h1, g2.elems, n_act_orbitals, n_act_electrons
        )  # Computes the energy and the FCI vector
        rdm1, rdm2 = fci.direct_spin1.make_rdm12(
        fcivec, n_act_orbitals, n_act_electrons
        )  # Computes the 1- and 2- body reduced density matrices
        rdm2 = np.swapaxes(rdm2, 1, 2)
        for i in range(len(rdm1)):
            print("rdm1[",i,",",i,"]:", rdm1[i,i])
        
        #print(h1)
        #print(g2.elems)
        h_m=H.to_matrix()
        #print(h_m)
        #print(np.shape(h_m))
        res = np.linalg.eigh(H.to_matrix())
        
        #print("eigenvector 0:", res[1][:,0])
        #print("eigenvector 1:", res[1][:,1])
        #print("eigenvector 2:", res[1][:,2])
        #print("eigenvector 3:", res[1][:,3])
        #print("eigenvector 4:", res[1][:,4])
        #print("eigenvector 5:", res[1][:,5])
        print("eigenvalues:", res[0])
        print("----------------------------------------------------------------------")
        print("iteration {} energy {:+2.7f}".format(iteration, e+c))
        lowest_ev.append(res[0][0])
        iteration_energy.append(e+c)
        # print("1-RDM:", rdm1)
        # print("2-RDM:", rdm2)

        #if abs(e+c - current) < 1e-6:
        #    break
        current = e+c

        opti = fe.Optimization3D(world, Vnuc, nuc_repulsion)
        orbitals = opti.get_orbitals(
            orbitals=orbitals,
            rdm1=rdm1,
            rdm2=rdm2,
            maxiter=1,
            opt_thresh=0.00000001,
            occ_thresh=0.00000001,
        )
        print("OrbOpt converged?:", opti.converged)
        active_orbitals = []
        for i in range(len(orbitals)):
            if orbitals[i].type == "active":
                active_orbitals.append(orbitals[i])
        c = opti.get_c()
        for i in range(len(orbitals)):
            world.line_plot(f"orb{i}_iter{iteration}.dat", orbitals[i])
        c, h1_elements, g2_elements = opti.get_integrals()
        #c, h1_elements2, g2_elements = opti.get_integrals()
        #for i in range(len(h1_elements2)):
        #        print(h1_elements[i]-h1_elements2[i])
        
    Energy_list.append(current)

    true_end = time.time()
    print("Total time: ", true_end - true_start)

print("distance_list =", distance_list)
print("Energy_list =", Energy_list)
print("Gradient_list =", Gradient_list)
print("Lowest eigenvalue of each iteration =", lowest_ev)
print("Energy of each iteration =", iteration_energy)

fe.cleanup(globals())   