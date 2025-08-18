import madpy
import numpy as np
import tequila as tq
import time


world = madpy.MadWorld()

distance_list = [0.74+0.005 * i for i in range(1)]
Energy_list=[]
Gradient_list=[]

for distance in distance_list:
    true_start = time.time()
    geometry = "H 0.0 0.0 0.0\nH 0.0 0.0 "+ str(distance)

    molecule=madpy.MadMolecule()
    molecule.add_atom(0.0,0.0,0.0,"H")
    molecule.add_atom(0.0,0.0,distance,"H")
    pno_start = time.time()
    madpno = madpy.MadPNO(world, geometry, n_orbitals=2)
    orbitals = madpno.get_orbitals()

    nuc_repulsion = madpno.get_nuclear_repulsion()
    Vnuc = madpno.get_nuclear_potential()

    integrals = madpy.Integrals(world)
    orbitals = integrals.orthonormalize(orbitals=orbitals)


    c = nuc_repulsion
    current=0.0
    for iteration in range(6):

        integrals = madpy.Integrals(world)
        G = integrals.compute_two_body_integrals(orbitals).elems
        T = integrals.compute_kinetic_integrals(orbitals)
        V = integrals.compute_potential_integrals(orbitals, Vnuc)
        S = integrals.compute_overlap_integrals(orbitals)

        mol = tq.Molecule(geometry, one_body_integrals=T + V, two_body_integrals=G, nuclear_repulsion=c)

        U = mol.make_ansatz(name="UpCCGD")
        H = mol.make_hamiltonian()
        E = tq.ExpectationValue(H=H, U=U)
        result = tq.minimize(E, silent=True)
        rdm1, rdm2 = mol.compute_rdms(U, variables=result.variables)

        print("iteration {} energy {:+2.7f}".format(iteration, result.energy))
        if abs(result.energy-current)<1e-6:
            break
        current = result.energy
        

        opti = madpy.Optimization(world, Vnuc, nuc_repulsion)
        orbitals = opti.get_orbitals(orbitals=orbitals, rdm1=rdm1, rdm2=rdm2, opt_thresh=0.001, occ_thresh=0.001)
        c = opti.get_c() # if there are no frozen core electrons, this should always be equal to the nuclear repulsion

        for i in range(len(orbitals)):
            world.line_plot(f"orb{i}.dat", orbitals[i])
    Energy_list.append(result.energy)
    #gradient calculation
    part_deriv_V = molecule.compute_nuclear_derivative(world, 1, 2)
    Deriv_tens = integrals.compute_potential_integrals(orbitals, part_deriv_V)
    part_deriv_c = molecule.nuclear_repulsion_derivative(1, 2)

    grad=0.0
    for i in range(len(orbitals)):
        for j in range(len(orbitals)):
            grad += rdm1[i, j] * Deriv_tens[i, j]

    print("gradient: ", grad+part_deriv_c)
    Gradient_list.append(grad+part_deriv_c)
    
    true_end = time.time()
    print("Total time: ", true_end - true_start)


print("Energy list: ", Energy_list)
print("Gradient list: ", Gradient_list)

del madpno
del integrals
del opti
del world

