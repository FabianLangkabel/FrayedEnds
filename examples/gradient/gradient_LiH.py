import time

import numpy as np
import tequila as tq

import frayedends

world = frayedends.MadWorld3D(thresh=1e-6)

distance_list = [30.1631 + 0.025 * i for i in range(1)]
Energy_list = []
Gradient_list = []


def s_orb_Li(x, y, z):
    r = np.array([x, y, z + 57 / 2])
    return np.exp(-np.sqrt(np.linalg.norm(r))) / np.sqrt(np.pi)


for distance in distance_list:
    true_start = time.time()
    geometry = "" "Li 0.0 0.0 " + str(-distance / 2) + "\nH 0.0 0.0 " + str(
        distance / 2
    )

    pno_start = time.time()
    madpno = frayedends.MadPNO(world, geometry, n_orbitals=3, dft={"localize": "canon"})
    pno_orbitals = madpno.get_orbitals()

    orbitals = []
    orb0 = pno_orbitals[0]
    orb1 = pno_orbitals[1]

    factory = frayedends.MRAFunctionFactory3D(world, s_orb_Li)
    orb2 = factory.get_function()
    orb2.type = "active"
    world.line_plot(f"pno3.dat", orb2)

    nuc_repulsion = madpno.get_nuclear_repulsion()
    Vnuc = madpno.get_nuclear_potential()

    integrals = frayedends.Integrals3D(world)
    orb2 = integrals.project_out(kernel=[orb0], target=[orb2])[0]
    orbitals.append(orb0)
    orbitals.append(orb1)
    orbitals.append(orb2)
    orbitals = integrals.orthonormalize(orbitals=orbitals)
    for i in range(len(orbitals)):
        world.line_plot(f"pno{i}.dat", orbitals[i])

    frozen_orbitals = []
    for orb in orbitals:
        if orb.type == "frozen_occ":
            frozen_orbitals.append(orb)

    c = nuc_repulsion
    current = 0.0
    for iteration in range(12):
        print(c)
        integrals = frayedends.Integrals3D(world)
        if iteration == 0:
            G = integrals.compute_two_body_integrals(orbitals)
            T = integrals.compute_kinetic_integrals(orbitals)
            V = integrals.compute_potential_integrals(orbitals, Vnuc)
            S = integrals.compute_overlap_integrals(orbitals)
            # print("h1:",T+V)
            # print("g2:",G.elems)
            # print("s:",S)
            print(T + V)
            print(G.elems)
            mol = tq.Molecule(
                geometry,
                one_body_integrals=T + V,
                two_body_integrals=G,
                nuclear_repulsion=c,
            )
            params0 = mol.parameters
        else:
            h1 = np.array(h1_elements).reshape(2, 2)
            g2 = np.array(g2_elements).reshape(2, 2, 2, 2)
            g2 = tq.quantumchemistry.NBodyTensor(g2, ordering="dirac")
            # print("h1:",h1)
            # print("g2:",g2.elems)
            params0.frozen_core = False
            mol = tq.Molecule(
                parameters=params0,
                one_body_integrals=h1,
                two_body_integrals=g2,
                nuclear_repulsion=c,
                n_electrons=2,
            )

        U = mol.make_ansatz(name="UpCCGSD")
        H = mol.make_hamiltonian()
        res = np.linalg.eigvalsh(H.to_matrix())
        E = tq.ExpectationValue(H=H, U=U)
        result = tq.minimize(E, silent=True)
        rdm1, rdm2 = mol.compute_rdms(U, variables=result.variables)

        # print("1-RDM:", rdm1)
        # print("2-RDM:", rdm2)

        print("iteration {} energy {:+2.7f}".format(iteration, result.energy))
        print("Lowest eigenvalue of Hamiltonian:", res[0])
        if abs(result.energy - current) < 1e-6:
            break
        current = result.energy

        opti = frayedends.Optimization3D(world, Vnuc, nuc_repulsion)
        orbitals = opti.get_orbitals(
            orbitals=orbitals,
            rdm1=rdm1,
            rdm2=rdm2,
            maxiter=1,
            opt_thresh=0.00001,
            occ_thresh=0.00001,
        )
        active_orbitals = []
        for i in range(len(orbitals)):
            if orbitals[i].type == "active":
                active_orbitals.append(orbitals[i])
        c = opti.get_c()
        for i in range(len(orbitals)):
            world.line_plot(f"orb{i}.dat", orbitals[i])
        c, h1_elements, g2_elements = opti.get_integrals()
    Energy_list.append(result.energy)

    true_end = time.time()
    print("Total time: ", true_end - true_start)

print("distance_list=", distance_list)
print("Energy_list=", Energy_list)
print("Gradient_list=", Gradient_list)

del factory
del madpno
del integrals
del opti
del world
