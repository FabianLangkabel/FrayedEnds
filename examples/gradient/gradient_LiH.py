import time

import numpy as np
import tequila as tq

import madpy


def get_best_initial_values(mol):
    tries = 20
    U = mol.make_ansatz(name="HCB-UpCCGD")
    best_opt = tq.quantumchemistry.optimize_orbitals(
        molecule=mol, circuit=U, silent=True, use_hcb=True, initial_guess="random"
    )
    opt = tq.quantumchemistry.optimize_orbitals(
        molecule=mol, circuit=U, silent=True, use_hcb=True
    )
    if opt.energy < best_opt.energy:
        best_opt = opt

    for _ in range(tries):
        # opt = tq.quantumchemistry.optimize_orbitals(molecule=mol, circuit=U, silent=True, use_hcb=True, initial_guess="random")
        initial_guess = np.eye(mol.n_orbitals) + np.random.normal(
            scale=1.0, loc=0.0, size=mol.n_orbitals**2
        ).reshape(mol.n_orbitals, mol.n_orbitals)
        opt = tq.quantumchemistry.optimize_orbitals(
            molecule=mol,
            circuit=U,
            silent=True,
            use_hcb=True,
            initial_guess=initial_guess,
        )
        if opt.energy < best_opt.energy:
            best_opt = opt

    return best_opt


def transform_rdms(TransformationMatrix, rdm1, rdm2):
    new_rdm1 = np.dot(
        np.dot(TransformationMatrix.transpose(), rdm1), TransformationMatrix
    )
    n = rdm2.shape[0]

    temp1 = np.zeros(shape=(n, n, n, n))
    for i in range(n):
        for j in range(n):
            for k2 in range(n):
                for l in range(n):
                    k_value = 0
                    for k in range(n):
                        k_value += TransformationMatrix[k][k2] * rdm2[i][j][k][l]
                    temp1[i][j][k2][l] = k_value

    temp2 = np.zeros(shape=(n, n, n, n))
    for i2 in range(n):
        for j in range(n):
            for k2 in range(n):
                for l in range(n):
                    i_value = 0
                    for i in range(n):
                        i_value += TransformationMatrix[i][i2] * temp1[i][j][k2][l]
                    temp2[i2][j][k2][l] = i_value

    temp3 = np.zeros(shape=(n, n, n, n))
    for i2 in range(n):
        for j in range(n):
            for k2 in range(n):
                for l2 in range(n):
                    l_value = 0
                    for l in range(n):
                        l_value += TransformationMatrix[l][l2] * temp2[i2][j][k2][l]
                    temp3[i2][j][k2][l2] = l_value

    new_rdm2 = np.zeros(shape=(n, n, n, n))
    for i2 in range(n):
        for j2 in range(n):
            for k2 in range(n):
                for l2 in range(n):
                    j_value = 0
                    for j in range(n):
                        j_value += TransformationMatrix[j][j2] * temp3[i2][j][k2][l2]
                    new_rdm2[i2][j2][k2][l2] = j_value

    return new_rdm1, new_rdm2


world = madpy.MadWorld3D(thresh=1e-6)

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
    madpno = madpy.MadPNO(world, geometry, n_orbitals=3, dft={"localize": "canon"})
    pno_orbitals = madpno.get_orbitals()

    orbitals = []
    orb0 = pno_orbitals[0]
    orb1 = pno_orbitals[1]

    factory = madpy.MRAFunctionFactory3D(world, s_orb_Li)
    orb2 = factory.get_function()
    orb2.type = "active"
    world.line_plot(f"pno3.dat", orb2)

    nuc_repulsion = madpno.get_nuclear_repulsion()
    Vnuc = madpno.get_nuclear_potential()

    integrals = madpy.Integrals3D(world)
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
        integrals = madpy.Integrals3D(world)
        if iteration == 0:
            G = integrals.compute_two_body_integrals(orbitals)
            T = integrals.compute_kinetic_integrals(orbitals)
            V = integrals.compute_potential_integrals(orbitals, Vnuc)
            S = integrals.compute_overlap_integrals(orbitals)
            # print("h1:",T+V)
            # print("g2:",G.elems)
            # print("s:",S)
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

        opti = madpy.Optimization3D(world, Vnuc, nuc_repulsion)
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

madpy.cleanup(globals())
