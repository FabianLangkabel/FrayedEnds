from time import time

import tequila as tq

import madpy


def run(
    world,
):  # one of the ways to avoid deleting all objects manually is putting everything in a function except the world
    # initialize the PNO interface
    geom = "H 0.0 0.0 -1.25\nH 0.0 0.0 1.25"  # geometry in Angstrom
    madpno = madpy.MadPNO(world, geom, n_orbitals=2)
    orbitals = madpno.get_orbitals()
    print(madpy.get_function_info(orbitals))

    nuc_repulsion = madpno.get_nuclear_repulsion()
    Vnuc = madpno.get_nuclear_potential()

    integrals = madpy.Integrals3D(world)
    orbitals = integrals.orthonormalize(orbitals=orbitals)

    for i in range(len(orbitals)):
        world.line_plot(f"pnoorb{i}.dat", orbitals[i])

    c = nuc_repulsion
    for iteration in range(6):

        integrals = madpy.Integrals3D(world)
        G = integrals.compute_two_body_integrals(orbitals)
        T = integrals.compute_kinetic_integrals(orbitals)
        V = integrals.compute_potential_integrals(orbitals, Vnuc)
        S = integrals.compute_overlap_integrals(orbitals)

        mol = tq.Molecule(
            geom, one_body_integrals=T + V, two_body_integrals=G, nuclear_repulsion=c
        )

        U = mol.make_ansatz(name="UpCCGD")
        H = mol.make_hamiltonian()
        E = tq.ExpectationValue(H=H, U=U)
        result = tq.minimize(E, silent=True)
        rdm1, rdm2 = mol.compute_rdms(U, variables=result.variables)

        print("iteration {} energy {:+2.5f}".format(iteration, result.energy))

        opti = madpy.Optimization3D(world, Vnuc, nuc_repulsion)
        orbitals = opti.get_orbitals(
            orbitals=orbitals, rdm1=rdm1, rdm2=rdm2, opt_thresh=0.001, occ_thresh=0.001
        )
        c = (
            opti.get_c()
        )  # if there are no frozen core electrons, this should always be equal to the nuclear repulsion

        for i in range(len(orbitals)):
            world.line_plot(f"orb{i}.dat", orbitals[i])


true_start = time()

world = madpy.MadWorld3D()
run(world)

true_end = time()
print("Total time: ", true_end - true_start)
