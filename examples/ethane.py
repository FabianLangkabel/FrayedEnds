from time import time

import tequila as tq

import frayedends as fe

true_start = time()
# initialize the PNO interface
geom = """C -0.7560    0.0000    0.0000 
C 0.7560    0.0000    0.0000 
H -1.1404    0.6586    0.7845 
H -1.1404    0.3501   -0.9626 
H -1.1405   -1.0087    0.1781 
H 1.1404   -0.3501    0.9626 
H 1.1405    1.0087   -0.1781 
H 1.1404   -0.6586   -0.7845"""

world = fe.MadWorld3D()

madpno = fe.MadPNO(world, geom, units="angstrom", n_orbitals=16)
orbitals = madpno.get_orbitals()
edges = madpno.get_spa_edges()
print(fe.get_function_info(orbitals))
print(edges)

nuc_repulsion = madpno.get_nuclear_repulsion()
Vnuc = madpno.get_nuclear_potential()

integrals = fe.Integrals3D(world)
orbitals = integrals.orthonormalize(orbitals=orbitals)

for i in range(len(orbitals)):
    orbitals[i].save_to_file(f"pnoorb{i}_saved.00001")

c = nuc_repulsion
current = 0.0
for iteration in range(10):
    print("c: ", c)
    integrals = fe.Integrals3D(world)
    if iteration == 0:
        G = integrals.compute_two_body_integrals(orbitals)
        T = integrals.compute_kinetic_integrals(orbitals)
        V = integrals.compute_potential_integrals(orbitals, Vnuc)
        mol = tq.Molecule(
            geom,
            units="angstrom",
            one_body_integrals=T + V,
            two_body_integrals=G,
            nuclear_repulsion=c,
        )
        params0 = mol.parameters
    else:
        G = integrals.compute_two_body_integrals(act_orbs)
        T = integrals.compute_kinetic_integrals(act_orbs)
        V = integrals.compute_potential_integrals(act_orbs, Vnuc)
        FC_interaction = integrals.compute_frozen_core_interaction(fr_orbs, act_orbs)
        # print("h1:",h1)
        # print("g2:",g2.elems)
        params0.frozen_core = False
        mol = tq.Molecule(
            parameters=params0,
            one_body_integrals=T + V + FC_interaction,
            two_body_integrals=G,
            nuclear_repulsion=c,
            n_electrons=14,
        )

    U = mol.make_spa_ansatz(edges=edges)
    H = mol.make_hamiltonian()
    E = tq.ExpectationValue(H=H, U=U)
    result = tq.minimize(E, silent=False)
    rdm1, rdm2 = mol.compute_rdms(U, variables=result.variables)
    print("rdm1:", rdm1)
    print("rdm2:", rdm2)

    print("iteration {} energy {:+2.5f}".format(iteration, result.energy))
    if abs(current - result.energy) < 1e-6:
        break
    current = result.energy

    opti = fe.Optimization3D(world, Vnuc, nuc_repulsion)
    orbitals = opti.get_orbitals(
        orbitals=orbitals, rdm1=rdm1, rdm2=rdm2, opt_thresh=0.001, occ_thresh=0.001
    )
    c = opti.get_c()
    fr_orbs = []
    act_orbs = []
    for i in range(len(orbitals)):
        if orbitals[i].type == "frozen_occ":
            fr_orbs.append(orbitals[i])
        else:
            act_orbs.append(orbitals[i])

    for i in range(len(orbitals)):
        orbitals[i].save_to_file(f"orb_iter{iteration}_orb{i}_saved.00001")

true_end = time()
print("Total time: ", true_end - true_start)

fe.cleanup(globals())
