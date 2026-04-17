from time import time

import tequila as tq

import frayedends

true_start = time()
# initialize the PNO interface
geom = "Li 0.0 0.0 0.0\nH 0.0 0.0 3.0"  # geometry in Angstrom

world = frayedends.MadWorld3D()

madpno = frayedends.MadPNO(world, geom, n_orbitals=3)
orbitals = madpno.get_orbitals()
edges = madpno.get_spa_edges()
atomics = madpno.get_sto3g()

nuc_repulsion = madpno.get_nuclear_repulsion()
Vnuc = madpno.get_nuclear_potential()

for i in range(len(orbitals)):
    world.line_plot(f"pnoorb{i}.dat", orbitals[i])

integrals = frayedends.Integrals3D(world)
orbitals = integrals.orthonormalize(orbitals=orbitals)

for i in range(len(atomics)):
    world.line_plot(f"atomics{i}.dat", atomics[i])

# project the first hf orbital out of the atomics
active = integrals.project_out(kernel=[orbitals[0]], target=atomics)
active = integrals.orthonormalize(orbitals=active)
# make an active space: hf, Li-s, H-1s
frozen_orbitals = [orbitals[0]]
active_orbitals = [active[4], active[5]]

c = nuc_repulsion
# frozen core energy
kin = 2*integrals.compute_kinetic_integrals(frozen_orbitals).trace() 
pot = 2*integrals.compute_potential_integrals(frozen_orbitals, Vnuc).trace()
e_rep = integrals.compute_two_body_integrals(frozen_orbitals).elems[0,0,0,0]
c += kin + pot + e_rep
u = None
for iteration in range(6):
    integrals = frayedends.Integrals3D(world)
    G = integrals.compute_two_body_integrals(active_orbitals)
    FC_int = integrals.compute_frozen_core_interaction(frozen_orbitals, active_orbitals)
    T = integrals.compute_kinetic_integrals(active_orbitals)
    V = integrals.compute_potential_integrals(active_orbitals, Vnuc)
    S = integrals.compute_overlap_integrals(active_orbitals)
    print(S)

    for i in range(len(frozen_orbitals)):
        world.line_plot(f"fr_orb{i}.dat", frozen_orbitals[i])

    for i in range(len(active_orbitals)):
        world.line_plot(f"act_orb{i}.dat", active_orbitals[i])
    
    mol = tq.Molecule(
        geom, one_body_integrals=T + V + FC_int, two_body_integrals=G, nuclear_repulsion=c, frozen_core=False, n_electrons=2
    )
    U = mol.make_ansatz(name="UpCCGSD")

    # opt = tq.quantumchemistry.optimize_orbitals(molecule=mol, circuit=U, silent=True, initial_guess=u)
    # u = opt.mo_coeff
    # mol = opt.molecule

    H = mol.make_hamiltonian()
    E = tq.ExpectationValue(H=H, U=U)
    result = tq.minimize(E, silent=True)
    rdm1, rdm2 = mol.compute_rdms(U, variables=result.variables)
    # print(rdm1)
    # rdm1, rdm2 = frayedends.transform_rdms(TransformationMatrix=u, rdm1=rdm1, rdm2=rdm2)
    # print(rdm1)

    print("iteration {} energy {:+2.5f}".format(iteration, result.energy))

    opti = frayedends.Optimization3D(world, Vnuc, nuc_repulsion)
    frozen_orbitals, active_orbitals = opti.get_orbitals(
        orbitals=[frozen_orbitals, active_orbitals], rdm1=rdm1, rdm2=rdm2, opt_thresh=0.001, occ_thresh=0.001
    )
    c = opti.get_c()
    print(c)

true_end = time()
print("Total time: ", true_end - true_start)

del madpno
del integrals
del opti
del world
