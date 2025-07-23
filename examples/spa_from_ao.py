import madpy
import tequila as tq
from time import time

true_start = time()
# initialize the PNO interface
geom = "Li 0.0 0.0 0.0\nH 0.0 0.0 3.0"  # geometry in Angstrom

world = madpy.MadWorld()

madpno = madpy.MadPNO(world, geom, n_orbitals=3)
orbitals = madpno.get_orbitals()
edges = madpno.get_spa_edges()
atomics = madpno.get_sto3g()

nuc_repulsion = madpno.get_nuclear_repulsion()
Vnuc = madpno.get_nuclear_potential()
# del madpno

for i in range(len(orbitals)):
    world.line_plot(f"pnoorb{i}.dat", orbitals[i])

integrals = madpy.Integrals(world)
orbitals = integrals.orthonormalize(orbitals=orbitals)

for i in range(len(atomics)):
    world.line_plot(f"atomics{i}.dat", atomics[i])

# project the first hf orbital out of the atomics
active = integrals.project_out(kernel=[orbitals[0]], target=atomics)
active = integrals.orthonormalize(orbitals=active)
# make an active space: hf, Li-s, H-1s
orbitals = [orbitals[0], active[4], active[5]]
orbitals[0].type="frozen_occ"
orbitals[1].type="active"
orbitals[2].type="active"
# del integrals

c = nuc_repulsion

u = None
for iteration in range(6):

    integrals = madpy.Integrals(world)
    G = integrals.compute_two_body_integrals(orbitals).elems
    T = integrals.compute_kinetic_integrals(orbitals)
    V = integrals.compute_potential_integrals(orbitals, Vnuc)
    S = integrals.compute_overlap_integrals(orbitals)
    print(S)

    for i in range(len(orbitals)):
        world.line_plot(f"orb{i}.dat", orbitals[i])

    mol = tq.Molecule(geom, one_body_integrals=T + V, two_body_integrals=G, nuclear_repulsion=c)
    U = mol.make_ansatz(name="UpCCGSD")

    #opt = tq.quantumchemistry.optimize_orbitals(molecule=mol, circuit=U, silent=True, initial_guess=u)
    #u = opt.mo_coeff
    #mol = opt.molecule

    H = mol.make_hamiltonian()
    E = tq.ExpectationValue(H=H, U=U)
    result = tq.minimize(E, silent=True)
    rdm1, rdm2 = mol.compute_rdms(U, variables=result.variables)
    # print(rdm1)
    # rdm1, rdm2 = madpy.transform_rdms(TransformationMatrix=u, rdm1=rdm1, rdm2=rdm2)
    # print(rdm1)

    print("iteration {} energy {:+2.5f}".format(iteration, result.energy))

    opti = madpy.Optimization(world, Vnuc, nuc_repulsion, parameters=param)
    orbitals = opti.get_orbitals(orbitals=orbitals, rdm1=rdm1, rdm2=rdm2, opt_thresh=0.001, occ_thresh=0.001)
    print(orbitals)
    # del opti

    for i in range(len(orbitals)):
        world.line_plot(f"orb{i}.dat", orbitals[i])

true_end = time()
print("Total time: ", true_end - true_start)




