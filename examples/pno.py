from time import time

import tequila as tq

import frayedends

true_start = time()
name = "LiH"
print(name)
geom = "Li 0.0 0.0 0.0\nH 0.0 0.0 1.25"  # geometry in Angstrom
molgeom = frayedends.MolecularGeometry(geom, units="angstrom")
world = frayedends.MadWorld3D()

# initialize the PNO interface
madpno = frayedends.MadPNO(world, geom, units="angstrom", n_orbitals=7)
orbitals = madpno.get_orbitals()

print("info:", frayedends.get_function_info(orbitals))
print("pno_groupings:", madpno.get_pno_groupings())
integrals = frayedends.Integrals3D(world)
orbitals = integrals.orthonormalize(orbitals=orbitals)
for i in range(len(orbitals)):
    world.cube_plot(f"new{name}orb{i}", orbitals[i], molgeom, zoom=5.0, datapoints=81)
print(madpno.get_spa_edges(frozen_core=False))
print(madpno.get_spa_edges(frozen_core=True))

nuc_repulsion = madpno.get_nuclear_repulsion()
Vnuc = madpno.get_nuclear_potential()

integrals = frayedends.Integrals3D(world)
orbitals = integrals.orthonormalize(orbitals=orbitals)

"""
for i in range(len(orbitals)):
    world.line_plot(f"pnoorb{i}.dat", orbitals[i])

c = nuc_repulsion
for iteration in range(10):

    integrals = frayedends.Integrals3D(world)
    G = integrals.compute_two_body_integrals(orbitals)
    T = integrals.compute_kinetic_integrals(orbitals)
    V = integrals.compute_potential_integrals(orbitals, Vnuc)
    S = integrals.compute_overlap_integrals(orbitals)

    mol = tq.Molecule(
        geom,
        units="angstrom",
        one_body_integrals=T + V,
        two_body_integrals=G,
        nuclear_repulsion=c,
    )

    U = mol.make_ansatz(name="UpCCGD")
    H = mol.make_hamiltonian()
    E = tq.ExpectationValue(H=H, U=U)
    result = tq.minimize(E, silent=True)
    rdm1, rdm2 = mol.compute_rdms(U, variables=result.variables)

    print("iteration {} energy {:+2.10f}".format(iteration, result.energy))

    opti = frayedends.Optimization3D(world, Vnuc, nuc_repulsion)
    orbitals = opti.get_orbitals(
        orbitals=orbitals, rdm1=rdm1, rdm2=rdm2, opt_thresh=0.001, occ_thresh=0.001
    )
    c = (
        opti.get_c()
    )  # if there are no frozen core electrons, this should always be equal to the nuclear repulsion

    for i in range(len(orbitals)):
        world.line_plot(f"orb{i}.dat", orbitals[i])

true_end = time()
print("Total time: ", true_end - true_start)
"""
frayedends.cleanup(globals())
