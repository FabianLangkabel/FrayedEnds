from time import time

import numpy as np
import tequila as tq

import frayedends

true_start = time()
name = "LiH"
print(name)
geom = "Li 0.0 0.0 0.0\nH 0.0 0.0 1.25"  # geometry in Angstrom
molgeom = frayedends.MolecularGeometry(geom, units="angstrom")
world = frayedends.MadWorld(ndims=3)

# initialize the PNO interface
madpno = frayedends.MadPNO(world, geom, units="angstrom", n_orbitals=3)
orbitals = madpno.get_orbitals()

print("info:", frayedends.get_function_info(orbitals))
print("pno_groupings:", madpno.get_pno_groupings())
integrals = frayedends.Integrals(world)
orbitals = integrals.orthonormalize(orbitals=orbitals)
print(madpno.get_spa_edges(frozen_core=False))
print(madpno.get_spa_edges(frozen_core=True))

nuc_repulsion = madpno.get_nuclear_repulsion()
Vnuc = madpno.get_nuclear_potential()

c = nuc_repulsion
for iteration in range(30):
    integrals = frayedends.Integrals(world)
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
    print(rdm1)
    print(len(orbitals))
    print("iteration {} energy {:+2.10f}".format(iteration, result.energy))

    orbitals = [orbitals[:1], orbitals[1:]]
    opti = frayedends.Optimization(world, Vnuc, nuc_repulsion)
    orbitals = opti.get_orbitals(
        orbitals=orbitals,
        rdm1=rdm1,
        rdm2=rdm2,
        opt_thresh=0.001,
        occ_thresh=0.001,
        redirect_filename=f"madopt{iteration}.log",
    )
    orbitals = orbitals[0] + orbitals[1]


true_end = time()
print("Total time: ", true_end - true_start)

frayedends.cleanup(globals())
