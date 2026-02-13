from time import time

import numpy as np
import tequila as tq
from pyscf import fci

import frayedends

def run_calculation(ortho_method):
    true_start = time()
    print(f"\n{'='*80}")
    print(f"Method: {ortho_method.upper()}")
    print(f"{'='*80}\n")

    # initialize the PNO interface
    # geom = "H 0.0 0.0 -0.5\nH 0.0 0.0 0.5"  # geometry in Angstrom
    geom =  "H 0.0 0.0 -0.8\nH 0.0 0.0 0.8\nH 0.0 0.0 -1.4\nH 0.0 0.0 1.4" # 2H2 geometrie ausprobieren
    n_electrons = tq.quantumchemistry.ParametersQC(geometry=geom, units="angstrom").total_n_electrons
    n_orbitals = 4 # ausprobieren
    units = "angstrom"

    world = frayedends.MadWorld3D()

    madpno = frayedends.MadPNO(world, geom, units=units, n_orbitals=n_orbitals)
    orbitals = madpno.get_orbitals()
    print(frayedends.get_function_info(orbitals))

    nuc_repulsion = madpno.get_nuclear_repulsion()
    Vnuc = madpno.get_nuclear_potential()

    integrals = frayedends.Integrals3D(world)

    # For mixed method, extract occupation numbers from info and set orbital.occupation field
    if ortho_method == "mixed":
        info = frayedends.get_function_info(orbitals)
        for i, orb in enumerate(orbitals):
            orb.occupation = info[i]['occ']

    # overlaps berechnen, überschneidung der orbitale
    orbitals = integrals.orthonormalize(orbitals=orbitals, method=ortho_method) # symmetric, cholesky mixed vergleichen

    for i in range(len(orbitals)):
        world.line_plot(f"pnoorb{i}_{ortho_method}.dat", orbitals[i])

    c = nuc_repulsion
    rdm1 = None
    for iteration in range(6):
        print(f"\n--- Iteration {iteration} ---")

        integrals = frayedends.Integrals3D(world)
        G = integrals.compute_two_body_integrals(orbitals, ordering='chem')
        T = integrals.compute_kinetic_integrals(orbitals)
        V = integrals.compute_potential_integrals(orbitals, Vnuc)
        S = integrals.compute_overlap_integrals(orbitals) # overlap

        # FCA verwenden
        e, fcivec = fci.direct_spin1.kernel(T + V, G.elems, n_orbitals, n_electrons)  # Computes the energy and the FCI vector
        print(f"Energy: {e:+2.8f}")

        rdm1, rdm2 = fci.direct_spin1.make_rdm12(
            fcivec, n_orbitals, n_electrons
        )  # Computes the 1- and 2- body reduced density matrices
        rdm2 = np.swapaxes(rdm2, 1, 2)  # swapping axes to match convention used in orbital refinement code

        # Re-orthonormalize for next iteration
        if iteration < 5:  # Don't orthonormalize after last iteration
            orbitals = integrals.orthonormalize(orbitals=orbitals, method=ortho_method, rdm1=rdm1)

    # Cleanup local objects before returning
    del integrals
    del madpno
    del Vnuc

    true_end = time()
    print(f"\nTotal time for {ortho_method}: {true_end - true_start:.2f}s")

    return world

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("PNO Test")
    print("=" * 80 + "\n")

    world3 = run_calculation("mixed")
    del world3

    world1 = run_calculation("symmetric")
    del world1

    world2 = run_calculation("cholesky")
    del world2


    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)