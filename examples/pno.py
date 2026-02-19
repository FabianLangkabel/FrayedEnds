from time import time


import numpy as np
import tequila as tq
from pyscf import fci
import pno_utils
import frayedends

n_orbitals = 4
max_iterations = 10

def run_calculation(ortho_method, config):
    true_start = time()
    print(f"\n{'='*80}")
    print(f"Method: {ortho_method.upper()}")
    print(f"{'='*80}\n")

    geom = config["geometry"].replace("\\n", "\n")
    n_orbitals_config = config["n_orbitals"]
    units = config["units"]

    n_electrons = tq.quantumchemistry.ParametersQC(geometry=geom, units=units).total_n_electrons

    world = frayedends.MadWorld3D()

    madpno = frayedends.MadPNO(world, geom, units=units, n_orbitals=n_orbitals_config)
    all_orbitals = madpno.get_orbitals()
    print(frayedends.get_function_info(all_orbitals))

    nuc_repulsion = madpno.get_nuclear_repulsion()
    Vnuc = madpno.get_nuclear_potential()

    integrals = frayedends.Integrals3D(world)

    energies = []
    current = 0.0
    current_orbitals = []

    min_orbitals_needed = (n_electrons + 1) // 2

    # For mixed method, set initial occupation numbers once from all_orbitals
    if ortho_method == "mixed":
        initial_info = frayedends.get_function_info(all_orbitals)
        for i, orb in enumerate(all_orbitals):
            orb.occupation = initial_info[i]['occ']

    for o in range(n_orbitals_config):
        current_orbitals.append(all_orbitals[o])
        print(f"\n{'='*60}")
        print(f"Orbital {o + 1} added! Total orbitals: {o + 1}")
        print(f"{'='*60}")


        current_orbitals = integrals.orthonormalize(orbitals=current_orbitals, method=ortho_method)

        if (o + 1) < min_orbitals_needed:
            print(f"Skipping refinement: need at least {min_orbitals_needed} orbitals for {n_electrons} electrons")
            continue

        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration} ---")

            integrals = frayedends.Integrals3D(world)
            S = integrals.compute_overlap_integrals(current_orbitals)
            G = integrals.compute_two_body_integrals(current_orbitals, ordering='chem')
            T = integrals.compute_kinetic_integrals(current_orbitals)
            V = integrals.compute_potential_integrals(current_orbitals, Vnuc)

            # FCI
            e, fcivec = fci.direct_spin1.kernel(T + V, G.elems, o + 1, n_electrons)
            print(f"Energy: {e:+2.8f}")
            energies.append(e)

            rdm1, rdm2 = fci.direct_spin1.make_rdm12(fcivec, o + 1, n_electrons)
            rdm2 = np.swapaxes(rdm2, 1, 2)

            print("Orbital occupations:")
            for i in range(len(rdm1)):
                print(f"  Orbital {i}: {rdm1[i,i]:.6e}")

            opti = frayedends.Optimization3D(world, Vnuc, nuc_repulsion=nuc_repulsion)
            opti.set_orthonormalization_method(ortho_method)
            current_orbitals = opti.get_orbitals(
                orbitals=current_orbitals, rdm1=rdm1, rdm2=rdm2,
                opt_thresh=0.001, occ_thresh=0.001
            )

            if iteration > 0 and np.isclose(e, current, atol=1e-8, rtol=0.0):
                print(f"\nConverged after {iteration + 1} iterations!")
                break
            current = e

    del integrals
    del madpno
    del Vnuc
    del all_orbitals

    true_end = time()
    print(f"\nTotal time for {ortho_method}: {true_end - true_start:.2f}s")

    return world, energies


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("PNO Test - Incremental Orbital Refinement")
    print("=" * 80 + "\n")

    # Configuration
    config = {
        "geometry": "H 0.0 0.0 -0.8\\nH 0.0 0.0 0.8\\nH 0.0 0.0 -1.4\\nH 0.0 0.0 1.4",
        "n_orbitals": 6,
        "units": "angstrom"
    }


    world1, energies_symmetric = run_calculation("symmetric", config)
    del world1

    world2, energies_cholesky = run_calculation("cholesky", config)
    del world2

    world3, energies_mixed = run_calculation("mixed", config)
    del world3


    all_energies = {
        "symmetric": energies_symmetric,
        "cholesky": energies_cholesky,
        "mixed": energies_mixed
    }

    pno_utils.save_pno_results_to_json(all_energies, config)
    pno_utils.plot_pno_results_from_json()
    pno_utils.plot_energy_differences_between_methods()

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)

