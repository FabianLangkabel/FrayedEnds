from time import time
import gc

import numpy as np
import tequila as tq
import pno_utils
import frayedends

n_orbitals = 4
max_iterations = 10

def run_calculation(ortho_method, config):
    true_start = time()
    print(f"\nMethod: {ortho_method.upper()}")

    geom = config["geometry"].replace("\\n", "\n")
    n_orbitals_config = config["n_orbitals"]
    units = config["units"]
    degeneracy_tol = config.get("degeneracy_tol", 1e-6)

    n_electrons = tq.quantumchemistry.ParametersQC(geometry=geom, units=units).total_n_electrons

    world = frayedends.MadWorld3D()

    madpno = frayedends.MadPNO(world, geom, units=units, n_orbitals=n_orbitals_config)
    all_orbitals = madpno.get_orbitals()

    nuc_repulsion = madpno.get_nuclear_repulsion()
    Vnuc = madpno.get_nuclear_potential()

    integrals = frayedends.Integrals3D(world)

    energies = []
    current_orbitals = []

    min_orbitals_needed = (n_electrons + 1) // 2

    if ortho_method == "mixed":
        initial_info = frayedends.get_function_info(all_orbitals)
        for i, orb in enumerate(all_orbitals):
            orb.occupation = initial_info[i]['occ']

    for o in range(n_orbitals_config):
        current_orbitals.append(all_orbitals[o])
        print(f"Orbital {o + 1}/{n_orbitals_config}")

        current_orbitals = integrals.orthonormalize(orbitals=current_orbitals, method=ortho_method, degeneracy_tol=degeneracy_tol)

        if (o + 1) < min_orbitals_needed:
            continue

        for iteration in range(max_iterations):
            integrals = frayedends.Integrals3D(world)
            S = integrals.compute_overlap_integrals(current_orbitals)
            G = integrals.compute_two_body_integrals(current_orbitals, ordering='chem')
            T = integrals.compute_kinetic_integrals(current_orbitals)
            V = integrals.compute_potential_integrals(current_orbitals, Vnuc)

            vqe_start = time()
            mol = tq.Molecule(geometry=geom, units="angstrom",
                            one_body_integrals=T + V,
                            two_body_integrals=G,
                            nuclear_repulsion=nuc_repulsion)

            edges = madpno.get_spa_edges()

            U = mol.make_ansatz(name="SPA", edges=edges)
            H = mol.make_hamiltonian()
            E_vqe = tq.ExpectationValue(U, H)
            result = tq.minimize(E_vqe, silent=True)
            vqe_end = time()

            print(f"  Iteration {iteration}: E = {result.energy:+2.8f} (VQE: {vqe_end - vqe_start:.2f}s)")

            e = result.energy
            energies.append(e)

            rdm1, rdm2 = mol.compute_rdms(U=U, variables=result.variables)

            opti = frayedends.Optimization3D(world, Vnuc, nuc_repulsion=nuc_repulsion)
            opti.set_orthonormalization_method(ortho_method)
            current_orbitals = opti.get_orbitals(
                orbitals=current_orbitals, rdm1=rdm1, rdm2=rdm2,
                opt_thresh=0.001, occ_thresh=0.001
            )


    del integrals
    del madpno
    del Vnuc
    del all_orbitals
    del current_orbitals
    gc.collect()

    true_end = time()
    print(f"Total time: {true_end - true_start:.2f}s")

    return world, energies


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("PNO Test - Incremental Orbital Refinement")
    print("=" * 80 + "\n")

    # Configuration
    config = {
        "geometry": "H 0.0 0.0 -1.2\\nH 0.0 0.0 -0.4\\nH 0.0 0.0 0.4\\nH 0.0 0.0 1.2",
        "n_orbitals": 6,
        "units": "angstrom",
        "degeneracy_tol": 1e-6,
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

