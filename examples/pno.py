from time import time

import numpy as np
import tequila as tq
from pyscf import fci

import frayedends
import pno_utils

def run_calculation(ortho_method, config, is_first_method=False):
    true_start = time()
    print(f"\n{'='*80}")
    print(f"Method: {ortho_method.upper()}")
    print(f"{'='*80}\n")

    geom = config["geometry"].replace("\\n", "\n")  # Convert escaped newlines
    n_orbitals = config["n_orbitals"]
    units = config["units"]
    n_iterations = config["n_iterations"]

    # Calculate n_electrons from geometry (always matches the geometry)
    n_electrons = tq.quantumchemistry.ParametersQC(geometry=geom, units=units).total_n_electrons

    world = frayedends.MadWorld3D()

    madpno = frayedends.MadPNO(world, geom, units=units, n_orbitals=n_orbitals)
    orbitals = madpno.get_orbitals()
    print(frayedends.get_function_info(orbitals))

    nuc_repulsion = madpno.get_nuclear_repulsion()
    Vnuc = madpno.get_nuclear_potential()

    integrals = frayedends.Integrals3D(world)

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
    energies = []

    for iteration in range(n_iterations):
        print(f"\n--- Iteration {iteration} ---")

        integrals = frayedends.Integrals3D(world)
        G = integrals.compute_two_body_integrals(orbitals, ordering='chem')
        T = integrals.compute_kinetic_integrals(orbitals)
        V = integrals.compute_potential_integrals(orbitals, Vnuc)
        S = integrals.compute_overlap_integrals(orbitals) # overlap

        # FCA verwenden
        vqe_start = time()
        mol = tq.Molecule(geometry=geom, units="angstrom", one_body_integrals=T + V, two_body_integrals=G,
                          nuclear_repulsion=c)
        edges = madpno.get_spa_edges()
        # print(f"Edges for SPA ansatz: {edges}")

        U = mol.make_ansatz(name="SPA", edges=edges)
        H = mol.make_hamiltonian()
        E = tq.ExpectationValue(U, H)
        result = tq.minimize(E, silent=True)
        vqe_end = time()
        print(f"VQE time: {vqe_end - vqe_start:.2f} seconds")

        e = result.energy
        print("SPA energy: {:+2.10f}".format(e))

        # Log energy
        pno_utils.log_iteration(ortho_method, iteration, e,
                               config=config if iteration == 0 else None,
                               is_first_method=is_first_method)
        energies.append(e)

    del integrals
    del madpno
    del Vnuc

    true_end = time()
    print(f"\nTotal time for {ortho_method}: {true_end - true_start:.2f}s")

    return world, energies

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("PNO Test")
    print("=" * 80 + "\n")

    # Configuration
    config = {
        "geometry": "H 0.0 0.0 -0.8\\nH 0.0 0.0 0.8\\nH 0.0 0.0 -1.4\\nH 0.0 0.0 1.4",
        "n_orbitals": 4,
        "n_iterations": 6,
        "units": "angstrom"
    }
    # Collect energies for all methods
    all_energies = {}
    world3, energies_mixed = run_calculation("mixed", config)
    all_energies["mixed"] = energies_mixed
    del world3

    world1, energies_symmetric = run_calculation("symmetric", config, is_first_method=True)
    all_energies["symmetric"] = energies_symmetric
    del world1

    world2, energies_cholesky = run_calculation("cholesky", config)
    all_energies["cholesky"] = energies_cholesky
    del world2


    print("\n" + "=" * 80)
    print("Creating energy convergence plot...")
    print("=" * 80 + "\n")
    pno_utils.plot_energy_convergence(all_energies)

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)