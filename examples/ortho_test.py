import os
from typing import Any

import numpy as np
import tequila as tq
from numpy import floating

import frayedends as fe
from ortho_test_utils import plot_orbitals_before_after, plot_energy_comparison, log_iteration

n_electrons = 4
n_orbitals = 4
econv = 1e-8

def potential_three_peaks(x: float, y: float) -> float:
    """Three Gaussian peaks potential"""
    a = -5
    b = -3
    c = -2


    # Peak 1 at (1, 0)
    r1 = np.linalg.norm(np.array([x, y]) - np.array([4.0, 0.0]))
    # Peak 2 at (0, 1)
    r2 = np.linalg.norm(np.array([x, y]) - np.array([0.0, 4.0]))
    # Peak 3 at (0, 0)
    r3 = np.linalg.norm(np.array([x, y]))  # Center

    return (a * np.exp(-0.5 * r1 ** 2) +
            b * np.exp(-0.5 * r2 ** 2) +
            c * np.exp(-0.5 * r3 ** 2))

def potential_single_peak(x: float, y: float) -> float:
    """Single Gaussian peak potential"""
    c = -3.0
    r = np.array([x, y])
    return c * np.exp(-0.5 * np.linalg.norm(r) ** 2)

def potential_helium(x: float, y: float) -> floating[Any]:
    """Helium-like potential: -1/(r^2 + epsilon) (attractive)"""
    r_vec = np.array([x, y])
    r_norm = np.linalg.norm(r_vec)
    epsilon = 0.0000001
    return -2.0 / np.sqrt(r_norm**2 + epsilon)


def run_calculation(potential_func, geometry, output_dir, ortho_method="mixed", max_iterations=6):
    print(f"\n{'='*80}")
    print(f"Running with {ortho_method.upper()} orthonormalization")
    print(f"{'='*80}\n")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    world = fe.MadWorld2D()

    factory = fe.MRAFunctionFactory2D(world, potential_func)
    mra_pot = factory.get_function()

    eigen = fe.Eigensolver2D(world, mra_pot)
    orbitals = eigen.get_orbitals(0, n_orbitals, 0, n_states=12)

    world.plane_plot( "potential.dat", mra_pot, datapoints=501, zoom=10)

    energies = []
    current = 0.0

    u = None

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration} ---")

        integrals = fe.Integrals2D(world)
        S = integrals.compute_overlap_integrals(orbitals)
        G = integrals.compute_two_body_integrals(orbitals, ordering="phys")
        T = integrals.compute_kinetic_integrals(orbitals)
        V = integrals.compute_potential_integrals(orbitals, mra_pot)
        # VQE
        mol = tq.Molecule(
            geometry=geometry,
            units="bohr",
            one_body_integrals=T+V,
            two_body_integrals=G,
            n_electrons=n_electrons,
            nuclear_repulsion=0.0
        )
        H2=mol.make_hamiltonian()
        res = np.linalg.eigvalsh(H2.to_matrix())
        print(res[0])

        # result.energy should be similar to res[0]
        # or loop till difference to loop before is small enough
        #while(True):
        U = mol.make_ansatz(name="HCB-UpCCGD")
        opt = tq.quantumchemistry.optimize_orbitals(
            molecule=mol, circuit=U, use_hcb=True, silent=True, initial_guess=u
        )
        H = opt.molecule.make_hardcore_boson_hamiltonian()
        E = tq.ExpectationValue(H=H, U=U)
        result = tq.minimize(E, silent=True)
        u = opt.mo_coeff


        rdm1, rdm2 = mol.compute_rdms(U, variables=result.variables, use_hcb=True)
        rdm1, rdm2 = fe.transform_rdms(u, rdm1, rdm2)

        print(f"Energy: {result.energy:+2.8f}")
        if iteration > 0:
            energies.append(result.energy)
        # Log this iteration
        log_iteration(iteration, result.energy, output_dir, ortho_method)

        print("Orbital occupations:")
        for i in range(len(rdm1)):
            print(f"  Orbital {i}: {rdm1[i,i]:.6e}")

        orbitals_before = [orb for orb in orbitals]

        opti = fe.Optimization2D(world, mra_pot, nuc_repulsion=0.0)

        opti.set_orthonormalization_method(ortho_method)
        orbitals = opti.get_orbitals(
            orbitals=orbitals, rdm1=rdm1, rdm2=rdm2,
            opt_thresh=0.001, occ_thresh=0.001
        )

        plot_orbitals_before_after(
            integrals.transform_to_natural_orbitals(orbitals_before, rdm1)[0],
            integrals.transform_to_natural_orbitals(orbitals, rdm1)[0],
            world,
            iteration,
            output_dir, method_name=ortho_method
        )
        print("occ_num:", integrals.transform_to_natural_orbitals(orbitals_before, rdm1)[1])
        if np.isclose(result.energy, current, atol=econv, rtol=0.0):
            print(f"\nConverged after {iteration+1} iterations!")
            break
        current = result.energy

    del orbitals
    del orbitals_before
    del opti
    del integrals
    del mra_pot
    del eigen
    del factory

    return energies, world


if __name__ == "__main__":
    test1 = True
    test2 = False
    test3 = False

    # Test 1: Three Gaussian Peaks
    if test1:
        print("\n" + "="*80)
        print("TEST 1: THREE GAUSSIAN PEAKS POTENTIAL")
        print("="*80)

        output_dir_three = os.path.join(os.path.dirname(__file__), 'results', 'three_gaussian_peaks')
        os.makedirs(output_dir_three, exist_ok=True)

        geometry = "H 0.0 0.0 0.0\nH 1.0 0.0 0.0\nH 0.0 1.0 0.0"

        energies_symmetric_three, world1 = run_calculation(potential_three_peaks, geometry, output_dir_three, ortho_method="symmetric")
        del world1

        energies_cholesky_three, world2 = run_calculation(potential_three_peaks, geometry, output_dir_three, ortho_method="cholesky")
        del world2

        energies_mixed_three, world3 = run_calculation(potential_three_peaks, geometry, output_dir_three, ortho_method="mixed")
        del world3

        plot_energy_comparison(
            {'symmetric': energies_symmetric_three, 'cholesky': energies_cholesky_three, 'mixed': energies_mixed_three},
            output_dir_three,
            title='Energy Convergence: Three Gaussian Peaks Potential'
        )

    # Test 2: Single Gaussian Peak
    if test2:
        print("\n" + "="*80)
        print("TEST 2: SINGLE GAUSSIAN PEAK POTENTIAL")
        print("="*80)

        output_dir_single = os.path.join(os.path.dirname(__file__), 'results', 'single_gaussian_peak')
        os.makedirs(output_dir_single, exist_ok=True)

        geometry = "H 0.0 0.0 0.0"
        energies_symmetric_single, world4 = run_calculation(potential_single_peak, geometry, output_dir_single, ortho_method="symmetric")
        del world4

        energies_cholesky_single, world5 = run_calculation(potential_single_peak, geometry, output_dir_single, ortho_method="cholesky")
        del world5

        energies_mixed_single, world6 = run_calculation(potential_single_peak, geometry, output_dir_single, ortho_method="mixed")
        del world6

        plot_energy_comparison(
            {'symmetric': energies_symmetric_single, 'cholesky': energies_cholesky_single, 'mixed': energies_mixed_single},
            output_dir_single,
            title='Energy Convergence: Single Gaussian Peak Potential'
        )

    # Test 3: Helium Potential
    if test3:
        print("\n" + "="*80)
        print("TEST 3: HELIUM POTENTIAL")
        print("="*80)

        output_dir_helium = os.path.join(os.path.dirname(__file__), 'results', 'helium_potential')
        os.makedirs(output_dir_helium, exist_ok=True)

        geometry = "He 0.0 0.0 0.0"
        energies_symmetric_helium, world7 = run_calculation(potential_helium, geometry, output_dir_helium, ortho_method="symmetric")
        del world7

        energies_cholesky_helium, world8 = run_calculation(potential_helium, geometry, output_dir_helium, ortho_method="cholesky")
        del world8

        energies_mixed_helium, world9 = run_calculation(potential_helium, geometry, output_dir_helium, ortho_method="mixed")
        del world9

        plot_energy_comparison(
            {'symmetric': energies_symmetric_helium, 'cholesky': energies_cholesky_helium, 'mixed': energies_mixed_helium},
            output_dir_helium,
            title='Energy Convergence: Helium Potential'
        )

    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)


