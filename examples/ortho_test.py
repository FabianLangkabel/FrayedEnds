import os
from typing import Any

import numpy as np
import tequila as tq
from numpy import floating

import frayedends as fe
from ortho_test_utils import *

n_electrons = 4
n_orbitals = 6
econv = 1e-8

def potential_three_peaks(x: float, y: float) -> float:
    """Three Gaussian peaks potential"""
    a = -(3.0 + 1.0 * n_electrons)
    b = -(2.0 + 0.8 * n_electrons)
    c = -(1.0 + 0.5 * n_electrons)

    width_scale = 1.0 / np.sqrt(max(1, n_electrons * 0.5))
    alpha = 0.5 * width_scale

    r1 = np.linalg.norm(np.array([x, y]) - np.array([4.0, 0.0]))
    r2 = np.linalg.norm(np.array([x, y]) - np.array([0.0, 4.0]))
    r3 = np.linalg.norm(np.array([x, y]))

    return (a * np.exp(-alpha * r1 ** 2) +
            b * np.exp(-alpha * r2 ** 2) +
            c * np.exp(-alpha * r3 ** 2))

def potential_single_peak(x: float, y: float) -> float:
    """Single Gaussian peak potential"""
    c = -1.5 * n_electrons
    width = 0.5

    if n_electrons > 2:
        width_scale = 0.15
    else:
        width_scale = 0.5

    r = np.array([x, y])

    # Using the standard width (0.5)
    return c * np.exp(-(width * width_scale) * np.linalg.norm(r) ** 2)

def potential_coulomb(x: float, y: float) -> floating[Any]:
    """Helium-like potential: -1/(r^2 + epsilon)"""
    r_vec = np.array([x, y])
    r_norm = np.linalg.norm(r_vec)
    epsilon = 0.0000001
    return -n_electrons / np.sqrt(r_norm**2 + epsilon)


def run_calculation(potential_func, geometry, output_dir, ortho_method="mixed", max_iterations=10, early_stop=False):
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

        U = mol.make_ansatz(name="HCB-UpCCGD")

        H = mol.make_hardcore_boson_hamiltonian()
        E = tq.ExpectationValue(H=H, U=U)
        result = tq.minimize(E, silent=True)

        rdm1, rdm2 = mol.compute_rdms(U, variables=result.variables, use_hcb=True)

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
            output_dir,
            method_name=ortho_method,
            enable_zoom=True
        )
        if early_stop and np.isclose(result.energy, current, atol=econv, rtol=0.0):
            print(f"\nConverged after {iteration+1} iterations!")
            break
        current = result.energy
        check_potential_depth_warning('madopt.log')

    del orbitals
    del orbitals_before
    del opti
    del integrals
    del mra_pot
    del eigen
    del factory

    return energies, world


if __name__ == "__main__":
    # Plot the potentials for visualization
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "="*80)
    print("PLOTTING POTENTIALS")
    print("="*80 + "\n")

    test1 = False
    test2 = True
    test3 = True

    # Test 1: Three Gaussian Peaks
    if test1:
        print("\n" + "="*80)
        print("TEST 1: THREE GAUSSIAN PEAKS POTENTIAL")
        print("="*80)

        output_dir_three = os.path.join(os.path.dirname(__file__), 'results', 'three_gaussian_peaks')
        os.makedirs(output_dir_three, exist_ok=True)

        plot_potential(potential_three_peaks,
                       os.path.join(output_dir_three, 'potential_three_peaks.png'),
                       n_points=201, x_range=(-6, 6), y_range=(-6, 6),
                       title='Three Gaussian Peaks Potential')

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

        plot_potential(potential_single_peak,
                       os.path.join(output_dir_single, 'potential_single_peak.png'),
                       n_points=201, x_range=(-6, 6), y_range=(-6, 6),
                       title='Single Gaussian Peak Potential')

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

        plot_potential(potential_coulomb,
                       os.path.join(output_dir_helium, 'potential_helium.png'),
                       n_points=201, x_range=(-0.01, 0.01), y_range=(-0.01, 0.01),
                       title='Helium-like Potential')

        geometry = "He 0.0 0.0 0.0"
        energies_symmetric_helium, world7 = run_calculation(potential_coulomb, geometry, output_dir_helium, ortho_method="symmetric")
        del world7

        energies_cholesky_helium, world8 = run_calculation(potential_coulomb, geometry, output_dir_helium, ortho_method="cholesky")
        del world8

        energies_mixed_helium, world9 = run_calculation(potential_coulomb, geometry, output_dir_helium, ortho_method="mixed")
        del world9

        plot_energy_comparison(
            {'symmetric': energies_symmetric_helium, 'cholesky': energies_cholesky_helium, 'mixed': energies_mixed_helium},
            output_dir_helium,
            title='Energy Convergence: Helium Potential'
        )

    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)


