import os
import sys
import numpy as np
from ortho_test import run_calculation, potential_single_peak


def compare_energy_arrays(energies1, energies2, tolerance=1e-10):
    if len(energies1) != len(energies2):
        return False, None, f"Different lengths: {len(energies1)} vs {len(energies2)}"

    if len(energies1) == 0:
        return True, 0.0, "Both arrays are empty"

    differences = np.abs(np.array(energies1) - np.array(energies2))
    max_diff = np.max(differences)
    is_equal = max_diff <= tolerance

    details = []
    for i, (e1, e2, diff) in enumerate(zip(energies1, energies2, differences)):
        details.append(f"  Iteration {i + 1}: {e1:.12f} vs {e2:.12f} (diff: {diff:.2e})")

    return is_equal, max_diff, "\n".join(details)


def determinism_test(ortho_method="mixed", potential_func=potential_single_peak, test_name="Single Gaussian",
                     iterations=8, geometry="H 0.0 0.0 0.0", early_stop=False):
    print("\n" + "=" * 80)
    print(f"DETERMINISM TEST: {test_name} with {ortho_method.upper()} orthonormalization")
    print("=" * 80)

    output_dir = os.path.join(os.path.dirname(__file__), 'results', 'determinism_test')
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "-" * 80)
    print("FIRST RUN")
    print("-" * 80)
    energies_run1, world1 = run_calculation(
        potential_func,
        geometry,
        output_dir,
        ortho_method=ortho_method,
        max_iterations=iterations,
        early_stop=early_stop
    )
    del world1

    print("\n" + "-" * 80)
    print("SECOND RUN")
    print("-" * 80)
    energies_run2, world2 = run_calculation(
        potential_func,
        geometry,
        output_dir,
        ortho_method=ortho_method,
        max_iterations=iterations,
        early_stop=early_stop
    )
    del world2

    is_equal, max_diff, details = compare_energy_arrays(energies_run1, energies_run2, tolerance=1e-10)

    print(f"\nRun 1 energies: {len(energies_run1)} iterations")
    for i, e in enumerate(energies_run1):
        print(f"  Iteration {i + 1}: {e:.12f}")

    print(f"\nRun 2 energies: {len(energies_run2)} iterations")
    for i, e in enumerate(energies_run2):
        print(f"  Iteration {i + 1}: {e:.12f}")

    print("\n" + "-" * 80)
    print("Detailed Comparison:")
    print("-" * 80)
    print(details)

    return is_equal, max_diff


if __name__ == "__main__":
    is_equal, max_diff = determinism_test(
        ortho_method="symmetric",
        potential_func=potential_single_peak,
        test_name="Single Gaussian Peak"
    )
    sys.exit(0 if is_equal else 1)
