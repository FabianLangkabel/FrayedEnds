import gc
from pno import run_calculation
import pno_utils

EXPERIMENTS = {
    "H2_chain_medium_lowest_degeneracy_tol": {
        "geometry": "H 0.0 0.0 -0.8\nH 0.0 0.0 0.8\nH 0.0 0.0 -2.0\nH 0.0 0.0 2.0",
        "n_orbitals": 6,
        "units": "angstrom",
        "degeneracy_tol": 1,
    },
    "H4_square_loweest__degeneracy_tol": {
        "geometry": "H 0.0 0.0 0.0\nH 1.5 0.0 0.0\nH 1.5 1.5 0.0\nH 0.0 1.5 0.0",
        "n_orbitals": 6,
        "units": "angstrom",
        "degeneracy_tol": 1e-2,
    },
}


def run_single_experiment(experiment_name, config):
    methods = ["symmetric", "cholesky", "mixed"]
    all_energies = {}

    for method in methods:
        world, energies = run_calculation(method, config)
        all_energies[method] = energies
        del world
        gc.collect()

    pno_utils.save_pno_results_to_json(
        all_energies,
        config,
        output_dir='results/pno/jsons',
        filename=f'pno_results_{experiment_name}.json'
    )

    gc.collect()


if __name__ == '__main__':
    for exp_name, config in EXPERIMENTS.items():
        print(f"\n{'='*80}")
        print(f"Experiment: {exp_name}")
        print(f"{'='*80}\n")
        run_single_experiment(exp_name, config)
        gc.collect()

    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETED")
    print(f"{'='*80}\n")
