import gc
from pno import run_calculation
import pno_utils

EXPERIMENTS = {
    "H2_chain_short": {
        "geometry": "H 0.0 0.0 -0.4\nH 0.0 0.0 0.4\nH 0.0 0.0 -0.8\nH 0.0 0.0 0.8",
        "n_orbitals": 6,
        "units": "angstrom"
    },
    "H2_chain_medium": {
        "geometry": "H 0.0 0.0 -0.8\nH 0.0 0.0 0.8\nH 0.0 0.0 -2.0\nH 0.0 0.0 2.0",
        "n_orbitals": 6,
        "units": "angstrom"
    },
    "H2_chain_long": {
        "geometry": "H 0.0 0.0 -0.8\nH 0.0 0.0 0.8\nH 0.0 0.0 -3.8\nH 0.0 0.0 3.8",
        "n_orbitals": 6,
        "units": "angstrom"
    },
    "H4_linear": {
        "geometry": "H 0.0 0.0 0.0\nH 0.0 0.0 1.0\nH 0.0 0.0 2.0\nH 0.0 0.0 3.0",
        "n_orbitals": 6,
        "units": "angstrom"
    },
    "H4_square": {
        "geometry": "H 0.0 0.0 0.0\nH 1.5 0.0 0.0\nH 1.5 1.5 0.0\nH 0.0 1.5 0.0",
        "n_orbitals": 6,
        "units": "angstrom"
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
