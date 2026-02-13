import os
import json
from datetime import datetime
from typing import Dict, List, Callable, Any

from ortho_test import *


class MultiRunTest:
    def __init__(self, base_results_dir: str, num_runs: int = 10,
                 max_iterations_per_run: int = 10):
        self.base_results_dir = base_results_dir
        self.num_runs = num_runs
        self.max_iterations_per_run = max_iterations_per_run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.main_dir = os.path.join(base_results_dir, f"multi_run_{self.timestamp}")
        os.makedirs(self.main_dir, exist_ok=True)

        self.main_log_path = os.path.join(self.main_dir, "main_log.txt")

        self.all_results = {
            'config': {},
            'runs': [],
            'statistics': {}
        }

    def write_config(self, test_configs: List[Dict]):
        with open(self.main_log_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MULTI-RUN TEST - CONFIGURATION\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("IMPORTANT CONFIGURATION:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Number of runs: {self.num_runs}\n")
            f.write(f"Iterations per run: {self.max_iterations_per_run}\n")
            f.write(f"Number of electrons: {n_electrons}\n")
            f.write(f"Number of orbitals: {n_orbitals}\n")
            f.write(f"Energy convergence threshold: {econv}\n\n")

            f.write("TEST CONFIGURATIONS:\n")
            f.write("-" * 80 + "\n")
            for i, config in enumerate(test_configs, 1):
                f.write(f"\nTest {i}: {config['name']}\n")
                f.write(f"  Potential function: {config['potential_func'].__name__}\n")
                f.write(f"  Geometry: {config['geometry']}\n")
                f.write(f"  Orthonormalization methods: {', '.join(config['ortho_methods'])}\n")

            f.write("\n" + "=" * 80 + "\n\n")

        self.all_results['config'] = {
            'timestamp': self.timestamp,
            'num_runs': self.num_runs,
            'max_iterations_per_run': self.max_iterations_per_run,
            'n_electrons': n_electrons,
            'n_orbitals': n_orbitals,
            'econv': econv,
            'tests': [
                {
                    'name': config['name'],
                    'potential_func': config['potential_func'].__name__,
                    'geometry': config['geometry'],
                    'ortho_methods': config['ortho_methods']
                }
                for config in test_configs
            ]
        }

    def run_single_test(self, run_number: int, test_name: str, potential_func: Callable,
                       geometry: str, ortho_method: str) -> Dict[str, Any]:
        """
        Run a single test configuration.
        """
        run_dir = os.path.join(self.main_dir, f"run_{run_number:02d}")
        test_dir = os.path.join(run_dir, test_name.lower().replace(' ', '_'))
        method_dir = os.path.join(test_dir, ortho_method)
        os.makedirs(method_dir, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"RUN {run_number}/{self.num_runs} - {test_name} - {ortho_method.upper()}")
        print(f"{'='*80}")

        start_time = datetime.now()

        try:
            energies, world = run_calculation(
                potential_func=potential_func,
                geometry=geometry,
                output_dir=method_dir,
                ortho_method=ortho_method,
                max_iterations=self.max_iterations_per_run,
                early_stop=False
            )
            del world

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            result = {
                'run_number': run_number,
                'test_name': test_name,
                'ortho_method': ortho_method,
                'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
                'duration_seconds': duration,
                'energies': energies,
                'num_iterations': len(energies),
                'final_energy': energies[-1] if energies else None,
                'converged': True,
                'error': None
            }

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            print(f"ERROR in run {run_number}: {str(e)}")
            result = {
                'run_number': run_number,
                'test_name': test_name,
                'ortho_method': ortho_method,
                'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
                'duration_seconds': duration,
                'energies': [],
                'num_iterations': 0,
                'final_energy': None,
                'converged': False,
                'error': str(e)
            }

        return result

    def log_run_result(self, result: Dict[str, Any]):
        with open(self.main_log_path, 'a') as f:
            f.write("-" * 80 + "\n")
            f.write(f"RUN {result['run_number']} - {result['test_name']} - {result['ortho_method']}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Timestamp: {result['start_time']}\n")
            f.write(f"Duration: {result['duration_seconds']:.2f} seconds\n")
            f.write(f"Converged: {result['converged']}\n")
            f.write(f"Number of iterations: {result['num_iterations']}\n")

            if result['error']:
                f.write(f"ERROR: {result['error']}\n")

            if result['energies']:
                f.write(f"Final energy: {result['final_energy']:+2.8f}\n")
                f.write("Energy progression:\n")
                for i, energy in enumerate(result['energies'], 1):
                    f.write(f"  Iteration {i}: {energy:+2.8f}\n")

            f.write("\n")

    def compute_statistics(self):
        grouped_results = {}

        for run in self.all_results['runs']:
            key = (run['test_name'], run['ortho_method'])
            if key not in grouped_results:
                grouped_results[key] = []

            if run['converged'] and run['final_energy'] is not None:
                grouped_results[key].append(run)

        statistics = {}

        for (test_name, ortho_method), results in grouped_results.items():
            if not results:
                continue

            final_energies = [r['final_energy'] for r in results]
            iterations = [r['num_iterations'] for r in results]
            durations = [r['duration_seconds'] for r in results]

            key = f"{test_name}_{ortho_method}"
            statistics[key] = {
                'test_name': test_name,
                'ortho_method': ortho_method,
                'num_successful_runs': len(results),
                'final_energy_mean': float(np.mean(final_energies)),
                'final_energy_std': float(np.std(final_energies)),
                'final_energy_min': float(np.min(final_energies)),
                'final_energy_max': float(np.max(final_energies)),
                'iterations_mean': float(np.mean(iterations)),
                'iterations_std': float(np.std(iterations)),
                'duration_mean': float(np.mean(durations)),
                'duration_std': float(np.std(durations)),
                'all_final_energies': final_energies,
                'all_iterations': iterations
            }

        self.all_results['statistics'] = statistics
        return statistics

    def write_statistics(self):
        statistics = self.all_results['statistics']

        with open(self.main_log_path, 'a') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("MEANS OVER ALL RUNS\n")
            f.write("=" * 80 + "\n\n")

            for key, stats in statistics.items():
                f.write("-" * 80 + "\n")
                f.write(f"{stats['test_name']} - {stats['ortho_method']}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Successful runs: {stats['num_successful_runs']}/{self.num_runs}\n\n")

                f.write("FINAL ENERGY:\n")
                f.write(f"  Mean:     {stats['final_energy_mean']:+2.8f}\n")
                f.write(f"  Std Dev:  {stats['final_energy_std']:2.8e}\n")
                f.write(f"  Min:      {stats['final_energy_min']:+2.8f}\n")
                f.write(f"  Max:      {stats['final_energy_max']:+2.8f}\n")
                f.write(f"  Range:    {stats['final_energy_max'] - stats['final_energy_min']:2.8e}\n\n")

                f.write("ITERATIONS:\n")
                f.write(f"  Mean:     {stats['iterations_mean']:.2f}\n")
                f.write(f"  Std Dev:  {stats['iterations_std']:.2f}\n\n")

                f.write("COMPUTATION TIME:\n")
                f.write(f"  Mean:     {stats['duration_mean']:.2f} seconds\n")
                f.write(f"  Std Dev:  {stats['duration_std']:.2f} seconds\n\n")

                f.write("All final energies:\n")
                for i, energy in enumerate(stats['all_final_energies'], 1):
                    f.write(f"  Run {i}: {energy:+2.8f}\n")

                f.write("\n")

    def save_results_json(self):
        json_path = os.path.join(self.main_dir, "results.json")

        with open(json_path, 'w') as f:
            json.dump(self.all_results, f, indent=2)

        print(f"\nResults saved to JSON: {json_path}")

    def run_all_tests(self, test_configs: List[Dict]):
        self.write_config(test_configs)

        total_configs = sum(len(config['ortho_methods']) for config in test_configs)
        total_runs_count = self.num_runs * total_configs

        print(f"\n{'='*80}")
        print(f"STARTING MULTI-RUN TEST")
        print(f"Total iterations: {self.num_runs}")
        print(f"Tests per iteration: {len(test_configs)}")
        print(f"Total runs: {total_runs_count}")
        print(f"Results directory: {self.main_dir}")
        print(f"{'='*80}\n")

        for run_num in range(1, self.num_runs + 1):
            for test_config in test_configs:
                for ortho_method in test_config['ortho_methods']:
                    result = self.run_single_test(
                        run_number=run_num,
                        test_name=test_config['name'],
                        potential_func=test_config['potential_func'],
                        geometry=test_config['geometry'],
                        ortho_method=ortho_method
                    )

                    self.all_results['runs'].append(result)
                    self.log_run_result(result)

        print(f"\n{'='*80}")
        print("Computing statistics...")
        print(f"{'='*80}\n")

        self.compute_statistics()
        self.write_statistics()
        self.save_results_json()

        print(f"\n{'='*80}")
        print("MULTI-RUN TEST COMPLETED")
        print(f"Results saved to: {self.main_dir}")
        print(f"{'='*80}\n")

if __name__ == "__main__":
    # Define test configurations
    NUM_RUNS = 10
    MAX_ITERATIONS_PER_RUN = 10
    BASE_RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
    test_configs = [
        {
            'name': 'Single Gaussian Peak',
            'potential_func': potential_single_peak,
            'geometry': "H 0.0 0.0 0.0",
            'ortho_methods': ['symmetric', 'cholesky', 'mixed']
        },
        {
            'name': 'Three Gaussian Peaks',
            'potential_func': potential_three_peaks,
            'geometry': "H 0.0 0.0 0.0\nH 1.0 0.0 0.0\nH 0.0 1.0 0.0",
            'ortho_methods': ['symmetric', 'cholesky', 'mixed']
        },
        {
            'name': 'Coulomb Potential',
            'potential_func': potential_coulomb,
            'geometry': "He 0.0 0.0 0.0",
            'ortho_methods': ['symmetric', 'cholesky', 'mixed']
        }
    ]

    test = MultiRunTest(
        base_results_dir=BASE_RESULTS_DIR,
        num_runs=NUM_RUNS,
        max_iterations_per_run=MAX_ITERATIONS_PER_RUN
    )

    test.run_all_tests(test_configs)