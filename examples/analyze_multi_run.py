import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict


def load_results(results_json_path: str) -> Dict:
    with open(results_json_path, 'r') as f:
        return json.load(f)

def plot_mean_energy_convergence(results: Dict, output_dir: str = None):
    # Group by test name
    test_names = set(stats['test_name'] for stats in results['statistics'].values())

    for test_name in sorted(test_names):
        fig, ax = plt.subplots(figsize=(10, 6))

        methods_data = {}

        for key, stats in sorted(results['statistics'].items()):
            if stats['test_name'] == test_name:
                method = stats['ortho_method']

                # Get all runs for this test and method
                runs_for_method = [
                    run for run in results['runs']
                    if run['test_name'] == test_name
                    and run['ortho_method'] == method
                    and run['converged']
                    and len(run['energies']) > 0
                ]

                if not runs_for_method:
                    continue

                # Find the maximum number of iterations across all runs
                max_iterations = max(len(run['energies']) for run in runs_for_method)

                # Compute mean and std for each iteration
                mean_energies = []
                std_energies = []

                for iteration in range(max_iterations):
                    energies_at_iteration = [
                        run['energies'][iteration]
                        for run in runs_for_method
                        if len(run['energies']) > iteration
                    ]

                    if energies_at_iteration:
                        mean_energies.append(np.mean(energies_at_iteration))
                        std_energies.append(np.std(energies_at_iteration))
                    else:
                        mean_energies.append(np.nan)
                        std_energies.append(np.nan)

                methods_data[method] = {
                    'mean': mean_energies,
                    'std': std_energies,
                    'num_runs': len(runs_for_method)
                }

        colors = {'symmetric': 'blue', 'cholesky': 'green', 'mixed': 'red'}
        markers = {'symmetric': 'o', 'cholesky': 's', 'mixed': '^'}

        for method, data in sorted(methods_data.items()):
            iterations = np.arange(1, len(data['mean']) + 1)
            color = colors.get(method, 'black')
            marker = markers.get(method, 'o')

            ax.plot(iterations, data['mean'],
                   color=color, marker=marker, linewidth=2, markersize=6,
                   label=f"{method} (n={data['num_runs']})")

            # Add shaded area for standard deviation
            mean_arr = np.array(data['mean'])
            std_arr = np.array(data['std'])
            ax.fill_between(iterations,
                           mean_arr - std_arr,
                           mean_arr + std_arr,
                           color=color, alpha=0.2)

        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Energy', fontsize=12)
        ax.set_title(f'Mean Energy Convergence: {test_name}\n({results["config"]["num_runs"]} runs)',
                    fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f'mean_energy_convergence_{test_name.lower().replace(" ", "_")}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
            print(f"Saved plot: {filename}")
        else:
            plt.show()

        plt.close()

if __name__ == "__main__":
    import sys

    results_path = sys.argv[1]

    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        sys.exit(1)

    results = load_results(results_path)

    analysis_dir = os.path.join(os.path.dirname(results_path), 'analysis')

    print("\n" + "="*80)
    print("Generating plots...")
    print("="*80)
    plot_mean_energy_convergence(results, analysis_dir)

    print("\n" + "="*80)
    print(f"Analysis complete! Results saved to: {analysis_dir}")
    print("="*80 + "\n")


