import os
import json
from datetime import datetime

import numpy as np
import tequila as tq
from matplotlib import pyplot as plt
from sympy.codegen.ast import none


def save_pno_results_to_json(energies_dict, config, output_dir="results/pno", filename="pno_results.json"):
    output_path = os.path.join(os.path.dirname(__file__), output_dir)
    os.makedirs(output_path, exist_ok=True)

    data = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "n_electrons": tq.quantumchemistry.ParametersQC(
            geometry=config["geometry"].replace("\\n", "\n"),
            units=config["units"]
        ).total_n_electrons,
        "methods": {}
    }

    for method_name, energies in energies_dict.items():
        method_data = {
            "total_iterations": len(energies),
            "final_energy": float(energies[-1]) if energies else None,
            "energies": [float(e) for e in energies]
        }
        data["methods"][method_name] = method_data

    if len(energies_dict) > 0:
        final_energies = {m: data["methods"][m]["final_energy"]
                          for m in data["methods"] if data["methods"][m]["final_energy"] is not None}

        if final_energies:
            sorted_methods = sorted(final_energies.items(), key=lambda x: x[1])
            data["summary"] = {
                "best_method": sorted_methods[0][0],
                "best_energy": sorted_methods[0][1],
                "energy_ranking": [{"method": m, "energy": e} for m, e in sorted_methods],
                "energy_differences": {}
            }

            best_energy = sorted_methods[0][1]
            for method, energy in sorted_methods[1:]:
                data["summary"]["energy_differences"][f"{method}_vs_best"] = energy - best_energy

    json_file = os.path.join(output_path, filename)
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n✓ Results saved to: {json_file}")
    return json_file

def plot_pno_results_from_json(json_file="results/pno/pno_results.json"):
    json_path = os.path.join(os.path.dirname(__file__), json_file)

    if not os.path.exists(json_path):
        print(f"Error: Results file not found at {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    output_dir = os.path.dirname(json_path)
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    methods_data = {}
    for method_name, method_data in data['methods'].items():
        methods_data[method_name] = method_data['energies']

    plt.figure(figsize=(12, 8))

    colors = {'symmetric': 'blue', 'cholesky': 'green', 'mixed': 'red'}
    markers = {'symmetric': 's', 'cholesky': '^', 'mixed': 'o'}

    for method_name, energies in methods_data.items():
        color = colors.get(method_name, 'black')
        marker = markers.get(method_name, 'x')
        plt.plot(range(len(energies)), energies,
                 marker=marker, color=color, linewidth=2, markersize=6,
                 label=method_name.capitalize())

    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Energy (a.u.)', fontsize=14)
    plt.title('PNO Energy Convergence Comparison', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_file = os.path.join(plots_dir, 'pno_energy_convergence.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Plots saved to: {plots_dir}")


def filter_by_threshold(iterations, differences, threshold):
    mask = np.abs(differences) <= threshold
    filtered_iterations = iterations[mask]
    filtered_differences = differences[mask]
    removed_count = np.sum(~mask)
    return filtered_iterations, filtered_differences, removed_count

def plot_energy_differences_between_methods(json_file="results/pno/pno_results.json", threshold=None):
    json_path = os.path.join(os.path.dirname(__file__), json_file)

    with open(json_path, 'r') as f:
        data = json.load(f)

    output_dir = os.path.dirname(json_path)
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    methods_energies = {}
    for method_name, method_data in data['methods'].items():
        methods_energies[method_name] = np.array(method_data['energies'])

    required_methods = ['symmetric', 'cholesky', 'mixed']
    min_length = min(len(methods_energies[m]) for m in required_methods)
    for method in required_methods:
        methods_energies[method] = methods_energies[method][:min_length]

    iterations = np.arange(min_length)
    diff_cholesky_symmetric = methods_energies['cholesky'] - methods_energies['symmetric']
    diff_mixed_symmetric = methods_energies['mixed'] - methods_energies['symmetric']
    diff_cholesky_mixed = methods_energies['cholesky'] - methods_energies['mixed']

    if threshold is not None:
        iter_1, diff_1, removed_1 = filter_by_threshold(iterations, diff_cholesky_symmetric, threshold)
        iter_2, diff_2, removed_2 = filter_by_threshold(iterations, diff_mixed_symmetric, threshold)
        iter_3, diff_3, removed_3 = filter_by_threshold(iterations, diff_cholesky_mixed, threshold)

    else:
        iter_1, diff_1, removed_1 = iterations, diff_cholesky_symmetric, 0
        iter_2, diff_2, removed_2 = iterations, diff_mixed_symmetric, 0
        iter_3, diff_3, removed_3 = iterations, diff_cholesky_mixed, 0

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    # Plot 1: Cholesky - Symmetric
    if len(iter_1) > 0:
        label_1 = f'Cholesky - Symmetric (|ΔE| ≤ {threshold:.0e})' if threshold else 'Cholesky - Symmetric'
        axes[0].plot(iter_1, diff_1,
                    marker='o', color='green', linewidth=2, markersize=6,
                    label=label_1)
    axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    if threshold is not None:
        axes[0].axhline(y=threshold, color='red', linestyle=':', linewidth=1, alpha=0.3, label=f'Threshold: ±{threshold:.0e}')
        axes[0].axhline(y=-threshold, color='red', linestyle=':', linewidth=1, alpha=0.3)
    axes[0].set_xlabel('Iteration', fontsize=12)
    axes[0].set_ylabel('Energy Difference (a.u.)', fontsize=12)
    title_1 = 'Cholesky - Symmetric (filtered)' if threshold else 'Cholesky - Symmetric'
    axes[0].set_title(title_1, fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11, loc='best')
    axes[0].grid(True, alpha=0.3)
    if len(diff_1) > 0:
        final_diff_1 = diff_1[-1] if len(diff_1) > 0 else 0
        avg_diff_1 = np.mean(diff_1)
        axes[0].text(0.02, 0.98, f'Final: {final_diff_1:+.6e}\nAverage: {avg_diff_1:+.6e}\nPoints: {len(diff_1)}',
                    transform=axes[0].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Mixed - Symmetric
    if len(iter_2) > 0:
        label_2 = f'Mixed - Symmetric (|ΔE| ≤ {threshold:.0e})' if threshold else 'Mixed - Symmetric'
        axes[1].plot(iter_2, diff_2,
                    marker='s', color='red', linewidth=2, markersize=6,
                    label=label_2)
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    if threshold is not None:
        axes[1].axhline(y=threshold, color='red', linestyle=':', linewidth=1, alpha=0.3, label=f'Threshold: ±{threshold:.0e}')
        axes[1].axhline(y=-threshold, color='red', linestyle=':', linewidth=1, alpha=0.3)
    axes[1].set_xlabel('Iteration', fontsize=12)
    axes[1].set_ylabel('Energy Difference (a.u.)', fontsize=12)
    title_2 = 'Mixed - Symmetric (filtered)' if threshold else 'Mixed - Symmetric'
    axes[1].set_title(title_2, fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11, loc='best')
    axes[1].grid(True, alpha=0.3)
    if len(diff_2) > 0:
        final_diff_2 = diff_2[-1] if len(diff_2) > 0 else 0
        avg_diff_2 = np.mean(diff_2)
        axes[1].text(0.02, 0.98, f'Final: {final_diff_2:+.6e}\nAverage: {avg_diff_2:+.6e}\nPoints: {len(diff_2)}',
                    transform=axes[1].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 3: Cholesky - Mixed
    if len(iter_3) > 0:
        label_3 = f'Cholesky - Mixed (|ΔE| ≤ {threshold:.0e})' if threshold else 'Cholesky - Mixed'
        axes[2].plot(iter_3, diff_3,
                    marker='^', color='purple', linewidth=2, markersize=6,
                    label=label_3)
    axes[2].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    if threshold is not None:
        axes[2].axhline(y=threshold, color='red', linestyle=':', linewidth=1, alpha=0.3, label=f'Threshold: ±{threshold:.0e}')
        axes[2].axhline(y=-threshold, color='red', linestyle=':', linewidth=1, alpha=0.3)
    axes[2].set_xlabel('Iteration', fontsize=12)
    axes[2].set_ylabel('Energy Difference (a.u.)', fontsize=12)
    title_3 = 'Cholesky - Mixed (filtered)' if threshold else 'Cholesky - Mixed'
    axes[2].set_title(title_3, fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=11, loc='best')
    axes[2].grid(True, alpha=0.3)
    if len(diff_3) > 0:
        final_diff_3 = diff_3[-1] if len(diff_3) > 0 else 0
        avg_diff_3 = np.mean(diff_3)
        axes[2].text(0.02, 0.98, f'Final: {final_diff_3:+.6e}\nAverage: {avg_diff_3:+.6e}\nPoints: {len(diff_3)}',
                    transform=axes[2].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    plot_file = os.path.join(plots_dir, 'pno_energy_differences_between_methods.png')
    if threshold is not None:
        plot_file = os.path.join(plots_dir, 'pno_energy_differences_between_methods_threshold.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Energy differences plot saved to: {plot_file}")
    return plot_file


def plot_all_experiments(json_dir="results/pno/jsons"):
    json_path = os.path.join(os.path.dirname(__file__), json_dir)

    if not os.path.exists(json_path):
        print(f"Error: Directory not found at {json_path}")
        return

    json_files = [f for f in os.listdir(json_path) if f.endswith('.json')]

    if not json_files:
        print(f"Error: No JSON files found in {json_path}")
        return

    plots_dir = os.path.join(json_path, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    colors = {'symmetric': 'blue', 'cholesky': 'green', 'mixed': 'red'}
    markers = {'symmetric': 's', 'cholesky': '^', 'mixed': 'o'}

    for json_file in json_files:
        exp_name = json_file.replace('pno_results_', '').replace('.json', '')

        with open(os.path.join(json_path, json_file), 'r') as f:
            data = json.load(f)

        methods_data = {}
        for method_name, method_data in data['methods'].items():
            methods_data[method_name] = method_data['energies']

        plt.figure(figsize=(12, 8))
        for method_name, energies in methods_data.items():
            color = colors.get(method_name, 'black')
            marker = markers.get(method_name, 'x')
            plt.plot(range(len(energies)), energies,
                     marker=marker, color=color, linewidth=2, markersize=6,
                     label=method_name.capitalize())

        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Energy (a.u.)', fontsize=14)
        plt.title(f'PNO Energy Convergence - {exp_name}', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_file = os.path.join(plots_dir, f'convergence_{exp_name}.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Convergence plot saved: {plot_file}")

    fig, axes = plt.subplots(len(json_files), 1, figsize=(14, 5 * len(json_files)))
    if len(json_files) == 1:
        axes = [axes]

    for idx, json_file in enumerate(json_files):
        exp_name = json_file.replace('pno_results_', '').replace('.json', '')

        with open(os.path.join(json_path, json_file), 'r') as f:
            data = json.load(f)

        methods_energies = {}
        for method_name, method_data in data['methods'].items():
            methods_energies[method_name] = np.array(method_data['energies'])

        required_methods = ['symmetric', 'cholesky', 'mixed']
        min_length = min(len(methods_energies[m]) for m in required_methods if m in methods_energies)
        for method in required_methods:
            if method in methods_energies:
                methods_energies[method] = methods_energies[method][:min_length]

        iterations = np.arange(min_length)
        diff_cholesky_symmetric = methods_energies['cholesky'] - methods_energies['symmetric']
        diff_mixed_symmetric = methods_energies['mixed'] - methods_energies['symmetric']

        axes[idx].plot(iterations, diff_cholesky_symmetric,
                      marker='o', color='green', linewidth=2, markersize=4,
                      label='Cholesky - Symmetric')
        axes[idx].plot(iterations, diff_mixed_symmetric,
                      marker='s', color='red', linewidth=2, markersize=4,
                      label='Mixed - Symmetric')
        axes[idx].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        axes[idx].set_xlabel('Iteration', fontsize=12)
        axes[idx].set_ylabel('Energy Difference (a.u.)', fontsize=12)
        axes[idx].set_title(f'{exp_name}', fontsize=14, fontweight='bold')
        axes[idx].legend(fontsize=10, loc='best')
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()

    plot_file = os.path.join(plots_dir, 'all_experiments_differences.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ All experiments differences plot saved: {plot_file}")

    fig, axes = plt.subplots(len(json_files), 1, figsize=(14, 5 * len(json_files)))
    if len(json_files) == 1:
        axes = [axes]

    for idx, json_file in enumerate(json_files):
        exp_name = json_file.replace('pno_results_', '').replace('.json', '')

        with open(os.path.join(json_path, json_file), 'r') as f:
            data = json.load(f)

        methods_data = {}
        for method_name, method_data in data['methods'].items():
            methods_data[method_name] = method_data['energies']

        for method_name, energies in methods_data.items():
            color = colors.get(method_name, 'black')
            marker = markers.get(method_name, 'x')
            axes[idx].plot(range(len(energies)), energies,
                         marker=marker, color=color, linewidth=2, markersize=4,
                         label=method_name.capitalize())

        axes[idx].set_xlabel('Iteration', fontsize=12)
        axes[idx].set_ylabel('Energy (a.u.)', fontsize=12)
        axes[idx].set_title(f'{exp_name}', fontsize=14, fontweight='bold')
        axes[idx].legend(fontsize=10, loc='best')
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()

    plot_file = os.path.join(plots_dir, 'all_experiments_convergence.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ All experiments convergence plot saved: {plot_file}")


if __name__ == '__main__':
    plot_all_experiments()
