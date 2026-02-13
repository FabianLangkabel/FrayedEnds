import os
from datetime import datetime
from typing import Dict, List, Any
import matplotlib.pyplot as plt


def log_iteration(method, iteration, energy, config=None, log_file="pno_energies.log", path="results/pno", is_first_method=False):
    path = os.path.join(os.path.dirname(__file__), path)
    os.makedirs(path, exist_ok=True)
    log_path = os.path.join(path, log_file)

    is_new_file = not os.path.exists(log_path)

    with open(log_path, 'a') as f:
        if is_new_file:
            f.write("=" * 100 + "\n")
            f.write("PNO ENERGY LOG\n")
            f.write("=" * 100 + "\n")
            f.write(f"Log created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n")

        if is_first_method and iteration == 0:
            if not is_new_file:
                f.write("\n\n")
                print(f"✓ Starting new run in existing log: {log_path}")

            f.write("=" * 100 + "\n")
            f.write(f"NEW RUN - Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 100 + "\n")

            if config:
                f.write("CONFIGURATION:\n")
                f.write("-" * 100 + "\n")
                for key, value in config.items():
                    f.write(f"  {key}: {value}\n")
                f.write("-" * 100 + "\n")

            f.write("\n")
            f.write(f"{'Timestamp':<20} {'Method':<15} {'Iteration':<10} {'Energy (a.u.)':<25}\n")
            f.write("-" * 100 + "\n")

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"{timestamp:<20} {method:<15} {iteration:<10} {energy:+25.12f}\n")


def plot_energy_convergence(energies_dict: Dict[str, List[float]], output_dir="results/pno",
                            title="PNO Energy Convergence Comparison"):
    output_path = os.path.join(os.path.dirname(__file__), output_dir)
    os.makedirs(output_path, exist_ok=True)

    plt.figure(figsize=(12, 8))

    colors = {'symmetric': 'blue', 'cholesky': 'green', 'mixed': 'red'}
    markers = {'symmetric': 'o', 'cholesky': 's', 'mixed': '^'}

    for method, energies in energies_dict.items():
        if len(energies) > 0:
            iterations = list(range(len(energies)))
            color = colors.get(method, 'black')
            marker = markers.get(method, 'x')
            plt.plot(iterations, energies, marker=marker, color=color,
                    label=f"{method.capitalize()}", linewidth=2, markersize=8)

    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Energy (a.u.)', fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_file = os.path.join(output_path, 'energy_convergence.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Energy convergence plot saved to: {plot_file}")




