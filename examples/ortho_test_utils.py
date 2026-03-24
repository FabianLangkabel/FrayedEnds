import os
import numpy as np
from matplotlib import pyplot as plt
import json
from datetime import datetime


def save_energies_to_combined_json(energies_dict, output_dir, n_orbitals, max_iterations):
    os.makedirs(output_dir, exist_ok=True)

    # Structure data
    data = {
        "timestamp": datetime.now().isoformat(),
        "n_orbitals": n_orbitals,
        "max_iterations": max_iterations,
        "methods": {}
    }

    # Add data for each method
    for method_name, energies in energies_dict.items():
        method_data = {
            "total_iterations": len(energies),
            "orbitals": []
        }

        # Group energies by orbital
        for orb_num in range(n_orbitals):
            start_idx = orb_num * max_iterations
            end_idx = start_idx + max_iterations

            if end_idx <= len(energies):
                orbital_energies = energies[start_idx:end_idx]
                method_data["orbitals"].append({
                    "orbital_number": orb_num + 1,
                    "iterations": [
                        {
                            "iteration": i,
                            "energy": float(e)
                        }
                        for i, e in enumerate(orbital_energies)
                    ]
                })

        data["methods"][method_name] = method_data

    # Save to single JSON file
    json_file = os.path.join(output_dir, 'energies_all_methods.json')
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✓ Saved all energies to: {json_file}")
    return json_file


def load_energies_from_combined_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Extract energies for each method
    methods_data = {}

    for method_name, method_data in data["methods"].items():
        # Flatten energies back into a single list
        energies = []
        for orbital_data in method_data["orbitals"]:
            for iteration_data in orbital_data["iterations"]:
                energies.append(iteration_data["energy"])

        methods_data[method_name] = energies

    return {
        "methods_energies": methods_data,
        "n_orbitals": data["n_orbitals"],
        "max_iterations": data["max_iterations"]
    }


def plot_energy_comparison_from_json(json_file, output_dir, title="Energy Convergence",
                                     skip_first_iteration=True):
    # Load all data from single JSON file
    data = load_energies_from_combined_json(json_file)
    methods_data = data["methods_energies"]
    n_orbitals = data["n_orbitals"]
    max_iterations = data["max_iterations"]

    # Create plots subdirectory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Debug: Print energy counts
    print(f"\n{'='*80}")
    print("Energy data summary:")
    for method_name, energies in methods_data.items():
        expected_count = n_orbitals * max_iterations
        print(f"  {method_name}: {len(energies)} energies (expected: {expected_count})")
    print(f"{'='*80}\n")

    markers = {'mixed': 'o', 'symmetric': 's', 'cholesky': '^'}
    colors = {'mixed': 'red', 'symmetric': 'blue', 'cholesky': 'green'}

    # Create a plot for each orbital number
    for orb_num in range(n_orbitals):
        plt.figure(figsize=(12, 7))

        for method_name, energies in methods_data.items():
            # Extract energies for this orbital
            start_idx = orb_num * max_iterations
            end_idx = start_idx + max_iterations

            if end_idx <= len(energies):
                orbital_energies = energies[start_idx:end_idx]

                # Skip first iteration if requested
                if skip_first_iteration and len(orbital_energies) > 1:
                    orbital_energies = orbital_energies[1:]
                    iteration_offset = 0
                else:
                    iteration_offset = 0

                marker = markers.get(method_name.lower(), 'o')
                color = colors.get(method_name.lower(), 'black')

                # Plot with adjusted iteration numbers
                iterations = list(range(iteration_offset, iteration_offset + len(orbital_energies)))
                plt.plot(iterations, orbital_energies,
                        marker=marker, linestyle='-', color=color,
                        label=method_name.capitalize(), linewidth=2, markersize=8)
            else:
                print(f"⚠ Warning: Not enough data for {method_name}, orbital {orb_num + 1}")
                print(f"  Expected index range [{start_idx}:{end_idx}], but only have {len(energies)} energies")

        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Energy (a.u.)', fontsize=14)

        if skip_first_iteration:
            plt.title(f'{title} - {orb_num + 1} Orbital(s) (excluding 1st iteration)',
                     fontsize=16, fontweight='bold')
        else:
            plt.title(f'{title} - {orb_num + 1} Orbital(s)', fontsize=16, fontweight='bold')

        plt.legend(fontsize=12, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save individual plot
        plot_filename = f'energy_convergence_{orb_num + 1}_orbitals.png'
        plt.savefig(os.path.join(plots_dir, plot_filename),
                    dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved plot for {orb_num + 1} orbital(s): {plot_filename}")

    # Also create an overview plot showing all orbitals together
    plt.figure(figsize=(14, 8))

    for method_name, energies in methods_data.items():
        marker = markers.get(method_name.lower(), 'o')
        color = colors.get(method_name.lower(), 'black')
        plt.plot(energies, marker=marker, linestyle='-', color=color,
                label=method_name.capitalize(), linewidth=2, markersize=6, alpha=0.7)

    # Add vertical lines to mark orbital additions
    for orb_num in range(1, n_orbitals):
        plt.axvline(x=orb_num * max_iterations, color='gray', linestyle=':',
                   linewidth=1, alpha=0.5)
        plt.text(orb_num * max_iterations, plt.ylim()[0],
                f' +Orb {orb_num + 1}', rotation=90,
                verticalalignment='bottom', fontsize=9, alpha=0.7)

    plt.xlabel('Total Iteration', fontsize=14)
    plt.ylabel('Energy (a.u.)', fontsize=14)
    plt.title(f'{title} - All Orbitals', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(os.path.join(plots_dir, 'energy_convergence_all.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved overview plot: energy_convergence_all.png")
    print(f"✓ All plots saved to: {plots_dir}")


def plot_potential(potential_func, output_path, n_points=201, x_range=(-5, 5), y_range=(-5, 5),
                   title="Potential", colormap='inferno', dpi=150):
    """
    Plot a 2D potential function.
    """
    # Create grid
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    X, Y = np.meshgrid(x, y)

    # Evaluate potential on grid
    Z = np.zeros_like(X)
    for i in range(n_points):
        for j in range(n_points):
            Z[i, j] = potential_func(X[i, j], Y[i, j])

    # Create plot
    plt.figure(figsize=(10, 8), dpi=dpi)
    plt.pcolormesh(X, Y, Z, shading='auto', cmap=colormap)
    plt.colorbar(label='Potential')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title(title, fontsize=14)
    plt.axis('equal')
    plt.tight_layout()

    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"Potential plot saved to: {output_path}")


def plot_orbitals_2d(orbitals, world, iteration, output_dir, method_name="mixed",
                     n_points=101, zoom_threshold=0.01, enable_zoom=True):
    plot_dir = os.path.join(output_dir, method_name)
    os.makedirs(plot_dir, exist_ok=True)

    n_orbs = len(orbitals)
    fig, axes = plt.subplots(1, n_orbs, figsize=(4*n_orbs, 4))
    if n_orbs == 1:
        axes = [axes]

    temp_files = []
    for i, orb in enumerate(orbitals):
        temp_file = f'_temp_orbital_{i}_iter_{iteration}_{method_name}.dat'
        temp_files.append(temp_file)
        world.plane_plot(temp_file, orb, datapoints=n_points)

    for i, temp_file in enumerate(temp_files):
        actual_file = f'plane_x1x2_{temp_file}'
        data = np.loadtxt(actual_file)

        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]

        x_unique = np.unique(x)
        y_unique = np.unique(y)
        Z = z.reshape(len(y_unique), len(x_unique))
        X, Y = np.meshgrid(x_unique, y_unique)

        z_max = np.max(np.abs(Z))
        threshold = z_max * zoom_threshold
        significant_mask = np.abs(Z) > threshold

        if enable_zoom and np.any(significant_mask):
            y_indices, x_indices = np.where(significant_mask)
            x_min_idx, x_max_idx = x_indices.min(), x_indices.max()
            y_min_idx, y_max_idx = y_indices.min(), y_indices.max()

            x_range = x_max_idx - x_min_idx
            y_range = y_max_idx - y_min_idx

            x_pad = max(int(x_range * 0.2), 10)
            y_pad = max(int(y_range * 0.2), 10)

            x_min_idx = max(0, x_min_idx - x_pad)
            x_max_idx = min(len(x_unique) - 1, x_max_idx + x_pad)
            y_min_idx = max(0, y_min_idx - y_pad)
            y_max_idx = min(len(y_unique) - 1, y_max_idx + y_pad)

            axes[i].set_xlim(x_unique[x_min_idx], x_unique[x_max_idx])
            axes[i].set_ylim(y_unique[y_min_idx], y_unique[y_max_idx])

        vmax = np.max(np.abs(Z))
        im = axes[i].contourf(X, Y, Z, levels=20, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[i].set_title(f'Orbital {i}\n(max: {z_max:.2e})')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')
        axes[i].set_aspect('equal')
        plt.colorbar(im, ax=axes[i], format='%.1e')

        os.remove(actual_file)

    plt.suptitle(f'Iteration {iteration} - {method_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'orbitals_iteration_{iteration}.png'), dpi=150)
    plt.close()


def plot_orbitals_before_after(orbitals_before, orbitals_after, world, iteration,
                                output_dir, method_name="mixed", n_points=101,
                                phase="normal", zoom_threshold=0.01, enable_zoom=True):
    plot_dir = os.path.join(output_dir, method_name)
    os.makedirs(plot_dir, exist_ok=True)

    n_orbs_before = len(orbitals_before)
    n_orbs_after = len(orbitals_after)
    n_orbs_max = max(n_orbs_before, n_orbs_after)

    fig, axes = plt.subplots(2, n_orbs_max, figsize=(4*n_orbs_max, 8))
    if n_orbs_max == 1:
        axes = axes.reshape(2, 1)

    def process_orbitals(orbitals, row_idx, prefix, label_suffix):
        """Helper function to process and plot a set of orbitals"""
        temp_files = []
        for i, orb in enumerate(orbitals):
            temp_file = f'_temp_{prefix}_{i}_iter_{iteration}_{phase}_{method_name}.dat'
            temp_files.append(temp_file)
            world.plane_plot(temp_file, orb, datapoints=n_points)

        for i, temp_file in enumerate(temp_files):
            actual_file = f'plane_x1x2_{temp_file}'
            data = np.loadtxt(actual_file)

            x = data[:, 0]
            y = data[:, 1]
            z = data[:, 2]

            x_unique = np.unique(x)
            y_unique = np.unique(y)
            Z = z.reshape(len(y_unique), len(x_unique))
            X, Y = np.meshgrid(x_unique, y_unique)

            z_max = np.max(np.abs(Z))
            threshold = z_max * zoom_threshold
            significant_mask = np.abs(Z) > threshold

            if enable_zoom and np.any(significant_mask):
                y_indices, x_indices = np.where(significant_mask)
                x_min_idx, x_max_idx = x_indices.min(), x_indices.max()
                y_min_idx, y_max_idx = y_indices.min(), y_indices.max()

                x_range = x_max_idx - x_min_idx
                y_range = y_max_idx - y_min_idx

                x_pad = max(int(x_range * 0.2), 10)
                y_pad = max(int(y_range * 0.2), 10)

                x_min_idx = max(0, x_min_idx - x_pad)
                x_max_idx = min(len(x_unique) - 1, x_max_idx + x_pad)
                y_min_idx = max(0, y_min_idx - y_pad)
                y_max_idx = min(len(y_unique) - 1, y_max_idx + y_pad)

                axes[row_idx, i].set_xlim(x_unique[x_min_idx], x_unique[x_max_idx])
                axes[row_idx, i].set_ylim(y_unique[y_min_idx], y_unique[y_max_idx])

            vmax = np.max(np.abs(Z))
            im = axes[row_idx, i].contourf(X, Y, Z, levels=20, cmap='RdBu_r', vmin=-vmax, vmax=vmax)

            # Determine title based on context
            if label_suffix == "Before":
                title = f'Orbital {i} ({label_suffix})\nmax: {z_max:.2e}'
            elif i < n_orbs_before:
                title = f'Orbital {i} ({label_suffix})\nmax: {z_max:.2e}'
            else:
                title = f'Orbital {i} (NEW)\nmax: {z_max:.2e}'
                axes[row_idx, i].set_title(title, color='red')
                axes[row_idx, i].set_xlabel('x')
                axes[row_idx, i].set_ylabel('y')
                axes[row_idx, i].set_aspect('equal')
                plt.colorbar(im, ax=axes[row_idx, i], format='%.1e')
                os.remove(actual_file)
                continue

            axes[row_idx, i].set_title(title)
            axes[row_idx, i].set_xlabel('x')
            axes[row_idx, i].set_ylabel('y')
            axes[row_idx, i].set_aspect('equal')
            plt.colorbar(im, ax=axes[row_idx, i], format='%.1e')

            os.remove(actual_file)

    # Process before orbitals
    process_orbitals(orbitals_before, 0, "before", "Before")
    for i in range(n_orbs_before, n_orbs_max):
        axes[0, i].axis('off')

    # Process after orbitals
    process_orbitals(orbitals_after, 1, "after", "After")
    for i in range(n_orbs_after, n_orbs_max):
        axes[1, i].axis('off')

    plt.suptitle(f'Iteration {iteration} - {method_name} - {phase}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'orbitals_iter_{iteration}_{phase}.png'), dpi=150)
    plt.close()




def log_iteration(iteration, energy, output_dir, method_name):
    """Log a single iteration to both current run log and history log"""
    from datetime import datetime

    # File 1: Current run only (cleared at start of each method)
    log_file_current = os.path.join(output_dir, 'energy_comparison.log')

    # File 2: History with timestamps (append forever)
    log_file_history = os.path.join(output_dir, 'energy_comparison_history.log')

    # For current run log: clear file when starting iteration 0
    if iteration == 0:
        with open(log_file_current, 'w') as f:
            f.write("="*80 + "\n")
            f.write("Energy Convergence Log - Current Run\n")
            f.write("="*80 + "\n\n")

    # Append to current run log
    with open(log_file_current, 'a') as f:
        f.write(f"Iteration {iteration:<6} | {method_name.capitalize():<12} | Energy: {energy:+20.10f}\n")

    # For history log: create header if file doesn't exist
    if not os.path.exists(log_file_history):
        with open(log_file_history, 'w') as f:
            f.write("="*80 + "\n")
            f.write("Energy Convergence Log - Complete History\n")
            f.write("="*80 + "\n\n")

    # Append to history log with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file_history, 'a') as f:
        f.write(f"[{timestamp}] Iteration {iteration:<6} | {method_name.capitalize():<12} | Energy: {energy:+20.10f}\n")


def save_energy_data(energies_dict, output_dir, title="Energy Convergence"):
    """Save detailed energy comparison data to a log file"""
    log_file = os.path.join(output_dir, 'energy_comparison.log')

    with open(log_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"{title}\n")
        f.write("="*80 + "\n\n")

        # Write iteration-by-iteration data
        # Determine number of iterations
        n_iters = len(next(iter(energies_dict.values())))

        # Header
        f.write(f"{'Iteration':<18}")
        for method_name in energies_dict.keys():
            f.write(f"{method_name.capitalize():<20}")
        f.write("\n")
        f.write("-"*80 + "\n")

        # Data rows
        for i in range(n_iters):
            f.write(f"{i+1:<12}")
            for energies in energies_dict.values():
                f.write(f"{energies[i]:+20.10f}")
            f.write("\n")


def check_potential_depth_warning(madopt_log_path='madopt.log'):
    """
    Check the madopt.log file for warnings about positive Lagrange multipliers.
    """
    warning_lines = []

    if not os.path.exists(madopt_log_path):
        print(f"Warning: madopt.log file not found at {madopt_log_path}")
        return False, []

    try:
        with open(madopt_log_path, 'r') as f:
            for line in f:
                # Check for the specific warning about positive Lagrange multipliers
                if "Warning" in line and "positive" in line and "lagrange" in line.lower() and "multiplier" in line.lower():
                    warning_lines.append(line.strip())
                # Also check for alternative warning formats
                elif "positive lagrange" in line.lower() or "setting to" in line.lower():
                    if "lagrange" in line.lower() or "multiplier" in line.lower():
                        warning_lines.append(line.strip())
    except Exception as e:
        print(f"Error reading madopt.log: {e}")
        return False, []

    has_warning = len(warning_lines) > 0

    if has_warning:
        print("\n" + "="*80)
        print("POTENTIAL DEPTH WARNING DETECTED")
        print("="*80)
        print("\nWarning messages found:")
        for warning in warning_lines:
            print(f"  - {warning}")
        print("="*80 + "\n")

    return has_warning, warning_lines


def recreate_plots_from_json():
    json_file = os.path.join(
        os.path.dirname(__file__),
        'results',
        'single_gaussian_peak',
        'energies_all_methods.json'
    )

    # Output directory (same as JSON file location)
    output_dir = os.path.dirname(json_file)

    # Check if JSON file exists
    if not os.path.exists(json_file):
        print(f"Error: JSON file not found at: {json_file}")

    print(f"\n{'='*80}")
    print("RECREATING PLOTS FROM JSON")
    print(f"{'='*80}\n")

    dir_name = os.path.basename(output_dir)
    title_map = {
        'single_gaussian_peak': 'Energy Convergence: Single Gaussian Peak Potential',
        'three_gaussian_peaks': 'Energy Convergence: Three Gaussian Peaks Potential',
        'helium_potential': 'Energy Convergence: Helium Potential'
    }
    title = title_map.get(dir_name, 'Energy Convergence')

    # Create plots
    plot_energy_comparison_from_json(
        json_file,
        output_dir,
        title=title,
        skip_first_iteration=True
    )

    print(f"\n{'='*80}")
    print("PLOTS SUCCESSFULLY RECREATED")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    recreate_plots_from_json()

