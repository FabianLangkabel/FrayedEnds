import os
import numpy as np
from matplotlib import pyplot as plt


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


def plot_energy_comparison(energies_dict, output_dir, title="Energy Convergence",
                           vline_at=None, vline_label=None):
    plt.figure(figsize=(12, 7))

    markers = {'mixed': 'o', 'symmetric': 's', 'cholesky': '^'}

    for method_name, energies in energies_dict.items():
        marker = markers.get(method_name.lower(), 'o')
        plt.plot(energies, marker=marker, linestyle='-',
                label=method_name.capitalize(), linewidth=2, markersize=8)

    if vline_at is not None:
        plt.axvline(x=vline_at, color='red', linestyle='--', linewidth=2,
                   label=vline_label or 'Event', alpha=0.7)

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Energy', fontsize=12)


    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'energy_comparison.png'),
                dpi=150, bbox_inches='tight')
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

