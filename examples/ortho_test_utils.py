import os
import numpy as np
from matplotlib import pyplot as plt


def plot_orbitals_2d(orbitals, world, iteration, output_dir, method_name="mixed",
                     n_points=101, zoom_threshold=0.01):
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

        if np.any(significant_mask):
            y_indices, x_indices = np.where(significant_mask)
            x_min_idx, x_max_idx = x_indices.min(), x_indices.max()
            y_min_idx, y_max_idx = y_indices.min(), y_indices.max()

            x_range = x_max_idx - x_min_idx
            y_range = y_max_idx - y_min_idx
            x_pad = max(int(x_range * 0.1), 5)
            y_pad = max(int(y_range * 0.1), 5)

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
                                phase="normal", zoom_threshold=0.01):
    plot_dir = os.path.join(output_dir, method_name)
    os.makedirs(plot_dir, exist_ok=True)

    n_orbs_before = len(orbitals_before)
    n_orbs_after = len(orbitals_after)
    n_orbs_max = max(n_orbs_before, n_orbs_after)

    fig, axes = plt.subplots(2, n_orbs_max, figsize=(4*n_orbs_max, 8))
    if n_orbs_max == 1:
        axes = axes.reshape(2, 1)

    temp_files_before = []
    for i, orb in enumerate(orbitals_before):
        temp_file = f'_temp_before_{i}_iter_{iteration}_{phase}_{method_name}.dat'
        temp_files_before.append(temp_file)
        world.plane_plot(temp_file, orb, datapoints=n_points)

    temp_files_after = []
    for i, orb in enumerate(orbitals_after):
        temp_file = f'_temp_after_{i}_iter_{iteration}_{phase}_{method_name}.dat'
        temp_files_after.append(temp_file)
        world.plane_plot(temp_file, orb, datapoints=n_points)

    for i, temp_file in enumerate(temp_files_before):
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

        if np.any(significant_mask):
            y_indices, x_indices = np.where(significant_mask)
            x_min_idx, x_max_idx = x_indices.min(), x_indices.max()
            y_min_idx, y_max_idx = y_indices.min(), y_indices.max()

            x_range = x_max_idx - x_min_idx
            y_range = y_max_idx - y_min_idx
            x_pad = max(int(x_range * 0.1), 5)
            y_pad = max(int(y_range * 0.1), 5)

            x_min_idx = max(0, x_min_idx - x_pad)
            x_max_idx = min(len(x_unique) - 1, x_max_idx + x_pad)
            y_min_idx = max(0, y_min_idx - y_pad)
            y_max_idx = min(len(y_unique) - 1, y_max_idx + y_pad)

            axes[0, i].set_xlim(x_unique[x_min_idx], x_unique[x_max_idx])
            axes[0, i].set_ylim(y_unique[y_min_idx], y_unique[y_max_idx])

        vmax = np.max(np.abs(Z))
        im = axes[0, i].contourf(X, Y, Z, levels=20, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[0, i].set_title(f'Orbital {i} (Before)\nmax: {z_max:.2e}')
        axes[0, i].set_xlabel('x')
        axes[0, i].set_ylabel('y')
        axes[0, i].set_aspect('equal')
        plt.colorbar(im, ax=axes[0, i], format='%.1e')

        os.remove(actual_file)

    for i in range(n_orbs_before, n_orbs_max):
        axes[0, i].axis('off')

    for i, temp_file in enumerate(temp_files_after):
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

        if np.any(significant_mask):
            y_indices, x_indices = np.where(significant_mask)
            x_min_idx, x_max_idx = x_indices.min(), x_indices.max()
            y_min_idx, y_max_idx = y_indices.min(), y_indices.max()

            x_range = x_max_idx - x_min_idx
            y_range = y_max_idx - y_min_idx
            x_pad = max(int(x_range * 0.1), 5)
            y_pad = max(int(y_range * 0.1), 5)

            x_min_idx = max(0, x_min_idx - x_pad)
            x_max_idx = min(len(x_unique) - 1, x_max_idx + x_pad)
            y_min_idx = max(0, y_min_idx - y_pad)
            y_max_idx = min(len(y_unique) - 1, y_max_idx + y_pad)

            axes[1, i].set_xlim(x_unique[x_min_idx], x_unique[x_max_idx])
            axes[1, i].set_ylim(y_unique[y_min_idx], y_unique[y_max_idx])

        vmax = np.max(np.abs(Z))
        im = axes[1, i].contourf(X, Y, Z, levels=20, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        if i < n_orbs_before:
            axes[1, i].set_title(f'Orbital {i} (After)\nmax: {z_max:.2e}')
        else:
            axes[1, i].set_title(f'Orbital {i} (NEW)\nmax: {z_max:.2e}', color='red')
        axes[1, i].set_xlabel('x')
        axes[1, i].set_ylabel('y')
        axes[1, i].set_aspect('equal')
        plt.colorbar(im, ax=axes[1, i], format='%.1e')

        os.remove(actual_file)

    for i in range(n_orbs_after, n_orbs_max):
        axes[1, i].axis('off')

    plt.suptitle(f'Iteration {iteration} - {method_name} - {phase}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'orbitals_iter_{iteration}_{phase}.png'), dpi=150)
    plt.close()


def plot_energy_comparison(energies_dict, output_dir, title="Energy Convergence",
                           subtitle="Comparison of Orthonormalization Methods",
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
    plt.ylabel('Energy (Hartree)', fontsize=12)

    full_title = f'{title}\n{subtitle}' if subtitle else title
    plt.title(full_title, fontsize=13)

    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'energy_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def print_summary(energies_dict, output_dir):
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)

    n_iters = len(next(iter(energies_dict.values())))
    print(f"Final energies after {n_iters} iterations:")

    for method_name, energies in energies_dict.items():
        print(f"  {method_name.capitalize():12s}: {energies[-1]:+2.8f} Hartree")

    if len(energies_dict) > 1:
        print(f"\nEnergy differences:")
        methods = list(energies_dict.keys())
        for i in range(len(methods)):
            for j in range(i+1, len(methods)):
                method1, method2 = methods[i], methods[j]
                diff = abs(energies_dict[method1][-1] - energies_dict[method2][-1])
                print(f"  |{method1.capitalize()} - {method2.capitalize()}|:  {diff:.2e} Hartree")

    print(f"\nComparison plot saved to: {os.path.join(output_dir, 'energy_comparison.png')}")
    print("="*80 + "\n")

