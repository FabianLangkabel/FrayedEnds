"""
Hydrogen atoms placed along the lines of the FrayedEnds logo (central ring, X with four corner loops, frame).
The Hartree-Fock electron density is computed with MADNESS through FrayedEnds and plotted in the molecular plane.
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import frayedends as fe

# logo geometry in units of its half width, measured from the logo
R, YC, M = 0.502, 0.585, 0.597  # ring radius, crossing point and slope of the X lines (tangent to the ring)
C, F = 0.616, 0.905  # corner arc center (+-C, +-C) and frame position


def logo_geometry(scale=10.0, spacing=1.1, frame_end=0.32, dmin=0.7):
    """hydrogen positions (angstrom) along the logo with half width `scale` and atom distance ~`spacing`"""
    d = spacing / scale

    def line(a, b):
        n = max(1, round(np.linalg.norm(b - a) / d))
        return a + np.linspace(0, 1, n + 1)[:, None] * (b - a)

    def arc(center, r, t0, t1, n=None, offset=0.0):
        n = n or max(1, round(r * np.radians(t1 - t0) / d))
        t = np.radians(t0 + (t1 - t0) * (np.arange(n + 1) + offset) / n)
        return center + r * np.stack([np.cos(t), np.sin(t)], 1)

    def c4(p):  # four copies rotated by 90 degrees
        return np.vstack([p @ np.linalg.matrix_power([[0, 1], [-1, 0]], k) for k in range(4)])

    r_arc = (YC - (1 - M) * C) / np.hypot(1, M)  # corner arc tangent to the X lines
    t0 = np.degrees(np.arctan(1 / M))  # arc angle where the X line touches
    Q = np.array([-C, C]) + r_arc * np.array([np.cos(np.radians(t0)), np.sin(np.radians(t0))])
    P = np.array([0.0, YC])
    n_ring = 4 * round(2 * np.pi * R / d / 4)
    curves = [
        arc(np.zeros(2), R, 0, 360, n_ring, 0.5)[:-1],  # ring, no atom directly below the crossings
        c4(line(P, Q)),  # X lines from the crossing to the corner arcs
        c4(line(P, Q * [-1, 1])),
        c4(arc(np.array([-C, C]), r_arc, t0, 270 - t0)),  # corner arcs
        c4(line(np.array([-C, F]), np.array([frame_end, F]))),  # frame, open towards the next corner
    ]
    atoms = []
    for p in np.vstack(curves):  # skip points that coincide with atoms of previous curves
        if all(np.linalg.norm(p - q) > dmin * d for q in atoms):
            atoms.append(p)
    return scale * np.array(atoms)


class CheapSCF(fe.AtomicBasisProjector):
    # a loosely converged density at thresh 1e-4 is enough for a plot (default: 1e-4 and 1e-6, up to 25 iterations)
    def parameter_string(self, *args, **kwargs):
        return super().parameter_string(*args, **kwargs)[:-1] + '; protocol=[1.e-4]; maxiter=10"'


def remove_madness_files():  # otherwise MADNESS restarts from the orbitals of a previous run
    for f in glob.glob("mad.*"):
        os.remove(f)


scale = 10.0
xy = logo_geometry(scale)
geometry = "\n".join(f"H {x:.6f} {y:.6f} 0.0" for x, y in xy)
print(f"{len(xy)} hydrogen atoms")

remove_madness_files()
world = fe.MadWorld(ndims=3, L=30.0, k=6, thresh=1e-4)
orbitals = CheapSCF(world, geometry, units="angstrom").solve_scf(thresh=1e-4)
remove_madness_files()
rho = fe.Integrals(world).compute_electron_density([], orbitals, 2.0 * np.eye(len(orbitals)))
rho.save_to_file("hydrogen_logo_density.mad")

world.cube_plot("logo_density", rho, fe.MolecularGeometry(geometry, units="angstrom"), zoom=1.0, datapoints=201)
# density in the molecular plane
n = 401
g = np.linspace(-1.05 * scale, 1.05 * scale, n)
X, Y = np.meshgrid(g, g)
points = np.stack([X.ravel(), Y.ravel(), np.zeros(n * n)], 1).ravel().tolist()
density = np.array(world.evaluate(rho, points, units="angstrom")).reshape(n, n)
np.save("hydrogen_logo_density.npy", density)

cmap = LinearSegmentedColormap.from_list("logo", ["white", "#3a1b6e"])
fig, ax = plt.subplots(figsize=(8, 8))
log_density = np.log10(np.maximum(density, 1e-8))  # clip small negative values from the MRA representation
ax.imshow(log_density, origin="lower", extent=[g[0], g[-1], g[0], g[-1]], cmap=cmap, vmin=-1.4, vmax=-1.0)
ax.set_axis_off()
fig.savefig("hydrogen_logo_density.png", dpi=200, bbox_inches="tight")

fe.cleanup(globals())
