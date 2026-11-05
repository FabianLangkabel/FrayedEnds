"""
Stop motion animation of the lowest orbitals written by hydrogen_logo_orbitals.py (run that first): each orbital
is shown as an isosurface, ray traced from its cube file, for a few frames, while the view turns from the side (molecule
horizontal, z axis up) to the top view of hydrogen_logo_orbitals.png (z axis pointing towards the viewer).
Writes the frames to hydrogen_logo_orbitals_animation/ and the animation to hydrogen_logo_orbitals.gif.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates, spline_filter

ISO = 0.014  # isovalue in bohr^-3/2
PIXELS, HALF_WIDTH = 900, 27.0  # image size and half width of the field of view (bohr)
ELEVATION, AZIMUTH = 8.0, -45.0  # view angles (degrees) of the first frame, the last one is the top view
EASE_IN = 0.2  # fraction of the animation in which the motion speeds up
POSITIVE, NEGATIVE, ATOM = np.array([[74, 166, 58], [106, 63, 176], [235, 235, 235]]) / 255
ATOM_RADIUS = 0.45  # bohr
ORBITALS, FRAMES_PER_ORBITAL = 36, 3


def read_cube(filename):
    with open(filename) as f:
        lines = f.readlines()
    n = int(lines[2].split()[0])
    origin = np.array(lines[2].split()[1:4], float)
    shape = [int(lines[3 + i].split()[0]) for i in range(3)]
    step = float(lines[3].split()[1])  # cubic grid
    atoms = np.array([line.split()[2:5] for line in lines[6 : 6 + n]], float)
    return np.array(" ".join(lines[6 + n :]).split(), float).reshape(shape), origin, step, atoms


def camera(t):
    """rows: screen right, screen up and towards the viewer, turning from the side view to the top view"""
    w = (t**2 / EASE_IN if t < EASE_IN else 2 * t - EASE_IN) / (2 - EASE_IN)  # accelerate, then constant speed
    e, a = np.radians(ELEVATION + (90 - ELEVATION) * w), np.radians(AZIMUTH * (1 - w))
    view = np.array([[1, 0, 0], [0, np.sin(e), np.cos(e)], [0, -np.cos(e), np.sin(e)]])
    return view @ np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def shade(color, normal, cam):
    light = np.array([-0.4, 0.6, 1.0]) @ cam / np.linalg.norm([-0.4, 0.6, 1.0])  # from the upper left
    half = (light + cam[2]) / np.linalg.norm(light + cam[2])
    diffuse = np.clip(normal @ light, 0, 1)[:, None]
    specular = np.clip(normal @ half, 0, 1)[:, None] ** 40
    return np.clip(color * (0.35 + 0.65 * diffuse) + 0.3 * specular, 0, 1)


def render(psi, origin, step, atoms, cam, ds=0.15):
    """ray traced isosurfaces |psi| = ISO (orthographic view) and atoms as spheres on a white background"""
    coeffs = spline_filter(psi, order=3)

    def value(points):  # cubic spline interpolation of psi at points (bohr)
        return map_coordinates(coeffs, ((points - origin) / step).T, order=3, prefilter=False)

    g = np.linspace(-HALF_WIDTH, HALF_WIDTH, PIXELS)
    U, V = np.meshgrid(g, -g)
    start = np.stack([U.ravel(), V.ravel()], 1) @ cam[:2]  # ray positions in the screen plane

    # rays run from the viewer through the slab of the molecule, |x|, |y| < extent and |z| < 4 bohr
    extent = np.abs(atoms).max(0) + [4, 4, 4]
    with np.errstate(divide="ignore", invalid="ignore"):
        s0, s1 = (-extent - start) / cam[2], (extent - start) / cam[2]
    s_in = np.nan_to_num(np.maximum(s0, s1), nan=np.inf, posinf=np.inf, neginf=-np.inf).min(1)
    s_out = np.nan_to_num(np.minimum(s0, s1), nan=-np.inf, posinf=np.inf, neginf=-np.inf).max(1)
    depth = np.full(len(start), -np.inf)  # distance of the visible surface towards the viewer
    color = np.ones((len(start), 3))
    active = np.nonzero(s_in > s_out)[0]
    s, f = s_in[active], np.full(len(active), -ISO)
    while len(active):
        f_old, s = f, s - ds
        f = np.abs(value(start[active] + s[:, None] * cam[2])) - ISO
        hit = f >= 0
        if hit.any():  # linear interpolation of the crossing and normals from the spline gradient
            i, sh = active[hit], s[hit] + ds * f[hit] / (f[hit] - f_old[hit])
            p = start[i] + sh[:, None] * cam[2]
            grad = np.stack([value(p + h) - value(p - h) for h in 0.05 * np.eye(3)], 1)
            sign = np.sign(value(p))
            normal = -sign[:, None] * grad / np.linalg.norm(grad, axis=1, keepdims=True)
            depth[i] = sh
            color[i] = shade(np.where(sign[:, None] > 0, POSITIVE, NEGATIVE), normal, cam)
        keep = ~hit & (s > s_out[active])
        active, s, f = active[keep], s[keep], f[keep]

    for a in atoms @ cam.T:  # spheres in front of the isosurfaces
        d2 = (U.ravel() - a[0]) ** 2 + (V.ravel() - a[1]) ** 2
        i = np.nonzero(d2 < ATOM_RADIUS**2)[0]
        h = np.sqrt(ATOM_RADIUS**2 - d2[i])
        i, h = i[a[2] + h > depth[i]], h[a[2] + h > depth[i]]
        normal = np.stack([U.ravel()[i] - a[0], V.ravel()[i] - a[1], h], 1) @ cam / ATOM_RADIUS
        depth[i] = a[2] + h
        color[i] = shade(ATOM, normal, cam)
    return color.reshape(PIXELS, PIXELS, 3)


data = np.load("hydrogen_logo_orbitals.npz")
energies, irreps = data["energies"], data["irreps"]
os.makedirs("hydrogen_logo_orbitals_animation", exist_ok=True)
frames = []
for k in range(ORBITALS * FRAMES_PER_ORBITAL):
    i = k // FRAMES_PER_ORBITAL
    if k % FRAMES_PER_ORBITAL == 0:
        psi, origin, step, atoms = read_cube(f"hydrogen_logo_orbitals/orbital_{i:02d}.cube")
    cam = camera(k / (ORBITALS * FRAMES_PER_ORBITAL - 1))
    fig = plt.figure(figsize=(9, 9), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(render(psi, origin, step, atoms, cam), extent=[0, 1, 0, 1])
    ax.set_axis_off()
    for axis, label, c in zip(cam[:2].T, "xyz", ["#c0392b", "#27ae60", "#2c6fbb"]):  # axes in the lower left
        ax.annotate(
            "",
            (0.08 + 0.05 * axis[0], 0.08 + 0.05 * axis[1]),
            (0.08, 0.08),
            arrowprops=dict(arrowstyle="-|>", color=c, lw=1.5),
        )
        ax.text(0.08 + 0.065 * axis[0], 0.08 + 0.065 * axis[1], label, color=c, ha="center", va="center", fontsize=11)
    ax.text(0.97, 0.03, f"orbital {i + 1}   {energies[i]:.3f} Eh ({irreps[i]})", ha="right", fontsize=11, color="0.3")
    fig.savefig(f"hydrogen_logo_orbitals_animation/frame_{k:03d}.png")
    plt.close(fig)
    frames.append(Image.open(f"hydrogen_logo_orbitals_animation/frame_{k:03d}.png").convert("RGB"))
    print(f"frame {k + 1}/{ORBITALS * FRAMES_PER_ORBITAL}", flush=True)

durations = [150] * (len(frames) - 1) + [2000]  # hold the top view at the end
frames[0].save("hydrogen_logo_orbitals.gif", save_all=True, append_images=frames[1:], duration=durations, loop=0)
