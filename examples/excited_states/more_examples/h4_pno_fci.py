import time

import numpy as np
from pyscf import fci

import frayedends as fe

molecule_name = "h4"
n_elec = 4
number_roots = 3
box_size = 50.0
wavelet_order = 7
madness_thresh = 1.0e-4

geometry_mode = "equidistant"  # "equidistant" or "h2_pair"
print(f"Geometry mode: {geometry_mode}")

if geometry_mode == "equidistant":  # linear H4 molecule with equidistant spacing d
    distance = np.arange(2.5, 0.45, -0.05).tolist()
elif geometry_mode == "h2_pair":  # for H2 pair getting closer
    distance = np.arange(1.5, 0.2, -0.03).tolist()
else:
    raise ValueError("geometry_mode must be 'equidistant' or 'h2_pair'")

output_file = f"{molecule_name}_{geometry_mode}_pno_fci.dat"
 
with open(output_file, "w") as f:
    f.write("distance time_s " + " ".join(f"energy_{i}" for i in range(number_roots)) + " energy_SA\n")
 
total_start = time.perf_counter()

for d in distance:
    dist_start = time.perf_counter()
    reported_distance = 2 * d if geometry_mode == "h2_pair" else d

    if geometry_mode == "equidistant":  # equidistant linear H4 molecule
        z = [-1.5 * d, -0.5 * d, 0.5 * d, 1.5 * d]
    else:  # two H2 molecules getting closer
        z = [-d - 2.55, -d, d, d + 2.55]
    geom = "".join(f"H 0.0 0.0 {zi}\n" for zi in z)

    world = fe.MadWorld(ndims=3, L=box_size, k=wavelet_order, thresh=madness_thresh)

    madpno = fe.MadPNO(world, geom, units="angstrom", n_orbitals=8, maxrank=8)
    orbs = madpno.get_orbitals()

    nuclear_repulsion_energy = madpno.get_nuclear_repulsion()
    Vnuc = madpno.get_nuclear_potential()

    integrals = fe.Integrals(world)
    orbs = integrals.orthonormalize(orbitals=orbs)

    # for i in range(len(orbs)):
    #    world.line_plot(f"orb{i}_d{d}.dat", orbs[i], axis="z", datapoints=2001)

    """
    Calculate initial integrals
    """
    integrals = fe.Integrals(world)
    G = integrals.compute_two_body_integrals(orbs, ordering="chem").elems
    T = integrals.compute_kinetic_integrals(orbs)
    V = integrals.compute_potential_integrals(orbs, Vnuc)
    h1 = T + V

    """
    Performe SA DMRG calculation and extract rdms
    """

    n_orbitals = len(orbs)

    cisolver = fci.direct_spin1.FCI()
    cisolver = fci.addons.fix_spin_(cisolver, ss=0)
    e, fcivec = cisolver.kernel(h1, G, n_orbitals, n_elec, nroots=number_roots)
    energies = e + nuclear_repulsion_energy

    rdms = [cisolver.make_rdm12(c, n_orbitals, n_elec) for c in fcivec]
    sa_1pdm = np.mean([r[0] for r in rdms], axis=0)  # Compute the state average 1-body rdm
    sa_2pdm = np.mean([r[1] for r in rdms], axis=0)  # Compute the state average 2-body rdm

    # State-averaged energy from SA-RDMs
    e_avg = (np.einsum("ij,ij->", sa_1pdm, h1)
            + 0.5 * np.einsum("ijkl,ijkl->", sa_2pdm, G)
            + nuclear_repulsion_energy)
    
    dist_time = time.perf_counter() - dist_start
    
    print(f"{'Dist':<8}{'Energies':<{14 * number_roots}}{'E_SA':<16}Time / s")
    print(f"{reported_distance:<8.3f}{''.join(f'{e:<14.8f}' for e in energies)}{e_avg:<16.8f}{dist_time:.1f}")

    with open(output_file, "a") as f:
        f.write(f"{reported_distance:.3f} {dist_time:.2f} " + " ".join(f"{x:.15f}" for x in energies) + f" {e_avg:.15f}\n")

    del integrals
    del madpno
    del world

with open(output_file, "a") as f:
    f.write(f"# total time = {time.perf_counter() - total_start:.2f} s\n")

print(f"Total time: {time.perf_counter() - total_start:.1f} s")

fe.cleanup(globals())