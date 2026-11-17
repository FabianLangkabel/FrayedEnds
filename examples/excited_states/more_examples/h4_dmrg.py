import subprocess as sp
import time

import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

import frayedends as fe

molecule_name = "h4"
n_elec = 4
number_roots = 3
box_size = 50.0
wavelet_order = 7
madness_thresh = 1.0e-4
basisset = "6-31g"

geometry_mode = "equidistant"  # "equidistant" or "h2_pair"
print(f"Geometry mode: {geometry_mode}")

if geometry_mode == "equidistant":  # linear H4 molecule with equidistant spacing d
    distance = np.arange(2.5, 0.45, -0.05).tolist()
elif geometry_mode == "h2_pair":  # for H2 pair getting closer
    distance = np.arange(1.5, 0.2, -0.03).tolist()
else:
    raise ValueError("geometry_mode must be 'equidistant' or 'h2_pair'")

output_file = f"{molecule_name}_{geometry_mode}_nwchem_dmrg.dat"
 
with open(output_file, "w") as f:
    f.write("distance time_s " + " ".join(f"energy_{i}" for i in range(number_roots)) + " energy_SA\n")
 
total_start = time.perf_counter()

for d in distance:
    dist_start = time.perf_counter()
    reported_distance = 2 * d if geometry_mode == "h2_pair" else d

    # Define the H4 geometry for this distance
    if geometry_mode == "equidistant":  # equidistant linear H4 molecule
        z = [-1.5 * d, -0.5 * d, 0.5 * d, 1.5 * d]
    else:  # two H2 molecules (bond length 2.55) getting closer
        z = [-d - 2.55, -d, d, d + 2.55]
    geom = "\n".join(f"    H 0.0 0.0 {zi}" for zi in z)
 
    # Create NWChem Input file
    nwchem_input = f"""
    title "molecule"
    memory stack 1500 mb heap 100 mb global 1400 mb
    charge 0
    geometry units angstrom noautosym nocenter
    {geom}
    end
    basis
    * library {basisset}
    end
    scf
    maxiter 200
    end
    task scf
    """

    with open("nwchem", "w") as f:
        f.write(nwchem_input)
    programm = sp.call(
        "/opt/anaconda3/envs/frayedends/bin/nwchem nwchem",
        stdout=open("nwchem.out", "w"),
        stderr=open("nwchem_err.log", "w"),
        shell=True,
    )

    world = fe.MadWorld(ndims=3, L=box_size, k=wavelet_order, thresh=madness_thresh)

    converter = fe.NWChem_Converter(world)
    converter.read_nwchem_file("nwchem")
    orbs = converter.get_mos()
    Vnuc = converter.get_Vnuc()
    nuclear_repulsion_energy = converter.get_nuclear_repulsion_energy()

    # for i in range(len(orbs)):
    #   world.line_plot(f"initial_orb{i}_d{d}.dat", orbs[i], axis="z", datapoints=2001)

    integrals = fe.Integrals(world)
    G = integrals.compute_two_body_integrals(orbs, ordering="chem").elems
    T = integrals.compute_kinetic_integrals(orbs)
    V = integrals.compute_potential_integrals(orbs, Vnuc)
    h1 = T + V

    n_orbitals = len(orbs)

    driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SU2, n_threads=8)
    driver.initialize_system(n_sites=n_orbitals, n_elec=n_elec, spin=0)

    idx = driver.orbital_reordering(h1, G)
    h1_new = h1[idx][:, idx]
    g2_new = G[idx][:, idx][:, :, idx][:, :, :, idx]

    mpo = driver.get_qc_mpo(h1e=h1_new, g2e=g2_new, ecore=nuclear_repulsion_energy, iprint=0)
    ket = driver.get_random_mps(tag="KET", bond_dim=100, nroots=number_roots)
    energies = driver.dmrg(mpo, ket, n_sweeps=10, bond_dims=[100], noises=[1e-5] * 4 + [0], thrds=[1e-10] * 8, iprint=0)

    kets = [driver.split_mps(ket, ir, tag="KET-%d" % ir) for ir in range(ket.nroots)]
    sa_1pdm = np.mean([driver.get_1pdm(k) for k in kets], axis=0)  # Compute the state average 1-body rdm
    sa_2pdm = np.mean([driver.get_2pdm(k) for k in kets], axis=0).transpose(0, 3, 1, 2)  # Compute the state average 2-body rdm
 
    # State-averaged energy from SA-RDMs
    e_avg = (np.einsum("ij,ij->", sa_1pdm, h1_new)
             + 0.5 * np.einsum("ijkl,ijkl->", sa_2pdm, g2_new)
             + nuclear_repulsion_energy)
 
    dist_time = time.perf_counter() - dist_start
     
    print(f"{'Dist':<8}{'Energies':<{14 * number_roots}}{'E_SA':<16}Time / s")
    print(f"{reported_distance:<8.3f}{''.join(f'{e:<14.8f}' for e in energies)}{e_avg:<16.8f}{dist_time:.1f}")
 
    with open(output_file, "a") as f:
        f.write(f"{reported_distance:.3f} {dist_time:.2f} " + " ".join(f"{x:.15f}" for x in energies) + f" {e_avg:.15f}\n")
 
    del converter
    del integrals
    del world
 
with open(output_file, "a") as f:
    f.write(f"# total time = {time.perf_counter() - total_start:.2f} s\n")
 
print(f"Total time: {time.perf_counter() - total_start:.1f} s")
 
fe.cleanup(globals())
