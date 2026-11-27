import time

import numpy as np
from pyscf import fci

import frayedends as fe

molecule_name = "h4"
n_elec = 4  # Number of electrons
number_roots = 3  # Number of states (groundstate, 1. excited state, 2. excited state)
box_size = 50.0  # Size of the simulation box
wavelet_order = 7  # Order of wavelet basis functions
madness_thresh = 1.0e-5  # Threshold for numerical precision of function representation

# Create .dat files for the results
output_file = f"{molecule_name}_cispd_pno_fci.dat"

with open(output_file, "w") as f:
    f.write("time_s n_orbitals " + " ".join(f"energy_{i}" for i in range(number_roots)) + " energy_SA\n")

total_start = time.perf_counter()

# Define a linear H4 molecule geometry with 1.0 Angstrom spacing between adjacent atoms
molecule = fe.MolecularGeometry(units="angstrom")
molecule.add_atom(0.0, 0.0, -1.5, "H")
molecule.add_atom(0.0, 0.0, -0.5, "H")
molecule.add_atom(0.0, 0.0, 0.5, "H")
molecule.add_atom(0.0, 0.0, 1.5, "H")
geom = "H 0.0 0.0 -1.5\nH 0.0 0.0 -0.5\nH 0.0 0.0 0.5\nH 0.0 0.0 1.5"

# Setting up the numerical environment for the MRA calculations
world = fe.MadWorld(ndims=3, L=box_size, k=wavelet_order, thresh=madness_thresh)
integrals = fe.Integrals(world)

# Calculate ground state orbitals
pno_start = time.perf_counter()
madpno = fe.MadPNO(world, geom, units="angstrom", n_orbitals=4)  # 2 HF + 2 MP2-PNOs
pno_end = time.perf_counter()
pno_time = pno_end - pno_start
print("Generating PNOs took %.2f seconds" % pno_time)

gs_orbs = madpno.get_orbitals()
hf_orbs = madpno.get_hf_orbitals()

# for i in range(len(gs_orbs)):
#     world.cube_plot(f"gs_orb{i}", gs_orbs[i], molecule, zoom=4.0)
#
# for i in range(len(hf_orbs)):
#     world.cube_plot(f"hf_orb{i}", hf_orbs[i], molecule, zoom=4.0)

# Calculate excited states orbitals
cis_start = time.perf_counter()
cis_orbs = madpno.compute_cis(n_excitation=2)  # Compute CIS for 2 excitations (1st and 2nd excited states)
cis_orbs = integrals.project_out(gs_orbs, cis_orbs)
cis_orbs = integrals.orthonormalize(cis_orbs)
cis_end = time.perf_counter()
cis_time = cis_end - cis_start
print("Generating CIS took %.2f seconds" % cis_time)

# for i in range(len(cis_orbs)):
#     world.cube_plot(f"cis_orb{i}", cis_orbs[i], molecule, zoom=4.0)

cispd_start = time.perf_counter()
cispd_orbs = madpno.compute_cispd(n_orbitals=4)  # 2 CIS X function + 2 CIS(D)-PNO PER EXCITATION
cispd_orbs = integrals.project_out(gs_orbs + cis_orbs, cispd_orbs)
cispd_end = time.perf_counter()
cispd_time = cispd_end - cispd_start
print("Generating CISPD took %.2f seconds" % cispd_time)

# for i in range(len(cispd_orbs)):
#     world.cube_plot(f"cispd_orb{i}", cispd_orbs[i], molecule, zoom=4.0)

nuclear_repulsion_energy = madpno.get_nuclear_repulsion()
Vnuc = madpno.get_nuclear_potential()

orbs = gs_orbs + cis_orbs + cispd_orbs
orbs = integrals.orthonormalize(orbitals=orbs)

# for i in range(len(orbs)):
#     world.cube_plot(f"orb{i}", orbs[i], molecule, zoom=4.0)

# Calculate initial integrals
T = integrals.compute_kinetic_integrals(orbs)
V = integrals.compute_potential_integrals(orbs, Vnuc)
h1 = T + V
G = integrals.compute_two_body_integrals(orbs, ordering="chem").elems
n_orbitals = len(orbs)

# FCI calculation
cisolver = fci.direct_spin1.FCI()
cisolver = fci.addons.fix_spin_(cisolver, shift=1.0, ss=0)
e, fcivec = cisolver.kernel(h1, G, n_orbitals, n_elec, nroots=number_roots)
energies = e + nuclear_repulsion_energy

rdms = [cisolver.make_rdm12(c, n_orbitals, n_elec) for c in fcivec]
sa_1pdm = np.mean([r[0] for r in rdms], axis=0)  # Compute the state average 1-body rdm
sa_2pdm = np.mean([r[1] for r in rdms], axis=0)  # Compute the state average 2-body rdm

# State-averaged energy from SA-RDMs
e_avg = np.einsum("ij,ij->", sa_1pdm, h1) + 0.5 * np.einsum("ijkl,ijkl->", sa_2pdm, G) + nuclear_repulsion_energy

total_time = time.perf_counter() - total_start

print(f"{'Energies':<{14 * number_roots}}{'E_SA':<16}Time / s")
print(f"{''.join(f'{e:<14.8f}' for e in energies)}{e_avg:<16.8f}{total_time:.1f}")

with open(output_file, "a") as f:
    f.write(f"{total_time:.2f} {n_orbitals} " + " ".join(f"{x:.15f}" for x in energies) + f" {e_avg:.15f}\n")
    f.write(f"# orbitals: {len(gs_orbs)} GS + {len(cis_orbs)} CIS + {len(cispd_orbs)} CIS(D) -> {n_orbitals}\n")
    f.write(f"# orbital generation time: PNO {pno_time:.2f} s, CIS {cis_time:.2f} s, CIS(D) {cispd_time:.2f} s\n")

fe.cleanup(globals())
