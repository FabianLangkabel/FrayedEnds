import time

import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

import frayedends as fe

molecule_name = "h2"
n_elec = 2 # Number of electrons
number_roots = 3 # Number of states (groundstate, 1. excited state, 2. excited state)
n_excitations = 2 # Number of excitations
iterations = 15 # Number of iterations
box_size = 50.0 # Size of the simulation box
wavelet_order = 7 # Order of wavelet basis functions
madness_thresh = 1.0e-6 # Threshold for numerical precision of function representation
econv = 1.0e-4 # Energy convergence threshold (state-averaged energy)

# Create .dat file for the results
output_file = f"{molecule_name}_cispd_pno_dmrg_oo.dat"
 
with open(output_file, "w") as f:
    f.write("iteration iteration_time_s " + " ".join(f"energy_{i}" for i in range(number_roots)) + " energy_SA\n")
 
total_start = time.perf_counter()

# Define a linear H2 molecule geometry with 1.0 Angstrom spacing
molecule = fe.MolecularGeometry(units="angstrom")
molecule.add_atom(0.0, 0.0, -0.5, "H")
molecule.add_atom(0.0, 0.0, 0.5, "H")
geom = "H 0.0 0.0 -0.5\nH 0.0 0.0 0.5"

# Setting up the numerical environment for the MRA calculations
world = fe.MadWorld(ndims=3, L=box_size, k=wavelet_order, thresh=madness_thresh)
integrals = fe.Integrals(world)

# Calculate ground state orbitals
pno_start = time.perf_counter()
madpno = fe.MadPNO(world, geom, units="angstrom", n_orbitals=4)  # 1 HF + 3 MP2-PNOs
pno_time = time.perf_counter() - pno_start
print("Generating PNOs took %.2f seconds" % pno_time)

gs_orbs = madpno.get_orbitals()
hf_orbs = madpno.get_hf_orbitals()

# for i in range(len(gs_orbs)):
#     world.cube_plot(f"gs_orb{i}", gs_orbs[i], molecule, zoom=4.0)

# Calculate excited states orbitals
cis_start = time.perf_counter()
cis_orbs = madpno.compute_cis(n_excitation=2)  # Compute CIS for 2 excitations (1st and 2nd excited states)
cis_orbs = integrals.project_out(gs_orbs, cis_orbs)
cis_orbs = integrals.orthonormalize(cis_orbs)
cis_time = time.perf_counter() - cis_start
print("Generating CIS took %.2f seconds" % cis_time)

# for i in range(len(cis_orbs)):
#     world.cube_plot(f"cis_orb{i}", cis_orbs[i], molecule, zoom=4.0)

cispd_start = time.perf_counter()
cispd_orbs = madpno.compute_cispd(n_orbitals=4)  # 1 CIS X function + 3 CIS(D)-PNO PER EXCITATION
cispd_orbs = integrals.project_out(gs_orbs + cis_orbs, cispd_orbs)
cispd_time = time.perf_counter() - cispd_start
print("Generating CISPD PNOs took %.2f seconds" % cispd_time)

# for i in range(len(cispd_orbs)):
#     world.cube_plot(f"cispd_orb{i}", cispd_orbs[i], molecule, zoom=4.0)

nuc_repulsion = madpno.get_nuclear_repulsion()
Vnuc = madpno.get_nuclear_potential()

orbs = gs_orbs + cis_orbs + cispd_orbs
orbs = integrals.orthonormalize(orbitals=orbs)

# for i in range(len(orbs)):
#     world.cube_plot(f"orb{i}", orbs[i], molecule, zoom=4.0)

n_orbitals = len(orbs)

with open(output_file, "a") as f:
    f.write(f"# orbitals: {len(gs_orbs)} GS + {len(cis_orbs)} CIS + {len(cispd_orbs)} CIS(D) -> {n_orbitals}\n")
    f.write(f"# orbital generation time: PNO {pno_time:.2f} s, CIS {cis_time:.2f} s, CIS(D) {cispd_time:.2f} s\n")

# Calculate initial integrals
T = integrals.compute_kinetic_integrals(orbs)
V = integrals.compute_potential_integrals(orbs, Vnuc)
h1 = T + V
G = integrals.compute_two_body_integrals(orbs, ordering="chem").elems

# DMRG calculation and extract RDMs
driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SU2, n_threads=8)
driver.initialize_system(n_sites=n_orbitals, n_elec=n_elec, spin=0)

idx = driver.orbital_reordering(h1, G)
h1_new = h1[idx][:, idx]
g2_new = G[idx][:, idx][:, :, idx][:, :, :, idx]

mpo = driver.get_qc_mpo(h1e=h1_new, g2e=g2_new, ecore=nuc_repulsion, iprint=0)
ket = driver.get_random_mps(tag="KET", bond_dim=100, nroots=number_roots)
energies = driver.dmrg(mpo, ket, n_sweeps=10, bond_dims=[100], noises=[1e-5] * 4 + [0], thrds=[1e-10] * 8, iprint=0)

kets = [driver.split_mps(ket, ir, tag="KET-%d" % ir) for ir in range(ket.nroots)]
sa_1pdm = np.mean([driver.get_1pdm(k) for k in kets], axis=0)
sa_2pdm = np.mean([driver.get_2pdm(k) for k in kets], axis=0).transpose(0, 3, 1, 2)

# State-averaged energy from SA-RDMs
e_avg = (np.einsum("ij,ij->", sa_1pdm, h1_new)
         + 0.5 * np.einsum("ijkl,ijkl->", sa_2pdm, g2_new)
         + nuc_repulsion)

idx_back = np.argsort(idx)
sa_1pdm = sa_1pdm[idx_back][:, idx_back]
sa_2pdm = sa_2pdm[idx_back][:, idx_back][:, :, idx_back][:, :, :, idx_back]
sa_2pdm_phys = sa_2pdm.swapaxes(1, 2)  # Physics Notation

# Natural orbitals of the initial state-averaged 1-RDM
natural_orbs, occ_n, vec = integrals.transform_to_natural_orbitals(orbs, sa_1pdm)
# print(f"Natural orbital occupations: {' '.join(f'{x:.6f}' for x in occ_n)}")

# for i in range(len(natural_orbs)):
#     world.cube_plot(f"nat_orb_it0_orb{i}", natural_orbs[i], molecule, zoom=4.0)
 
with open(output_file, "a") as f:
    f.write("0 0.00 " + " ".join(f"{x:.15f}" for x in energies) + f" {e_avg:.15f}\n")
 
print(f"{'Iter':<6}{'Energies':<{14 * number_roots}}{'E_SA':<16}dE_SA")
print(f"{0:<6}{''.join(f'{e:<14.8f}' for e in energies)}{e_avg:<16.8f}-")
 
current = e_avg
converged = False

for iter in range(iterations):
    iter_start = time.perf_counter()

    # Orbital Refinement
    opti = fe.OrbitalRefinement(world, Vnuc, nuc_repulsion)
    orbs = opti.get_orbitals(orbitals=orbs, rdm1=sa_1pdm, rdm2=sa_2pdm_phys, opt_thresh=1.0e-5, occ_thresh=1.0e-5)

    # for i in range(n_orbitals):
    #     world.cube_plot(f"iter{iter}_orb{i}", orbs[i], molecule, zoom=4.0)

    # DMRG calculation with refined orbitals
    G = integrals.compute_two_body_integrals(orbs, ordering="chem").elems  # g-tensor (electron-electron interaction)
    T = integrals.compute_kinetic_integrals(orbs)  # Kinetic energy
    V = integrals.compute_potential_integrals(orbs, Vnuc)  # Potential energy
    h1 = T + V

    driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SU2, n_threads=8)
    driver.initialize_system(n_sites=n_orbitals, n_elec=n_elec, spin=0)

    idx = driver.orbital_reordering(h1, G)
    h1_new = h1[idx][:, idx]
    g2_new = G[idx][:, idx][:, :, idx][:, :, :, idx]

    mpo = driver.get_qc_mpo(h1e=h1_new, g2e=g2_new, ecore=nuc_repulsion, iprint=0)
    ket = driver.get_random_mps(tag="KET", bond_dim=100, nroots=number_roots)
    energies = driver.dmrg(mpo, ket, n_sweeps=10, bond_dims=[100], noises=[1e-5] * 4 + [0], thrds=[1e-10] * 8, iprint=0)

    kets = [driver.split_mps(ket, ir, tag="KET-%d" % ir) for ir in range(ket.nroots)]
    sa_1pdm = np.mean([driver.get_1pdm(k) for k in kets], axis=0)  # Compute the state average 1-body rdm
    sa_2pdm = np.mean([driver.get_2pdm(k) for k in kets], axis=0).transpose(0, 3, 1, 2)  # Compute the state average 2-body rdm

    # State-averaged energy from SA-RDMs
    e_avg = (np.einsum("ij,ij->", sa_1pdm, h1_new)
            + 0.5 * np.einsum("ijkl,ijkl->", sa_2pdm, g2_new)
            + nuc_repulsion)

    idx_back = np.argsort(idx)
    sa_1pdm = sa_1pdm[idx_back][:, idx_back]
    sa_2pdm = sa_2pdm[idx_back][:, idx_back][:, :, idx_back][:, :, :, idx_back]
    sa_2pdm_phys = sa_2pdm.swapaxes(1, 2)  # Change to physics Notation

    natural_orbs, occ_n, vec = integrals.transform_to_natural_orbitals(orbs, sa_1pdm)
    # print(f"Natural orbital occupations: {' '.join(f'{x:.6f}' for x in occ_n)}")

    # for i in range(len(natural_orbs)):
    #     world.cube_plot(f"nat_orb_it{iter+1}_orb{i}", natural_orbs[i], molecule, zoom=4.0)

    print(f"{iter + 1:<6}{''.join(f'{e:<14.8f}' for e in energies)}{e_avg:<16.8f}{e_avg - current:+.1e}")
 
    iter_time = time.perf_counter() - iter_start
    with open(output_file, "a") as f:
        f.write(f"{iter + 1} {iter_time:.2f} " + " ".join(f"{x:.15f}" for x in energies) + f" {e_avg:.15f}\n")
 
    if abs(e_avg - current) < econv:
        converged = True
        print(f"Converged after {iter + 1} iterations (|dE_SA| < {econv:.0e})")
        break  # stop as soon as the state-averaged energy changes less than econv
    current = e_avg
 
if not converged:
    print(f"Not converged after {iterations} iterations")
 
with open(output_file, "a") as f:
    status = "converged" if converged else "not converged"
    f.write("# final natural orbital occupations: " + " ".join(f"{x:.10f}" for x in occ_n) + "\n")
    f.write(f"# {status}, econv = {econv:.0e}, total time = {time.perf_counter() - total_start:.2f} s\n")
 
fe.cleanup(globals())
