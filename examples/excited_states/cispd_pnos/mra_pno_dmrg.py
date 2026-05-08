import time
import frayedends as fe

import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

molecule_name = "h2"
n_electrons = 2
number_roots = 3
iterations = 6
box_size = 50.0
wavelet_order = 7
madness_thresh = 0.0001
econv = 1.0e-6

iteration_results = []

# Create .dat files for the results
with open("iteration_pno_dmrg_oo.dat", "w") as f:
    header = "iteration iteration_time_s " + " ".join(f"energy_{i}" for i in range(number_roots))
    f.write(header + "\n")

with open("results_pno_dmrg_oo.dat", "w") as f:
    header = " ".join(f"energy_{i}" for i in range(number_roots))
    f.write(header + "\n")

total_start = time.perf_counter()

molecule = fe.MolecularGeometry(units="angstrom")
molecule.add_atom(0.0, 0.0, -0.5, "H")
molecule.add_atom(0.0, 0.0, 0.5, "H")
geom = "H 0.0 0.0 -0.5\nH 0.0 0.0 0.5"

world = fe.MadWorld3D(L=box_size, k=wavelet_order, thresh=madness_thresh)

pno_start = time.perf_counter()
madpno = fe.MadPNO(world, geom, n_orbitals=4, cispd=1)  # ground state + 2 excited states 
pno_end = time.perf_counter()
pno_time = pno_end - pno_start
print("Generating PNOs took %.2f seconds" % pno_time)

gs_orbs = madpno.get_orbitals()
ex_orbs = madpno.get_ex_orbitals()

for i in range(len(gs_orbs)):
    world.cube_plot(f"initial_gs_orb{i}", gs_orbs[i], molecule, zoom=4.0)

for i in range(len(ex_orbs)):
    world.cube_plot(f"initial_ex_orb{i}", ex_orbs[i], molecule, zoom=4.0)

nuc_repulsion = madpno.get_nuclear_repulsion()
Vnuc = madpno.get_nuclear_potential()

integrals = fe.Integrals3D(world)
combined_orbs = gs_orbs + ex_orbs
combined_orbs = integrals.orthonormalize(orbitals=combined_orbs)

n_gs_orbs = len(gs_orbs)
n_ex_orbs = len(ex_orbs)

n_orbitals = len(combined_orbs)

T = integrals.compute_kinetic_integrals(combined_orbs)
V = integrals.compute_potential_integrals(combined_orbs, Vnuc)
h1 = T + V
G = integrals.compute_two_body_integrals(combined_orbs, ordering="chem").elems
S = integrals.compute_overlap_integrals(combined_orbs)

driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SU2, n_threads=4)
driver.initialize_system(n_sites=n_orbitals, n_elec=n_electrons, spin=0)
mpo = driver.get_qc_mpo(h1e=h1, g2e=G, ecore=nuc_repulsion, iprint=0)
ket = driver.get_random_mps(tag="KET", bond_dim=100, nroots=number_roots)
energies = driver.dmrg(mpo, ket, n_sweeps=10, bond_dims=[100], noises=[1e-5] * 4 + [0], thrds=[1e-10] * 8, iprint=1)

idx = driver.orbital_reordering(h1, G)
h1_new = h1[idx][:, idx]
g2_new = G[idx][:, idx][:, :, idx][:, :, :, idx]

driver.initialize_system(n_sites=n_orbitals, n_elec=n_electrons, spin=0)
mpo = driver.get_qc_mpo(h1e=h1_new, g2e=g2_new, ecore=nuc_repulsion, iprint=0)
ket = driver.get_random_mps(tag="KET", bond_dim=100, nroots=number_roots)
energies = driver.dmrg(mpo, ket, n_sweeps=10, bond_dims=[100], noises=[1e-5] * 4 + [0], thrds=[1e-10] * 8, iprint=1)
print("State-averaged MPS energies = [%s]" % " ".join("%20.15f" % x for x in energies))

kets = [driver.split_mps(ket, ir, tag="KET-%d" % ir) for ir in range(ket.nroots)]
sa_1pdm = np.mean([driver.get_1pdm(k) for k in kets], axis=0)
sa_2pdm = np.mean([driver.get_2pdm(k) for k in kets], axis=0).transpose(0, 3, 1, 2)
print(
    "Energy from SA-pdms = %20.15f"
    % (np.einsum("ij,ij->", sa_1pdm, T + V) + 0.5 * np.einsum("ijkl,ijkl->", sa_2pdm, G) + nuc_repulsion)
)

idx_back = np.zeros(len(idx), dtype=int)
for i in range(len(idx)):
    idx_back[idx[i]] = i

sa_1pdm = sa_1pdm[idx_back][:, idx_back]
sa_2pdm = sa_2pdm[idx_back][:, idx_back][:, :, idx_back][:, :, :, idx_back]
sa_2pdm_phys = sa_2pdm.swapaxes(1, 2)  # Physics Notation

with open("iteration_pno_dmrg_oo.dat", "a") as f:
    f.write(f"{-1} {0.00} " + " ".join(f"{x:.15f}" for x in energies) + "\n")


for iter in range(iterations):
    iter_start = time.perf_counter()

    # Orbital Refinement
    opti = fe.Optimization3D(world, Vnuc, nuc_repulsion)
    combined_orbs = opti.get_orbitals(orbitals=combined_orbs, rdm1=sa_1pdm, rdm2=sa_2pdm_phys, opt_thresh=0.001, occ_thresh=0.001)
    for i in range(n_gs_orbs):
        world.cube_plot(f"gs_orb{i}_iter{iter}", combined_orbs[i], molecule, zoom=4.0)  # Plot the refined orbitals
    
    for i in range(n_gs_orbs, n_orbitals):
        ex_index = i - n_gs_orbs
        world.cube_plot(f"ex_orb{ex_index}_iter{iter}", combined_orbs[i], molecule, zoom=4.0)

    # DMRG calculation with refined orbitals
    G = integrals.compute_two_body_integrals(combined_orbs, ordering="chem").elems  # g-tensor (electron-electron interaction)
    T = integrals.compute_kinetic_integrals(combined_orbs)  # Kinetic energy
    V = integrals.compute_potential_integrals(combined_orbs, Vnuc)  # Potential energy
    h1 = T + V
    S = integrals.compute_overlap_integrals(combined_orbs)  # Overlap

    driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SU2, n_threads=8)
    driver.initialize_system(n_sites=n_orbitals, n_elec=n_electrons, spin=0)
    mpo = driver.get_qc_mpo(h1e=h1, g2e=G, ecore=nuc_repulsion, iprint=0)
    ket = driver.get_random_mps(tag="KET", bond_dim=100, nroots=number_roots)
    energies = driver.dmrg(mpo, ket, n_sweeps=10, bond_dims=[100], noises=[1e-5] * 4 + [0], thrds=[1e-10] * 8, iprint=1)

    idx = driver.orbital_reordering(h1, G)
    h1_new = h1[idx][:, idx]
    g2_new = G[idx][:, idx][:, :, idx][:, :, :, idx]

    driver.initialize_system(n_sites=n_orbitals, n_elec=n_electrons, spin=0)
    mpo = driver.get_qc_mpo(h1e=h1_new, g2e=G, ecore=nuc_repulsion, iprint=0)
    ket = driver.get_random_mps(tag="KET", bond_dim=100, nroots=number_roots)
    energies = driver.dmrg(mpo, ket, n_sweeps=10, bond_dims=[100], noises=[1e-5] * 4 + [0], thrds=[1e-10] * 8, iprint=1)
    print("State-averaged MPS energies after refinement = [%s]" % " ".join("%20.15f" % x for x in energies))

    kets = [driver.split_mps(ket, ir, tag="KET-%d" % ir) for ir in range(ket.nroots)]
    sa_1pdm = np.mean([driver.get_1pdm(k) for k in kets], axis=0)  # Compute the state average 1-body rdm
    sa_2pdm = np.mean([driver.get_2pdm(k) for k in kets], axis=0).transpose(0, 3, 1, 2)  # Compute the state average 2-body rdm
    print(
        "Energy from SA-pdms = %20.15f"
        % (np.einsum("ij,ij->", sa_1pdm, T + V) + 0.5 * np.einsum("ijkl,ijkl->", sa_2pdm, G) + nuc_repulsion)
    )
    idx_back = np.zeros(len(idx), dtype=int)
    for i in range(len(idx)):
        idx_back[idx[i]] = i

    sa_1pdm = sa_1pdm[idx_back][:, idx_back]
    sa_2pdm = sa_2pdm[idx_back][:, idx_back][:, :, idx_back][:, :, :, idx_back]
    sa_2pdm_phys = sa_2pdm.swapaxes(1, 2)  # Change to physics Notation

    iter_end = time.perf_counter()
    iter_time = iter_end - iter_start

    with open("iteration_pno_dmrg_oo.dat", "a") as f:
        f.write(f"{iter} {iter_time:.2f} " + " ".join(f"{x:.15f}" for x in energies) + "\n")

    iteration_results.append({"iteration": iter, "iteration_time": iter_time, "energies": energies})

with open("results_pno_dmrg_oo.dat", "a") as f:
    f.write(" ".join(f"{x:.15f}" for x in energies) + "\n")


fe.cleanup(globals())