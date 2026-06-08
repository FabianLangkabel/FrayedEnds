from pyscf import fci
import frayedends as fe

import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

molecule_name = "h2"
n_electrons = 2
box_size = 50.0
wavelet_order = 7
madness_thresh = 0.0001
econv = 1.0e-6
number_roots = 3


molecule = fe.MolecularGeometry(units="angstrom")
molecule.add_atom(0.0, 0.0, -0.5, "H")
molecule.add_atom(0.0, 0.0, 0.5, "H")
geom = "H 0.0 0.0 -0.5\nH 0.0 0.0 0.5"

world = fe.MadWorld3D(L=box_size, k=wavelet_order, thresh=madness_thresh)
integrals = fe.Integrals3D(world)

madpno = fe.MadPNO(world, geom, n_orbitals=4)  # ground state + 2 excited states 

gs_orbs = madpno.get_orbitals()
hf_orbs = madpno.get_hf_orbitals()

for i in range(len(gs_orbs)):
    world.cube_plot(f"gs_orb{i}", gs_orbs[i], molecule, zoom=4.0)

for i in range(len(hf_orbs)):
    world.cube_plot(f"hf_orb{i}", hf_orbs[i], molecule, zoom=4.0)
 
cis_orbs = madpno.compute_cis(n_excitation=2)
cis_orbs = integrals.project_out(gs_orbs, cis_orbs)
cis_orbs = integrals.orthonormalize(cis_orbs)

for i in range(len(cis_orbs)):
    world.cube_plot(f"cis_orb{i}", cis_orbs[i], molecule, zoom=4.0)

cispd_orbs = madpno.compute_cispd(n_orbitals=4)
cispd_orbs = integrals.project_out(gs_orbs + cis_orbs, cispd_orbs)

for i in range(len(cispd_orbs)):
    world.cube_plot(f"cispd_orb{i}", cispd_orbs[i], molecule, zoom=4.0)

nuc_repulsion = madpno.get_nuclear_repulsion()
Vnuc = madpno.get_nuclear_potential()

orbs = gs_orbs + cis_orbs + cispd_orbs
orbs = integrals.orthonormalize(orbitals=orbs)

for i in range(len(orbs)):
    world.cube_plot(f"orb{i}", orbs[i], molecule, zoom=4.0)

T = integrals.compute_kinetic_integrals(orbs)
V = integrals.compute_potential_integrals(orbs, Vnuc)
h1 = T + V
G = integrals.compute_two_body_integrals(orbs, ordering="chem").elems
S = integrals.compute_overlap_integrals(orbs)
n_orbitals = len(orbs)

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

natural_orbs, occ_n, vec = integrals.transform_to_natural_orbitals(orbs, sa_1pdm)
print("Natural orbital occupation: ", occ_n)

for i in range(len(natural_orbs)):
    world.cube_plot(f"nat_orb{i}", natural_orbs[i], molecule, zoom=4.0)

fe.cleanup(globals())