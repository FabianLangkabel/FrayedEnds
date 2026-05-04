import frayedends as fe

import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

molecule_name = "h2"
n_electrons = 2
box_size = 50.0
wavelet_order = 7
madness_thresh = 0.0001
econv = 1.0e-6
number_roots = 1


molecule = fe.MolecularGeometry(units="angstrom")
molecule.add_atom(0.0, 0.0, -0.5, "H")
molecule.add_atom(0.0, 0.0, 0.5, "H")
geom = "H 0.0 0.0 -0.5\nH 0.0 0.0 0.5"

world = fe.MadWorld3D(L=box_size, k=wavelet_order, thresh=madness_thresh)

madpno = fe.MadPNO(world, geom, n_orbitals=4)

gs_orbs = madpno.get_gs_orbs()

for i in range(len(gs_orbs)):
    world.cube_plot(f"gs_orb{i}", gs_orbs[i], molecule, zoom=4.0)

nuc_repulsion = madpno.get_nuclear_repulsion()
Vnuc = madpno.get_nuclear_potential()

integrals = fe.Integrals3D(world)
combined_orbs = gs_orbs
combined_orbs = integrals.orthonormalize(orbitals=combined_orbs)

T = integrals.compute_kinetic_integrals(combined_orbs)
V = integrals.compute_potential_integrals(combined_orbs, Vnuc)
h1 = T + V
G = integrals.compute_two_body_integrals(combined_orbs, ordering="chem").elems
S = integrals.compute_overlap_integrals(combined_orbs)
n_orbitals = len(combined_orbs)

driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SU2, n_threads=4)
driver.initialize_system(n_sites=n_orbitals, n_elec=n_electrons, spin=0)
mpo = driver.get_qc_mpo(h1e=h1, g2e=G, ecore=nuc_repulsion, iprint=0)
ket = driver.get_random_mps(tag="GS", bond_dim=250, nroots=number_roots)
energy = driver.dmrg(mpo, ket, n_sweeps=500, bond_dims=[100], noises=[1e-5] * 4 + [0], thrds=[1e-10] * 8, iprint=1)

idx = driver.orbital_reordering(h1, G)
h1_new = h1[idx][:, idx]
g2_new = G[idx][:, idx][:, :, idx][:, :, :, idx]

driver.initialize_system(n_sites=n_orbitals, n_elec=n_electrons, spin=0)
mpo = driver.get_qc_mpo(h1e=h1_new, g2e=g2_new, ecore=nuc_repulsion, iprint=0)
ket = driver.get_random_mps(tag="GS", bond_dim=250, nroots=number_roots)
energy = driver.dmrg(mpo, ket, n_sweeps=500, bond_dims=[100], noises=[1e-5] * 4 + [0], thrds=[1e-10] * 8, iprint=1)
print(f"MPS energies = [{energy:20.15f}]")

sa_1pdm = driver.get_1pdm(ket)
sa_2pdm = driver.get_2pdm(ket)
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

fe.cleanup(globals())