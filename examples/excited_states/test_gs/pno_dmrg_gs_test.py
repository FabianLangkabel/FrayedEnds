import frayedends as fe

import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

molecule_name = "h2"
n_electrons = 2
box_size = 50.0
wavelet_order = 7
madness_thresh = 1.0e-4
econv = 1.0e-6
number_roots = 1


molecule = fe.MolecularGeometry(units="angstrom")
molecule.add_atom(0.0, 0.0, -0.5, "H")
molecule.add_atom(0.0, 0.0, 0.5, "H")
geom = "H 0.0 0.0 -0.5\nH 0.0 0.0 0.5"

world = fe.MadWorld3D(L=box_size, k=wavelet_order, thresh=madness_thresh)

madpno = fe.MadPNO(world, geom, units="angstrom", n_orbitals=4)

gs_orbs = madpno.get_orbitals()

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

n_orbitals = len(combined_orbs)

driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SU2, n_threads=4)
driver.initialize_system(n_sites=n_orbitals, n_elec=n_electrons, spin=0)

idx = driver.orbital_reordering(h1, G)
h1_new = h1[idx][:, idx]
g2_new = G[idx][:, idx][:, :, idx][:, :, :, idx]

mpo = driver.get_qc_mpo(h1e=h1_new, g2e=g2_new, ecore=nuc_repulsion, iprint=0)
ket = driver.get_random_mps(tag="GS", bond_dim=250, nroots=number_roots)
energies = driver.dmrg(mpo, ket, n_sweeps=10, bond_dims=[100], noises=[1e-5] * 4 + [0], thrds=[1e-10] * 8, iprint=0)

# Extract rdms
kets = [driver.split_mps(ket, ir, tag="KET-%d" % ir) for ir in range(ket.nroots)]
sa_1pdm = np.mean([driver.get_1pdm(k) for k in kets], axis=0)  # Compute the state average 1-body rdm
sa_2pdm = np.mean([driver.get_2pdm(k) for k in kets], axis=0).transpose(0, 3, 1, 2)  # Compute the state average 2-body rdm

# State-averaged energy from SA-RDMs
e_avg = (np.einsum("ij,ij->", sa_1pdm, h1_new)
         + 0.5 * np.einsum("ijkl,ijkl->", sa_2pdm, g2_new)
         + nuc_repulsion)

print(f"{'Energies':<{14 * number_roots}}{'E_SA':<16}")
print(f"{''.join(f'{e:<14.8f}' for e in energies)}{e_avg:<16.8f}")

fe.cleanup(globals())