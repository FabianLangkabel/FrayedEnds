import numpy as np
from pyscf import fci
import frayedends as fe

molecule_name = "h2"
n_electrons = 2
box_size = 50.0
wavelet_order = 7
madness_thresh = 0.0001
econv = 1.0e-6


molecule = fe.MolecularGeometry(units="angstrom")
molecule.add_atom(0.0, 0.0, -0.5, "H")
molecule.add_atom(0.0, 0.0, 0.5, "H")
geom = "H 0.0 0.0 -0.5\nH 0.0 0.0 0.5"

world = fe.MadWorld3D(L=box_size, k=wavelet_order, thresh=madness_thresh)
integrals = fe.Integrals3D(world)

madpno = fe.MadPNO(world, geom, n_orbitals=4)  # ground state + 2 excited states 

gs_orbs = madpno.get_orbitals()

for i in range(len(gs_orbs)):
    world.cube_plot(f"gs_orb{i}", gs_orbs[i], molecule, zoom=4.0)

cis_x_per_root = madpno.compute_cis(n_excitation=2)
cis_orbs = madpno.orthonormalize_cis(integrals_obj=integrals)

for i in range(len(cis_orbs)):
    world.cube_plot(f"cis_orb{i}", cis_orbs[i], molecule, zoom=4.0)

cispd_orbs = madpno.compute_cispd(n_orbitals=4)

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
H_core = T + V
G = integrals.compute_two_body_integrals(orbs, ordering="chem").elems
n_orbitals = len(orbs)

e_roots, fcivecs = fci.direct_spin0.kernel(H_core, G, n_orbitals, n_electrons, nroots=3)

e_ground_tot = e_roots[0] + nuc_repulsion
e_excited_tot = e_roots[1] + nuc_repulsion
e_excited_tot_2 = e_roots[2] + nuc_repulsion

print(f"Total Energy (Ground State):    {e_ground_tot:+2.8f}")
print(f"Total Energy (1st Excited):     {e_excited_tot:+2.8f}")
print(f"Total Energy (2nd Excited):     {e_excited_tot_2:+2.8f}")

fe.cleanup(globals())