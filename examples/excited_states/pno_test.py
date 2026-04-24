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

madpno = fe.MadPNO(world, geom, n_orbitals=2, cispd=1)

gs_orbs = madpno.get_gs_orbs()
ex_orbs = madpno.get_ex_orbs()

for i in range(len(gs_orbs)):
    world.cube_plot(f"gs_orb{i}", gs_orbs[i], molecule)

for i in range(len(ex_orbs)):
    world.cube_plot(f"ex_orb{i}", ex_orbs[i], molecule)

combined_orbs = madpno.get_gs_orbs() + madpno.get_ex_orbs()
nuc_repulsion = madpno.get_nuclear_repulsion()
Vnuc = madpno.get_nuclear_potential()

integrals = fe.Integrals3D(world)

combined_orbs = integrals.orthonormalize(orbitals=combined_orbs)

T = integrals.compute_kinetic_integrals(combined_orbs)
V = integrals.compute_potential_integrals(combined_orbs, Vnuc)
H_core = T + V
G = integrals.compute_two_body_integrals(combined_orbs, ordering="chem").elems
n_orbitals = len(combined_orbs)

e_roots, fcivecs = fci.direct_spin0.kernel(H_core, G, n_orbitals, n_electrons, nroots=2)

e_ground_tot = e_roots[0] + nuc_repulsion
e_excited_tot = e_roots[1] + nuc_repulsion

print(f"Total Energy (Ground State):    {e_ground_tot:+2.8f}")
print(f"Total Energy (1st Excited):     {e_excited_tot:+2.8f}")
print(f"Excitation Energy (Vertical):   {(e_excited_tot - e_ground_tot)*27.2114:2.4f}")

fe.cleanup(globals())