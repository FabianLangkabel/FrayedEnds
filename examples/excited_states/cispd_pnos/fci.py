import numpy as np
import tequila as tq
import frayedends as fe
from pyscf import fci
from math import pi
import time

n_electrons = 2
box_size = 50.0
wavelet_order = 7
madness_thresh = 1.0e-6
econv = 1.0e-6


distance = np.arange(1.5, 0.15, -0.05).tolist()

with open("fci_h2_ccpVQZ.dat", "w") as f:
    header = "distance dist_time_s fci_energy_0 fci_energy_1"
    f.write(header + "\n")

for d in distance:
    dist_start = time.perf_counter()
    reported_distance = d*2
    geom = (
            "H 0.0 0.0 " + (-d).__str__() + "\n"
            "H 0.0 0.0 " + d.__str__() + "\n"
    )

    mol = tq.Molecule(geometry=geom, basis_set='aug-cc-pvqz', units='a')
    c, h, g_chem = mol.get_integrals(ordering='chem')
    e_roots, fcivecs = fci.direct_spin0.kernel(h, g_chem.elems, mol.n_orbitals, mol.n_electrons, nroots=3)

    fci_energy_0 = e_roots[0] + c 
    fci_energy_1 = e_roots[1] + c 

    print("gs: ", e_roots[0] + c)
    print("ex: ", e_roots[1] + c)

    dist_end = time.perf_counter()
    dist_time = dist_end - dist_start
    print(f"Distance {reported_distance:.3f} took {dist_time:.2f} s")
    
    with open("fci_h2_ccpVQZ.dat", "a") as f:
            f.write(f"{reported_distance:.3f} {dist_time:.2f} {fci_energy_0: .15f} {fci_energy_1: .15f}" + "\n")
