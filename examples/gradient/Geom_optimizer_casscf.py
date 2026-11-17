from time import time

from pyscf import gto, mcscf, scf
from pyscf.geomopt import geometric_solver

"""
Reference calculation for Geom_optimizer.py using a Gaussian-basis CASSCF solver from pyscf.
Same molecule, starting geometry and active space (n_orbitals active orbitals, all electrons active) as in Geom_optimizer.py,
but with orbitals expanded in a (large) Gaussian basis set instead of MRA functions.
The analytic CASSCF nuclear gradients are passed to geomeTRIC through pyscf's geomeTRIC interface.
"""


def make_casscf(pyscf_mol, n_orbitals):
    mf = scf.RHF(pyscf_mol).run()
    return mcscf.CASSCF(mf, n_orbitals, pyscf_mol.nelectron)  # all electrons active, as in the FCI-based MRA calculation


geom_opt_start = time()
pyscf_mol = gto.M(atom="H 0.0 0.0 -2.5\nH 0.0 0.0 2.5", unit="bohr", basis="cc-pV5Z")  # initial guess molecule
print(pyscf_mol.atom_coords())

mc = make_casscf(pyscf_mol, n_orbitals=2)
new_mol = geometric_solver.optimize(mc)  # geometry optimization with geomeTRIC using analytic CASSCF gradients
geom_opt_end = time()

mc_final = make_casscf(new_mol, n_orbitals=2)
mc_final.kernel()
print("Final CASSCF energy:", mc_final.e_tot)
print("New geometry (bohr)")
print(new_mol.atom_coords())
print("new geometry a")
print(new_mol.tostring())
print("Geometry optimization time:", geom_opt_end - geom_opt_start)
