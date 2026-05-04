import numpy as np
import tequila as tq

mol = tq.Molecule(geometry='H 0.0 0.0 -0.5\nH 0.0 0.0 0.5',basis_set='sto-3g')

H = mol.make_hamiltonian().to_matrix()
print(H.shape)
eival, eivect = np.linalg.eigh(mol.make_hamiltonian().to_matrix())
print(eival)