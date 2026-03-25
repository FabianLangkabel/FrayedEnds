import frayedends as fe
from frayedends.atomicbasisprojector import AtomicBasisProjector
import tequila as tq
import numpy as np

geom = "Li 0.0 0.0 -5\nH 0.0 0.0 5"

world = fe.MadWorld3D()

#this returns the raw basis functions of the according quantum chemistry basis set in MRA form
bp = AtomicBasisProjector(world, geom, units="bohr", aobasis="sto-3g")
basis = bp.get_orbitals()

mol=tq.Molecule(geom, units="bohr", basis_set="sto-3g")

for i in range(len(basis)):
    world.cube_plot(f"atomic{i}", basis[i], fe.MolecularGeometry.from_tq_mol(mol), zoom=5)

#to construct an orbital basis according to a matrix C of basis coefficients:
C=np.eye(len(basis)) #replace with your coefficient matrix
intg=fe.Integrals3D(world)
orbitals = intg.transform(basis, C) #transforms orbitals according to: orbtials[i] = sum[j] basis[j]*C[j,i]

fe.cleanup(globals())