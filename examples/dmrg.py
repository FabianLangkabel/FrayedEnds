import madpy
import numpy

# define geometry (Angstrom)
# block2 has some issues with h2 (probably too small?)
geom = """
H 0.0 0.0 0.0
H 0.0 0.0 1.0 
H 0.0 0.0 2.0 
H 0.0 0.0 3.0
"""

# initialize madness
thresh=1.e-4
world = madpy.MadWorld(thresh=thresh)

# the many-body method to determine the RDMs
print("supported are: ", madpy.methods.SUPPORTED_RDM_METHODS)
print("available are: ", madpy.methods.AVAILABLE_RDM_METHODS)
method="fci"

# initial orbitals
# can also just be a vector of MRA functions imported from somewhere else
orbitals = "pno" # "pno", "sto-3g"

# optimize the basis
energy, orbitals, rdm1, rdm2 = madpy.optimize_basis(world=world, many_body_method=method, geometry=geom, econv=thresh, orbitals=orbitals)

print("final energy is ", energy)
print(f"should be: -1.0 up to {thresh}: ", numpy.isclose(energy,-1.0,atol=thresh))