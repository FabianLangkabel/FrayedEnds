import numpy

import frayedends

# define geometry (Angstrom)
geom = """
Li 0.0 0.0 0.0
H 0.0 0.0 5.0
"""

# initialize madness
thresh = 1.0e-4
world = frayedends.MadWorld3D(thresh=thresh)

# the many-body method to determine the RDMs
print("supported are: ", frayedends.methods.SUPPORTED_RDM_METHODS)
print("available are: ", frayedends.methods.AVAILABLE_RDM_METHODS)
method = "fci_dhf_slow"  # dhf_slow allows complex integrals, but doesn't converge great, sometimes needs manual tweeking with initial guess

# initial orbitals
# can also just be a vector of MRA functions imported from somewhere else
orbitals = "sto-3g"  # "pno", "sto-3g"

# run the tandem optimization
# either madness + tequila
# or madness + pyscf
# depends on the chosen method
energy, orbitals, rdm1, rdm2 = frayedends.optimize_basis_3D(
    world=world, many_body_method=method, geometry=geom, econv=thresh, orbitals=orbitals
)

print("final energy is ", energy)
print(f"should be: -1.0 up to {thresh}: ", numpy.isclose(energy, -1.0, atol=thresh))
