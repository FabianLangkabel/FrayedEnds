import madpy
import numpy

# define geometry (Angstrom)
geom = """
H 0.0 0.0 0.0
H 0.0 0.0 5.0
"""

# initialize madness
thresh=1.e-4
world = madpy.MadWorld3D(thresh=thresh)

# the many-body method to determine the RDMs
print("supported are: ", madpy.methods.SUPPORTED_RDM_METHODS)
print("available are: ", madpy.methods.AVAILABLE_RDM_METHODS)
method="fci" #"fci_dhf_slow" #dhf_slow allows complex integrals, but doesn't converge great, sometimes needs manual tweeking with initial guess

# initial orbitals
# can also just be a vector of MRA functions imported from somewhere else
orbitals = "pno" # "pno", "sto-3g"
if orbitals == "manual":
    world.change_nthreads(0)
    f1 = madpy.MRAFunctionFactory3D(world, pyfunc=lambda x,y,z:  numpy.exp(-0.5*(x**2+y**2+z**2)**0.5)).get_function(type="active")
    f2 = madpy.MRAFunctionFactory3D(world, pyfunc=lambda x,y,z: numpy.exp(-0.5*(x**2+y**2+(z-1.0)**2)**0.5)).get_function(type="active")
    orbitals = [f1,f2]
    world.change_nthreads(11)
    integrals = madpy.Integrals3D(world)
    orbitals = integrals.orthonormalize(orbitals=orbitals)

# run the tandem optimization
# either madness + tequila
# or madness + pyscf
# depends on the chosen method
energy, orbitals, rdm1, rdm2 = madpy.optimize_basis_3D(world=world, many_body_method=method, geometry=geom, econv=thresh, orbitals=orbitals)

print("final energy is ", energy)
print(f"should be: -1.0 up to {thresh}: ", numpy.isclose(energy,-1.0,atol=thresh))