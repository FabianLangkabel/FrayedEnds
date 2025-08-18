from time import time

import numpy as np
import tequila as tq
from pyscf import fci

import madpy as mad

true_start = time()
n_electrons = 2 # Number of electrons
n_orbitals = 2 # Number of orbitals (all active in this example)
econv = 1.e-8 # Energy convergence threshold

def potential(x: float, y: float, z: float) -> float: # Qdot potential
    a = -5.0
    r = np.array([x, y, z])
    return a * np.exp(-0.5 * np.linalg.norm(r) ** 2)


world = mad.MadWorld() # This is required for any MADNESS calculation as it initializes the required environment

factory = mad.MRAFunctionFactory(world, potential) # This transform a python function into a MRA function which can be read by MADNESS
mra_pot = factory.get_function() # Potential as MRA function

eigen = mad.Eigensolver(world, mra_pot) # This sets up the eigensolver, which provides initial guess orbitals
orbitals = eigen.get_orbitals(
    0, n_orbitals, 0, n_states=5
)  # The first three numbers are the numbers of frozen_core, active and frozen_virtual orbitals (in this case all orbitals are active)
   # The last number is the number of computed guess orbitals (in this case the ES will compute 5 orbitals and return the n_orbitals states with the lowest energy)

world.line_plot("potential.dat", mra_pot, axis="z", datapoints=2001)  # This plots the potential along the z-axis
for i in range(len(orbitals)):
    world.line_plot(f"es_orb{i}.dat", orbitals[i], axis="z", datapoints=2001) # Plots guess orbitals

current = 0.0
for iteration in range(6):
    integrals = mad.Integrals(world) # Setup for integrals
    G = integrals.compute_two_body_integrals(orbitals, ordering="chem") # g-tensor (electron-electron interaction)
    T = integrals.compute_kinetic_integrals(orbitals) # Kinetic energy
    V = integrals.compute_potential_integrals(orbitals, mra_pot) # Potential energy (h-tensor=T+V)
    
    #FCI calculation
    e, fcivec = fci.direct_spin0.kernel(T + V, G.elems, n_orbitals, n_electrons) # Computes the energy and the FCI vector
    rdm1, rdm2 = fci.direct_spin0.make_rdm12(fcivec, n_orbitals, n_electrons) # Computes the 1- and 2- body reduced density matrices
    rdm2 = np.swapaxes(rdm2, 1, 2)

    print("iteration {} FCI energy {:+2.8f}".format(iteration, e))

    # Orbital optimization
    opti = mad.Optimization(world, mra_pot, nuc_repulsion=0.0)
    orbitals = opti.get_orbitals(
        orbitals=orbitals, rdm1=rdm1, rdm2=rdm2, opt_thresh=0.001, occ_thresh=0.001
    ) # Optimizes the orbitals and returns the new ones

    for i in range(len(orbitals)):
        world.line_plot(f"orb{i}.dat", orbitals[i]) # Plots the optimized orbitals

    if np.isclose(e, current, atol=econv, rtol=0.0):
            break # The loop terminates as soon as the energy changes less than econv in one iteration step
    current = e


true_end = time()
print("Total time: ", true_end - true_start)

del factory
del integrals
del opti
del eigen
del world
