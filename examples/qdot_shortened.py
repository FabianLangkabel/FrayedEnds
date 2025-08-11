from time import time

import numpy as np
import tequila as tq
from pyscf import fci

import madpy as mad

true_start = time()
n_electrons = 2 # Number of electrons
n_orbitals = 2 # Number of orbitals (all active in this example)


def potential(x: float, y: float, z: float) -> float: # Qdot potential
    a = -5.0
    r = np.array([x, y, z])
    return a * np.exp(-0.5 * np.linalg.norm(r) ** 2)


world = mad.MadWorld() # This is required for any MADNESS calculation as it initializes the required environment

factory = mad.MRAFunctionFactory(world, potential) # This transform a python function into a MRA function which can be read by MADNESS
mra_pot = factory.get_function() # Potential as MRA function

# This function takes care of the algorithm, with orbitals= you can set the method to deternmine the initial guess orbitals and many_body_method= specifies the method to determine the rdms
energy, orbitals, rdm1, rdm2= mad.optimize_basis(world, Vnuc=mra_pot, n_electrons=n_electrons, n_orbitals=n_orbitals, 
                                                 orbitals="eigen", many_body_method="fci", maxiter=6, econv=1.e-8)

true_end = time()
print("Total time: ", true_end - true_start)

del factory
del world
