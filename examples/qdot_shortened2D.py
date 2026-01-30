from time import time

import numpy as np
import tequila as tq
from pyscf import fci

import frayedends as fe

true_start = time()
n_electrons = 2  # Number of electrons
n_orbitals = 6  # Number of orbitals (all active in this example)


def potential(x: float, y: float) -> float:  # Qdot potential
    r = np.array([x, y, 1e-10])
    return - 2 / np.linalg.norm(r)


world = fe.MadWorld2D(
    L=100, thresh=1e-4
)  # This is required for any MADNESS calculation as it initializes the required environment

factory = fe.MRAFunctionFactory2D(
    world, potential
)  # This transform a python function into a MRA function which can be read by MADNESS
mra_pot = factory.get_function()  # Potential as MRA function

# This function takes care of the algorithm, with orbitals= you can set the method to deternmine the initial guess orbitals and many_body_method= specifies the method to determine the rdms
energy, orbitals, rdm1, rdm2 = fe.optimize_basis_2D(
    world,
    Vnuc=mra_pot,
    n_electrons=n_electrons,
    n_orbitals=n_orbitals,
    orbitals="eigen",
    many_body_method="fci",
    maxiter=10,
    econv=1.0e-8,
)

print(np.linalg.eig(rdm1))
true_end = time()
print("Total time: ", true_end - true_start)

fe.cleanup(globals())
