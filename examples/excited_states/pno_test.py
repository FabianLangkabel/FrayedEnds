import time

import numpy as np
from pyscf import fci

import frayedends as fe

# Parameter Configuration
molecule_name = "h2"
n_electrons = 2  # Number of electrons
iterations = 1  # Number of iterations
box_size = 50.0  # Size of the simulation box
wavelet_order = 7  # Order of wavelet basis functions
madness_thresh = 0.0001  # Threshold for numerical precision of function representation
econv = 1.0e-6  # Energy convergence threshold
basisset = "6-31g"  # Initial basis set for calculation


# Defines a linear H4 molecule geometry with 1.0 Angstrom spacing between adjacent atoms
geom = "H 0.0 0.0 -0.5\nH 0.0 0.0 0.5\n"

# Setting up the numerical environment for the MRA calculations
world = fe.MadWorld3D(L=box_size, k=wavelet_order, thresh=madness_thresh)

madpno = fe.MadPNO(world, geom, n_orbitals=4, cispd=1)
orbs = madpno.get_orbitals()
print("got orbs")

'''
nuc_repulsion = madpno.get_nuclear_repulsion()  # Compute nuclear repulsion energy
Vnuc = madpno.get_nuclear_potential()  # Compute nuclear potential

integrals = fe.Integrals3D(world)
orbs = integrals.orthonormalize(orbitals=orbs)  # Orthonormalize orbitals

n_orbitals = len(orbs)

current = 0.0
for iteration in range(iterations):

    # Calculate initial integrals
    integrals = fe.Integrals3D(world)
    G = integrals.compute_two_body_integrals(orbs, ordering="chem").elems
    T = integrals.compute_kinetic_integrals(orbs)
    V = integrals.compute_potential_integrals(orbs, Vnuc)

    # Full Configuration Interaction (FCI) calculation
    e, fcivec = fci.direct_spin0.kernel(T + V, G, n_orbitals, n_electrons)  # Computes the energy and the FCI vector
    # Calculate reduced density matrices needed for orbital refinement
    rdm1, rdm2 = fci.direct_spin0.make_rdm12(
        fcivec, n_orbitals, n_electrons
    )  # Computes the 1- and 2- body reduced density matrices
    rdm2 = np.swapaxes(rdm2, 1, 2)  # Change to physics notation

    e_tot = e + nuc_repulsion  # Computes total energy

    print("iteration {} FCI electronic energy {:+2.8f}, total energy {:+2.8f}".format(iteration, e, e_tot))

    # Orbital Refinement
    opti = fe.Optimization3D(world, Vnuc, nuc_repulsion)
    orbs = opti.get_orbitals(
        orbitals=orbs, rdm1=rdm1, rdm2=rdm2, opt_thresh=0.001, occ_thresh=0.001
    )  # Refines the orbitals and returns the new ones

    if np.isclose(e_tot, current, atol=econv, rtol=0.0):
        break  # Loop terminates as soon as the energy changes less than econv in one iteration step
    current = e_tot

'''

fe.cleanup(globals())
