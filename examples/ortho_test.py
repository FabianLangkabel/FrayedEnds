<<<<<<< ortho-test
import os
from typing import Any

import numpy as np
import tequila as tq
from numpy import floating
from pyscf import fci

import frayedends
import frayedends as fe

n_electrons = 2
n_orbitals = 6
max_iterations = 10
econv = 1e-8

def potential_three_peaks(x: float, y: float) -> float:
    """Three Gaussian peaks potential"""
    a = -(3.0 + 1.0 * n_electrons)
    b = -(2.0 + 0.8 * n_electrons)
    c = -(1.0 + 0.5 * n_electrons)

    width_scale = 1.0 / np.sqrt(max(1, n_electrons * 0.5))
    alpha = 0.5 * width_scale

    r1 = np.linalg.norm(np.array([x, y]) - np.array([4.0, 0.0]))
    r2 = np.linalg.norm(np.array([x, y]) - np.array([0.0, 4.0]))
    r3 = np.linalg.norm(np.array([x, y]))

    return (a * np.exp(-alpha * r1 ** 2) +
            b * np.exp(-alpha * r2 ** 2) +
            c * np.exp(-alpha * r3 ** 2))

def potential_single_peak(x: float, y: float) -> float:
    """Single Gaussian peak potential"""
    c = -1.5 * n_electrons
    width = 0.5

    if n_electrons > 2:
        width_scale = 0.15
    else:
        width_scale = 0.5

    r = np.array([x, y])

    return c * np.exp(-(width * width_scale) * np.linalg.norm(r) ** 2)

def potential_coulomb(x: float, y: float) -> floating[Any]:
    """Helium-like potential: -1/(r^2 + epsilon)"""
    r_vec = np.array([x, y])
    r_norm = np.linalg.norm(r_vec)
    epsilon = 0.0000001
    return -n_electrons / np.sqrt(r_norm**2 + epsilon)

def potential_three_peaks_4e(x: float, y: float, *args) -> float:
    a = -7.0
    b = -5.2
    c = -3.0

    alpha = 0.35355

    r1 = np.linalg.norm(np.array([x, y]) - np.array([4.0, 0.0]))
    r2 = np.linalg.norm(np.array([x, y]) - np.array([0.0, 4.0]))
    r3 = np.linalg.norm(np.array([x, y]))

    return (a * np.exp(-alpha * r1 ** 2) +
            b * np.exp(-alpha * r2 ** 2) +
            c * np.exp(-alpha * r3 ** 2))


def potential_single_peak_4e(x: float, y: float, *args) -> float:
    c = -6.0

    alpha = 0.075
    r_norm = np.linalg.norm(np.array([x, y]))

    return c * np.exp(-alpha * r_norm ** 2)


def potential_coulomb_4e(x: float, y: float, *args) -> float:
    r_norm = np.linalg.norm(np.array([x, y]))
    epsilon = 1e-7

    return -4.0 / np.sqrt(r_norm ** 2 + epsilon)


def run_calculation(potential_func, geometry, output_dir, ortho_method="mixed", max_iterations=10, early_stop=False):
    print(f"\n{'='*80}")
    print(f"Running with {ortho_method.upper()} orthonormalization")
    print(f"{'='*80}\n")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    world = fe.MadWorld2D()

    factory = fe.MRAFunctionFactory2D(world, potential_func)
    mra_pot = factory.get_function()

    eigen = fe.Eigensolver2D(world, mra_pot)
    all_orbitals = eigen.get_orbitals(0, n_orbitals, 0, n_states=12)

    world.plane_plot( "potential.dat", mra_pot, datapoints=501, zoom=10)
    integrals = fe.Integrals2D(world)

    energies = []
    current = 0.0
    current_orbitals = []

    for o in range(n_orbitals):
        current_orbitals.append(all_orbitals[o])
        print(f"Orbital {o} added!")
        if ortho_method == "symmetric":
            current_orbitals = integrals.orthonormalize(orbitals=current_orbitals)
        else:
            current_orbitals = integrals.orthonormalize(orbitals=current_orbitals)

        current_orbitals = integrals.orthonormalize(orbitals=current_orbitals)
        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration} ---")

            integrals = fe.Integrals2D(world)
            S = integrals.compute_overlap_integrals(current_orbitals)
            G = integrals.compute_two_body_integrals(current_orbitals, ordering="chem")
            T = integrals.compute_kinetic_integrals(current_orbitals)
            V = integrals.compute_potential_integrals(current_orbitals, mra_pot)
            # FCI
            e, fcivec = fci.direct_spin1.kernel(T+V, G.elems, o+1, n_electrons)  # Computes the energy and the FCI vector
            rdm1, rdm2 = fci.direct_spin1.make_rdm12(
                fcivec, o+1, n_electrons
            )  # Computes the 1- and 2- body reduced density matrices
            rdm2 = np.swapaxes(rdm2, 1, 2)  # swapping axes to match convention used in orbital refinement code

            print(f"Energy: {e:+2.8f}")
            energies.append(e)

            print("Orbital occupations:")
            for i in range(len(rdm1)):
                print(f"  Orbital {i}: {rdm1[i,i]:.6e}")

            orbitals_before = [orb for orb in current_orbitals]

            opti = fe.Optimization2D(world, mra_pot, nuc_repulsion=0.0)

            opti.set_orthonormalization_method(ortho_method)
            current_orbitals = opti.get_orbitals(
                orbitals=current_orbitals, rdm1=rdm1, rdm2=rdm2,
                opt_thresh=0.001, occ_thresh=0.001
            )

            if early_stop and np.isclose(e, current, atol=econv, rtol=0.0):
                print(f"\nConverged after {iteration+1} iterations!")
                break
            current = e

    del all_orbitals
    del orbitals_before
    del opti
    del integrals
    del mra_pot
    del eigen
    del factory

    return energies, world
=======
import numpy as np
import tequila as tq

import frayedends as fe

n_electrons = 2  # Number of electrons
n_orbitals = 6  # Number of orbitals (all active in this example)
geometry = "H 0.0 0.0 0.0"  # dummy geometry (not actually used for calculations but needed to initialize tq.Molecule())
econv = 1e-8


def potential(x: float, y: float) -> float:  # The potential V(x, y), which binds the electrons
    r = np.array([x, y, 1e-6])
    return -20.0 / np.linalg.norm(r)


world = fe.MadWorld2D()  # This is required for any MADNESS calculation as it initializes the required environment

factory = fe.MRAFunctionFactory2D(
    world, potential
)  # This transform a python function into a MRA function which can be read by MADNESS
mra_pot = factory.get_function()  # Potential as MRA function

eigen = fe.Eigensolver2D(world, mra_pot)  # This sets up the eigensolver, which provides initial guess orbitals
orbitals = eigen.get_orbitals(
    0, n_orbitals, 0, n_states=10
)  # The first three numbers are the numbers of frozen_core, active and frozen_virtual orbitals (in this case all orbitals are active)
# The last number is the number of computed guess orbitals (in this case the ES will compute 10 orbitals and return the n_orbitals states with the lowest energy)

world.plane_plot("potential.dat", mra_pot, datapoints=501)  # This plots the potential
for i in range(len(orbitals)):
    world.plane_plot(f"es_orb{i}.dat", orbitals[i], datapoints=501)  # Plots guess orbitals

current = 0.0
# Start of the main algorithm
for iteration in range(10):
    integrals = fe.Integrals2D(world)  # Setup for integrals
    G = integrals.compute_two_body_integrals(orbitals, ordering="phys")  # g-tensor (electron-electron interaction)
    T = integrals.compute_kinetic_integrals(orbitals)  # Kinetic energy
    V = integrals.compute_potential_integrals(orbitals, mra_pot)  # Potential energy (h-tensor=T+V)

    # VQE
    mol = tq.Molecule(
        geometry,
        units="bohr",
        one_body_integrals=T + V,
        two_body_integrals=G,
        n_electrons=n_electrons,
        nuclear_repulsion=0.0,
    )  # initialize a molecule object (used to construct the quantum circuit)
    U = mol.make_ansatz(name="UpCCGSD")  # circuit ansatz
    H = mol.make_hamiltonian()
    E = tq.ExpectationValue(H=H, U=U)
    result = tq.minimize(E, silent=True)  # this optimizes the circuit to find the many body wavefunction
    rdm1, rdm2 = mol.compute_rdms(
        U, variables=result.variables
    )  # compute the one body annd two body reduced density matrices

    print("iteration {} energy {:+2.8f}".format(iteration, result.energy))

    for i in range(len(rdm1)):
        print(f"rdm1[{i},{i}]:", rdm1[i, i])

    # Orbital optimization
    opti = fe.Optimization2D(world, mra_pot, nuc_repulsion=0.0)  # initializes the orbital optimization
    orbitals = opti.get_orbitals(
        orbitals=orbitals, rdm1=rdm1, rdm2=rdm2, opt_thresh=0.001, occ_thresh=0.001
    )  # Optimizes the orbitals and returns the new ones

    for i in range(len(orbitals)):
        world.plane_plot(f"es_orb{i}.dat", orbitals[i], datapoints=501)  # Plots the optimized orbitals

    if np.isclose(result.energy, current, atol=econv, rtol=0.0):
        break  # The loop terminates as soon as the energy changes less than econv in one iteration step
    current = result.energy


fe.cleanup(globals())  # general cleanup function to ensure all objects are garbage collected in the correct order
>>>>>>> devel
