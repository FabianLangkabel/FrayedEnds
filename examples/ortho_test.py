import os
import numpy as np
import tequila as tq
import frayedends as fe
from ortho_test_utils import plot_orbitals_2d, plot_energy_comparison, print_summary

# Output directory for this test
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'basic_test')
os.makedirs(OUTPUT_DIR, exist_ok=True)

n_electrons = 2  # Number of electrons
n_orbitals = 8  # Number of orbitals (all active in this example)
geometry="H 0.0 0.0 0.0" #dummy geometry (not actually used for calculations but needed to initialize tq.Molecule())
econv=1e-8

def potential(x: float, y: float) -> float:  # The potential V(x, y), which binds the electrons
    a = -5.0
    r = np.array([x, y])
    return a * np.exp(-0.5 * np.linalg.norm(r) ** 2)


def run_calculation(ortho_method="mixed", degeneracy_tol=1e-6, max_iterations=6):
    """Run calculation with specified orthonormalization method"""

    print(f"\n{'='*80}")
    print(f"Running with {ortho_method.upper()} orthonormalization")
    print(f"Basic 2D potential test")
    print(f"{'='*80}\n")

    world = fe.MadWorld2D()  # This is required for any MADNESS calculation as it initializes the required environment

    factory = fe.MRAFunctionFactory2D(
        world, potential
    )  # This transform a python function into a MRA function which can be read by MADNESS
    mra_pot = factory.get_function()  # Potential as MRA function

    eigen = fe.Eigensolver2D(
        world, mra_pot
    )  # This sets up the eigensolver, which provides initial guess orbitals
    orbitals = eigen.get_orbitals(
        0, n_orbitals, 0, n_states=10
    )  # The first three numbers are the numbers of frozen_core, active and frozen_virtual orbitals (in this case all orbitals are active)
    # The last number is the number of computed guess orbitals (in this case the ES will compute 10 orbitals and return the n_orbitals states with the lowest energy)

    # Plot initial potential (plane_plot only accepts filename, not path)
    world.plane_plot("potential.dat", mra_pot, datapoints=501)

    energies = []
    current = 0.0

    #Start of the main algorithm
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration} ---")

        # Describes the System physical interactions, encoded in tensors
        integrals = fe.Integrals2D(world)  # Setup for integrals
        G = integrals.compute_two_body_integrals(
            orbitals, ordering="phys"
        )  # g-tensor (electron-electron interaction, two body interaction)
        T = integrals.compute_kinetic_integrals(orbitals)  # Kinetic energy
        V = integrals.compute_potential_integrals(
            orbitals, mra_pot
        )  # Potential energy (h-tensor=T+V)

        #VQE
        mol = tq.Molecule(geometry, units="bohr", one_body_integrals=T+V, two_body_integrals=G, n_electrons=n_electrons, nuclear_repulsion=0.0)
        U = mol.make_ansatz(name="HCB-UpCCGD") #circuit ansatz
        u = None
        opt = tq.quantumchemistry.optimize_orbitals(molecule=mol, circuit=U, use_hcb=True, silent=True, initial_guess=u)

        H = opt.molecule.make_hardcore_boson_hamiltonian()
        E = tq.ExpectationValue(H=H, U=U)
        result = tq.minimize(E, silent=True) #this optimizes the circuit to find the many body wavefunction

        rdm1, rdm2 = mol.compute_rdms(U, variables=result.variables) #compute the one body and two body reduced density matrices
        u = opt.mo_coeff
        rdm1, rdm2 = fe.transform_rdms(u, rdm1, rdm2)

        print(f"Energy: {result.energy:+2.8f}")
        energies.append(result.energy)

        # Orbital optimization
        opti = fe.Optimization2D(world, mra_pot, nuc_repulsion=0.0)

        if ortho_method == "mixed":
            opti.enable_mixed_orthonormalization(degeneracy_tol=degeneracy_tol)

        orbitals = opti.get_orbitals(
            orbitals=orbitals, rdm1=rdm1, rdm2=rdm2, opt_thresh=0.001, occ_thresh=0.001
        )  # Optimizes the orbitals and returns the new ones

        plot_orbitals_2d(orbitals, world, iteration, OUTPUT_DIR, method_name=ortho_method)

        for i in range(len(orbitals)):
            world.plane_plot(f"es_orb{i}.dat", orbitals[i], datapoints=501)  # Plots the optimized orbitals

        if np.isclose(result.energy, current, atol=econv, rtol=0.0):
            print(f"\nConverged after {iteration+1} iterations!")
            break  # The loop terminates as soon as the energy changes less than econv in one iteration step
        current = result.energy

    print(f"\n{'='*80}")
    print(f"Final energy ({ortho_method}): {energies[-1]:+2.8f}")
    print(f"{'='*80}\n")

    # Cleanup
    del orbitals
    del opti
    del integrals
    del mra_pot
    del eigen
    del factory

    return energies, world


if __name__ == "__main__":
    # Run all orthonormalization methods
    print("\n" + "="*80)
    print("COMPARING ORTHONORMALIZATION METHODS")
    print("="*80)

    # Test 1: Mixed orthonormalization
    energies_mixed, world1 = run_calculation(ortho_method="mixed", degeneracy_tol=1e-6)
    del world1

    # Test 2: Symmetric orthonormalization
    energies_symmetric, world2 = run_calculation(ortho_method="symmetric")
    del world2

    # Plot energy convergence comparison using utility function
    plot_energy_comparison(
        {'mixed': energies_mixed, 'symmetric': energies_symmetric},
        OUTPUT_DIR,
        title='Energy Convergence: Basic 2D Potential Test',
        subtitle='Comparison of Orthonormalization Methods'
    )

    # Print summary using utility function
    print_summary(
        {'mixed': energies_mixed, 'symmetric': energies_symmetric},
        OUTPUT_DIR
    )

