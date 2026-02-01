import os
import numpy as np
import tequila as tq
import frayedends as fe
from ortho_test_utils import plot_orbitals_before_after, plot_energy_comparison, print_summary

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'gauß_potential', '1')
os.makedirs(OUTPUT_DIR, exist_ok=True)

n_electrons = 2
n_orbitals = 4
geometry = "H 0.0 0.0 0.0 \nH 0.0 0.0 1.0"
econv = 1e-8

def potential(x: float, y: float) -> float:
    a = -5.0
    b = -2.0
    c = -3.0
    r = np.array([x, y])
    return c * np.exp(-0.5 * np.linalg.norm(r) ** 2)
    # return a * np.exp(-0.5 * np.linalg.norm(r - np.array([1, 0])) ** 2) + b * np.exp(-0.5 * np.linalg.norm(r - np.array([0, 1])) ** 2) + c * np.exp(-0.5 * np.linalg.norm(r) ** 2)



def run_calculation(ortho_method="mixed", max_iterations=6):
    print(f"\n{'='*80}")
    print(f"Running with {ortho_method.upper()} orthonormalization")
    print(f"{'='*80}\n")

    world = fe.MadWorld2D()

    factory = fe.MRAFunctionFactory2D(world, potential)
    mra_pot = factory.get_function()

    eigen = fe.Eigensolver2D(world, mra_pot)
    orbitals = eigen.get_orbitals(0, n_orbitals, 0, n_states=12)

    world.plane_plot("potential_linear.dat", mra_pot, datapoints=501)

    energies = []
    current = 0.0

    u = None

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration} ---")

        integrals = fe.Integrals2D(world)
        S = integrals.compute_overlap_integrals(orbitals)
        G = integrals.compute_two_body_integrals(orbitals, ordering="phys")
        T = integrals.compute_kinetic_integrals(orbitals)
        V = integrals.compute_potential_integrals(orbitals, mra_pot)
        # VQE
        mol = tq.Molecule(
            geometry, units="bohr",
            one_body_integrals=T+V,
            two_body_integrals=G,
            n_electrons=n_electrons,
            nuclear_repulsion=0.0
        )
        print("1")
        H2=mol.make_hamiltonian()
        print("2")
        res = np.linalg.eigvalsh(H2.to_matrix())
        print("3")
        print(res[0])

        # result.energy should be similar to res[0]
        # or loop till difference to loop before is small enough
        while(True):
            print("loop")
            U = mol.make_ansatz(name="HCB-UpCCGD")
            print(mol.n_electrons)
            opt = tq.quantumchemistry.optimize_orbitals(
                molecule=mol, circuit=U, use_hcb=True, silent=True, initial_guess=u
            )
            H = opt.molecule.make_hardcore_boson_hamiltonian()
            E = tq.ExpectationValue(H=H, U=U)
            result = tq.minimize(E, silent=True)
            u = opt.mo_coeff
            print(f"{result.energy}")
            if res[0] - result.energy < 10e-4:
                break
            #

        rdm1, rdm2 = mol.compute_rdms(U, variables=result.variables, use_hcb=True)
        rdm1, rdm2 = fe.transform_rdms(u, rdm1, rdm2)

        print(f"Energy: {result.energy:+2.8f}")
        if iteration > 0:
            energies.append(result.energy)

        print("Orbital occupations:")
        for i in range(len(rdm1)):
            print(f"  Orbital {i}: {rdm1[i,i]:.6e}")

        orbitals_before = [orb for orb in orbitals]

        opti = fe.Optimization2D(world, mra_pot, nuc_repulsion=0.0)

        if ortho_method == "mixed":
            opti.enable_mixed_orthonormalization()
        orbitals = opti.get_orbitals(
            orbitals=orbitals, rdm1=rdm1, rdm2=rdm2,
            opt_thresh=0.001, occ_thresh=0.001
        )

        plot_orbitals_before_after(
            integrals.transform_to_natural_orbitals(orbitals_before, rdm1)[0],
            integrals.transform_to_natural_orbitals(orbitals, rdm1)[0],
            world,
            iteration,
            OUTPUT_DIR, method_name=ortho_method
        )

        if np.isclose(result.energy, current, atol=econv, rtol=0.0):
            print(f"\nConverged after {iteration+1} iterations!")
            break
        current = result.energy

    print(f"\n{'='*80}")
    print(f"Final energy ({ortho_method}): {energies[-1]:+2.8f}")
    print(f"{'='*80}\n")

    del orbitals
    del orbitals_before
    del opti
    del integrals
    del mra_pot
    del eigen
    del factory

    return energies, world


if __name__ == "__main__":
    print("\n" + "="*80)
    print("COMPARING ORTHONORMALIZATION METHODS")
    print("="*80)

    energies_mixed, world1 = run_calculation(ortho_method="mixed")
    del world1

    energies_symmetric, world2 = run_calculation(ortho_method="symmetric")
    del world2



    plot_energy_comparison(
        {'mixed': energies_mixed, 'symmetric': energies_symmetric},
        OUTPUT_DIR,
        title='Energy Convergence: Linear Potential with Degeneracies',
        subtitle='Comparison of Orthonormalization Methods'
    )

    print_summary(
        {'mixed': energies_mixed, 'symmetric': energies_symmetric},
        OUTPUT_DIR
    )
