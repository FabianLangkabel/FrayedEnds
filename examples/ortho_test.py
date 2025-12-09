import numpy as np
import tequila as tq

import frayedends as fe


n_electrons = 2  # Number of electrons
n_orbitals = 6  # Number of orbitals (all active in this example)
geometry="H 0.0 0.0 0.0" #dummy geometry (not actually used for calculations but needed to initialize tq.Molecule())
econv=1e-8

def potential(x: float, y: float) -> float:  # The potential V(x, y), which binds the electrons
    a = -5.0
    r = np.array([x, y])
    return a * np.exp(-0.5 * np.linalg.norm(r) ** 2)


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

world.plane_plot(
    "potential.dat", mra_pot, datapoints=501
)  # This plots the potential
for i in range(len(orbitals)):
    world.plane_plot(
        f"es_orb{i}.dat", orbitals[i], datapoints=501
    )  # Plots guess orbitals

current = 0.0
#Start of the main algorithm
for iteration in range(6):
    # Describes the System physical interactions, encoded in tensors
    # Are not calculated to get the hamiltonian but to describe its system
    integrals = fe.Integrals2D(world)  # Setup for integrals
    G = integrals.compute_two_body_integrals(
        orbitals, ordering="phys"
    )  # g-tensor (electron-electron interaction, two body interaction)
    T = integrals.compute_kinetic_integrals(orbitals)  # Kinetic energy
    V = integrals.compute_potential_integrals(
        orbitals, mra_pot
    )  # Potential energy (h-tensor=T+V)
    # T+V one body interaction

    #VQE
    # Molecule gets properties of the system. two/one body interaction, number of electrons, geometry=Specifies position of electrons, units=distance unit for geometry, nuclear repulsion=Describes repulsion energy between electrons
    mol = tq.Molecule(geometry, units="bohr", one_body_integrals=T+V, two_body_integrals=G, n_electrons=n_electrons, nuclear_repulsion=0.0) #initialize a molecule object (used to construct the quantum circuit)
    # Creates a circuit with parameters which can be optimized to find better solutions
    # UpCCGSD = "Unitary Pair Coupled Cluster with Generalized Singles and Doubles."
    U = mol.make_ansatz(name="UpCCGSD") #circuit ansatz
    # Calculates the Hamiltonian with this equation: \hat{H} = \sum_{pq} h_{pq} a_p^\dagger a_q + \frac{1}{2} \sum_{pqrs} g_{pqrs} a_p^\dagger a_q^\dagger a_r a_s + E_{\text{nuc}}
    H = mol.make_hamiltonian()
    # Creates the Expectation value which is treated like the cost function for the optimization in VQE
    # E is the Energy of the system H in the circuit U
    E = tq.ExpectationValue(H=H, U=U)
    # Optimizes U to find the minimum of the cost function E
    result = tq.minimize(E, silent=True) #this optimizes the circuit to find the many body wavefunction
    # rdm = "reduced density matrix"
    # result.variables are minimized variables of the circuit U
    # rdm1: Describes the probability of finding an electron in the orbital
    # rdm1 = $\gamma_{pq} = \langle \Psi | a_p^\dagger a_q | \Psi \rangle$, for n_orbitals = 4, 4x4 matrix
    # rdm2: Correlated motion of pairs of electrons - Needed for calculating electron-electron interaction energy
    # rdm2 = $\Gamma_{pqrs} = \langle \Psi | a_p^\dagger a_q^\dagger a_s a_r | \Psi \rangle$, for n_orbitals = 4, 4x4x4x4 tensor
    rdm1, rdm2 = mol.compute_rdms(U, variables=result.variables) #compute the one body and two body reduced density matrices

    print("iteration {} energy {:+2.8f}".format(iteration, result.energy))

    for i in range(len(rdm1)):
        print(f"rdm1[{i},{i}]:",rdm1[i,i])
    
    # Orbital optimization
    opti = fe.Optimization2D(world, mra_pot, nuc_repulsion=0.0)  #initializes the orbital optimization
    orbitals = opti.get_orbitals(
        orbitals=orbitals, rdm1=rdm1, rdm2=rdm2, opt_thresh=0.001, occ_thresh=0.001
    )  # Optimizes the orbitals and returns the new ones

    for i in range(len(orbitals)):
        world.plane_plot(f"es_orb{i}.dat", orbitals[i], datapoints=501)  # Plots the optimized orbitals

    if np.isclose(result.energy, current, atol=econv, rtol=0.0):
        break  # The loop terminates as soon as the energy changes less than econv in one iteration step
    current = result.energy


fe.cleanup(globals()) #general cleanup function to ensure all objects are garbage collected in the correct order