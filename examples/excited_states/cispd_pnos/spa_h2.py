import numpy as np
import tequila as tq
import frayedends as fe
from math import pi
import time

n_electrons = 4
number_roots = 3
iterations = 6
box_size = 50.0
wavelet_order = 7
madness_thresh = 1.0e-6
econv = 1.0e-6

def expectation_value_orthogonality_constraint(H,U, circuit_list, constant_list):
    E = tq.ExpectationValue(H=H, U=U)
    if (len(circuit_list) != len(constant_list)):
        raise ValueError(f"Circuit_list and constant_list have different lengths. len(circuit_list): '{len(circuit_list)}', len(constant_list): '{len(constant_list)}'")
    list_length = len(circuit_list)
    for l in range(list_length):
        if (circuit_list[l].extract_variables() == None):
            raise ValueError(f"Circuit_list contains unparametrized elements")
    U_list = []
    for i in range(0, list_length):
        U_k = U + circuit_list[i].dagger()
        P_k = 1
        for k in U_k.qubits:
            P_k*= tq.paulis.Qp(k)
        E_k = tq.ExpectationValue(H=P_k, U=U_k)
        U_list.append(constant_list[i]*E_k)
    return E + sum(U_list)

distance = np.arange(1.5, 0.25, -0.05).tolist()

with open("spa_h2.dat", "w") as f:
    header = "distance dist_time_s fci_energy_0 spa_energy_0 fci_energy_1 spa_energy_1"
    f.write(header + "\n")

total_start = time.perf_counter()

for d in distance:
    dist_start = time.perf_counter()
    reported_distance = d
    geom = (
            "H 0.0 0.0 " + (-d).__str__() + "\n"
            "H 0.0 0.0 " + d.__str__() + "\n"
    )

    world = fe.MadWorld(ndims=3, L=box_size, k=wavelet_order, thresh=madness_thresh)
    integrals = fe.Integrals(world)

    pno_start = time.perf_counter()
    madpno = fe.MadPNO(world, geom, n_orbitals=2)
    pno_end = time.perf_counter()
    pno_time = pno_end - pno_start
    print("Generating PNOs took %.2f seconds" % pno_time)

    gs_orbs_original = madpno.get_orbitals() # HF + MP2

    cis_start = time.perf_counter()
    cis_orbs_original = madpno.compute_cis(n_excitation=1) # CIS X Functions
    cis_end = time.perf_counter()
    cis_time = cis_end - cis_start
    print("Generating CIS took %.2f seconds" % cis_time)

    cispd_start = time.perf_counter()
    cispd_orbs_original = madpno.compute_cispd(n_orbitals=2) # CISPD PNO
    cispd_end = time.perf_counter()
    cispd_time = cispd_end - cispd_start
    print("Generating CISPD took %.2f seconds" % cispd_time)

    cis_orbs = integrals.project_out(gs_orbs_original, cis_orbs_original)
    cis_orbs = integrals.orthonormalize(cis_orbs)
    cispd_orbs = integrals.project_out(gs_orbs_original + cis_orbs, cispd_orbs_original) # Cispd only project out the ground state orbitals

    orbitals_sym = gs_orbs_original + cis_orbs_original + cispd_orbs_original
    orbitals_sym = integrals.orthonormalize(orbitals_sym)

    T = integrals.compute_kinetic_integrals(orbitals_sym)
    V = integrals.compute_potential_integrals(orbitals_sym, V=madpno.get_nuclear_potential())
    G = integrals.compute_two_body_integrals(orbitals_sym)
    c = madpno.get_nuclear_repulsion()

    mol = tq.Molecule(geometry=geom, one_body_integrals=T+V, two_body_integrals=G, nuclear_repulsion=c)
    H_gs = mol.make_hamiltonian()

    print("--- FCI Ground State with H_gs----")
    energies, eivect = np.linalg.eigh(H_gs.to_matrix())
    for i in range(len(eivect)):
        w = tq.QubitWaveFunction.from_array(eivect[:,i])
        t = [i for i in w.items()]
        if [i for i in w.items()][0][0].binary.count('1') == mol.n_electrons:
            print(energies[i],'-->',w)

    print("\n=============== SPA Calculation GS ===============\n")
    U = mol.make_ansatz(name="spa", edges=[(0,1,2,3)])

    # U += mol.UR(0, 1, (tq.Variable('a') + 0.5) * pi)
    U += mol.UR(1, 2, (tq.Variable("b") + 0.5) * pi)
    U += mol.UR(2, 3, (tq.Variable("c") + 0.5) * pi)
    # U += mol.UR(0, 3, (tq.Variable("d") + 0.5) * pi)
    U += mol.UR(1, 3, (tq.Variable("e") + 0.5) * pi)
    # U += mol.UR(0, 2, (tq.Variable("f") + 0.5) * pi)

    E = tq.ExpectationValue(U=U, H=H_gs)
    result = tq.minimize(E, silent=True)
    circuit_gs = tq.simulate(U, result.variables)

    print(f"FCI Ground state: {energies[0]}")
    print(f"SPA + UR GS energy: {result.energy}")
    print("SPA/FCI error: {:+2.5f}".format(result.energy-energies[0]))
    print(result.variables)
    print(f"Ground State Circuit: {circuit_gs}")

    gs_circuit = U.map_variables(result.variables)

    fci_energy_0 = energies[0]
    spa_energy_0 = result.energy

    orbitals_ch = gs_orbs_original[:1] + cis_orbs_original + cispd_orbs_original + gs_orbs_original[1:] # 0: HF, 1: CIS, 2: CISPD, 3: MP2 PNO
    orbitals_ch = integrals.orthonormalize(orbitals_ch, method="cholesky")

    S = integrals.compute_overlap_integrals(orbitals_sym, orbitals_ch)
    rotation = mol.get_givens_circuit(S)

    print("\n=============== SPA Calculation ES ===============\n")
    circuit_list = [gs_circuit]
    constants = [5.0]

    U_ex = mol.make_ansatz(name="spa", edges=[(0,1,2,3)])

    UR = mol.UR(0, 1, (tq.Variable('u') + 0.5) * pi)
    # UR += mol.UR(1, 2, (tq.Variable('v') + 0.5) * pi)
    # UR += mol.UR(2, 3, (tq.Variable('w') + 0.5) * pi)
    # UR += mol.UR(0, 3, (tq.Variable('x') + 0.5) * pi)
    # UR += mol.UR(1, 3, (tq.Variable("y") + 0.5) * pi)
    UR += mol.UR(0, 2, (tq.Variable("z") + 0.5) * pi)

    E = expectation_value_orthogonality_constraint(
        H=H_gs, # use ground state Hamiltonian and rotate the circuit into the different basis
        U=U_ex + UR + rotation,
        circuit_list=circuit_list, 
        constant_list=constants
    )
    result = tq.minimize(E, silent=True)
    circuit_ex = tq.simulate(U_ex + UR + rotation, result.variables)

    print(f"FCI Singlet excited state energy: {energies[9]}")
    print(f"SPA Singlet excited state energy: {result.energy}")
    print("SPA/FCI error: {:+2.5f}".format(result.energy-energies[9]))
    print(result.variables)
    print(f"Excited State Circuit: {circuit_ex}")

    fci_energy_1 = energies[9] 
    spa_energy_1 = result.energy

    dist_end = time.perf_counter()
    dist_time = dist_end - dist_start
    print(f"Distance {reported_distance:.3f} took {dist_time:.2f} s")
    
    with open("spa_h2.dat", "a") as f:
            f.write(f"{reported_distance:.3f} {dist_time:.2f} {fci_energy_0: .15f} {spa_energy_0: .15f} {fci_energy_1: .15f} {spa_energy_1: .15f}" + "\n")

    del integrals
    del madpno
    del world