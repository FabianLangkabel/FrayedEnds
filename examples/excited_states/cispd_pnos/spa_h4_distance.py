import numpy as np
import tequila as tq
import frayedends as fe
from pyscf import fci
from math import pi
import time

n_electrons = 4
box_size = 50.0
wavelet_order = 7
madness_thresh = 1.0e-6
econv = 1.0e-6


distance = np.round(np.arange(3.0, 0.55, -0.05, dtype=np.float64), 3).tolist()

with open("spa_h4.dat", "w") as f:
    header = "distance dist_time_s fci_energy_0 spa_energy_0 fci_energy_1 spa_energy_1"
    f.write(header + "\n")

for d in distance:
    dist_start = time.perf_counter()
    reported_distance = d
    geom = (
                "H 0.0 0.0 0.0\n"
                "H 0.0 0.0 " + (d).__str__() + "\n"
                "H 0.0 0.0 " + (2*d).__str__() + "\n"
                "H 0.0 0.0 " + (3*d).__str__() + "\n"
    )
    molecule = fe.MolecularGeometry(geometry=geom, units='angstrom')

    world = fe.MadWorld(ndims=3, L=box_size, k=wavelet_order, thresh=madness_thresh)
    integrals = fe.Integrals(world)

    pno_start = time.perf_counter()
    madpno = fe.MadPNO(world, geom, n_orbitals=4)
    pno_end = time.perf_counter()
    pno_time = pno_end - pno_start
    print("Generating PNOs took %.2f seconds" % pno_time)

    # ---------- original orbital set ------------
    gs_orbs_original = madpno.get_orbitals() # HF + MP2

    cis_start = time.perf_counter()
    cis_orbs_original = madpno.compute_cis(n_excitation=1) # CIS X Functions
    cis_end = time.perf_counter()
    cis_time = cis_end - cis_start
    print("Generating CIS took %.2f seconds" % cis_time)

    cispd_start = time.perf_counter()
    cispd_orbs_original = madpno.compute_cispd(n_orbitals=4) # CISPD PNO
    cispd_end = time.perf_counter()
    cispd_time = cispd_end - cispd_start
    print("Generating CISPD took %.2f seconds" % cispd_time)

    # ----------- symmetric orthonormalized orbital set -----------
    cis_orbs = integrals.project_out(gs_orbs_original, cis_orbs_original)
    cis_orbs = integrals.orthonormalize(cis_orbs)
    cispd_orbs = integrals.project_out(gs_orbs_original + cis_orbs, cispd_orbs_original) # Cispd only project out the ground state orbitals

    orbitals_sym = gs_orbs_original + cis_orbs_original + cispd_orbs_original
    orbitals_sym = integrals.orthonormalize(orbitals_sym)

    # ------------- create Hamiltonian from symmetric orbital set
    T = integrals.compute_kinetic_integrals(orbitals_sym)
    V = integrals.compute_potential_integrals(orbitals_sym, V=madpno.get_nuclear_potential())
    G = integrals.compute_two_body_integrals(orbitals_sym)
    c = madpno.get_nuclear_repulsion()
    h1 = T + V
    n_orbitals = len(orbitals_sym)

    
    mol = tq.Molecule(geometry=geom, one_body_integrals=T+V, two_body_integrals=G, nuclear_repulsion=c)
    H_gs = mol.make_hamiltonian()

    G_chem = integrals.compute_two_body_integrals(orbitals_sym, ordering="chem").elems
    e_roots, fcivecs = fci.direct_spin0.kernel(h1, G_chem, n_orbitals, n_electrons, nroots=3)
    fci_energy_0 = e_roots[0] + c
    fci_energy_1 = e_roots[1] + c
    print(f"c: {c}")

    spa_edges = madpno.get_spa_edges()
    print("SPA edges: ", spa_edges)

    print("\n=============== SPA Calculation GS ===============\n")
    U = mol.make_ansatz(name="spa", edges=spa_edges) # [(0, 1) (2, 3)]

    first_edge = spa_edges[0]
    second_edge = spa_edges[1]

    for i in range(len(first_edge)):
        print(f"first edge i: {first_edge[i]} ")

    for i in range(len(second_edge)):
        print(f"second edge i: {second_edge[i]} ")

    # U += mol.UR(first_edge[0], first_edge[1], (tq.Variable('a') + 0.5) * pi)
    U += mol.UR(first_edge[1], first_edge[2], (tq.Variable('b') + 0.5) * pi)
    U += mol.UR(first_edge[1], first_edge[3], (tq.Variable('c') + 0.5) * pi)
    U += mol.UR(first_edge[2], first_edge[3], (tq.Variable('d') + 0.5) * pi)
    # U += mol.UR(second_edge[0], second_edge[1], (tq.Variable('e') + 0.5) * pi)
    U += mol.UR(second_edge[1], second_edge[2], (tq.Variable('f') + 0.5) * pi)
    U += mol.UR(second_edge[1], second_edge[3], (tq.Variable('g') + 0.5) * pi)
    U += mol.UR(second_edge[2], second_edge[3], (tq.Variable('h') + 0.5) * pi)

    E = tq.ExpectationValue(U=U, H=H_gs)
    result = tq.minimize(E, silent=True)
    circuit_gs = tq.simulate(U, result.variables)

    print(f"FCI Ground state: {fci_energy_0}")
    print(f"SPA + UR GS energy: {result.energy}")
    print("SPA/FCI error: {:+2.5f}".format(result.energy-fci_energy_0))
    print(result.variables)
    print(f"Ground State Circuit: {circuit_gs}")

    gs_circuit = U.map_variables(result.variables)
    spa_energy_0 = result.energy


    # ----------- cholesky orthonormalized orbital set ------------------
    orbitals_ch = gs_orbs_original[:2] + cis_orbs_original + cispd_orbs_original + gs_orbs_original[2:] # f,0: HF, 1: CIS, 2,3: CISPD, 4,5: MP2 PNO
    orbitals_ch = integrals.orthonormalize(orbitals_ch, method="cholesky")
    
    # ---------- rotate the circuit into excited state orbitals basis (cholesky orthonormalized set) ----------
    S = integrals.compute_overlap_integrals(orbitals_sym, orbitals_ch)
    rotation = mol.get_givens_circuit(S)

    # ----------- SPA excited state with cholesky orthonormalized orbital set-----------
    print("\n=============== SPA Calculation ES ===============\n")
    circuit_list = [gs_circuit]
    constants = [5.0]

    ex_spa_edges = madpno.get_spa_edges(orbitals=orbitals_ch)
    print("SPA edges: ", ex_spa_edges)

    ex_first_edge = ex_spa_edges[0]
    ex_second_edge = ex_spa_edges[1]

    for i in range(len(ex_first_edge)):
        print(f"ex first edge i: {ex_first_edge[i]} ")

    for i in range(len(ex_second_edge)):
        print(f"ex second edge i: {ex_second_edge[i]} ")

    U_ex = mol.make_ansatz(name="spa", edges=ex_spa_edges)

    UR = mol.UR(ex_first_edge[0], ex_first_edge[1], (tq.Variable('s') + 0.5) * pi)
    UR += mol.UR(ex_first_edge[0], ex_first_edge[2], (tq.Variable('t') + 0.5) * pi)

    UR += mol.UR(ex_second_edge[0], ex_second_edge[1], (tq.Variable('w') + 0.5) * pi)
    UR += mol.UR(ex_second_edge[0], ex_second_edge[2], (tq.Variable('x') + 0.5) * pi)


    ti = fe.TequilaInterface(mol=mol)
    E = ti.expectation_value_orthogonality_constraint(
        H=H_gs, # use ground state Hamiltonian and rotate the circuit into the different basis
        U=U_ex + UR + rotation,
        circuit_list=circuit_list, 
        constant_list=constants
    )

    minimize_start = time.perf_counter()
    result = tq.minimize(E, silent=True)
    circuit_ex = tq.simulate(U_ex + UR + rotation, result.variables)
    minimize_end = time.perf_counter()
    # print(f"minimize & simulate time: {minimize_end - minimize_start}")

    print(f"FCI Singlet excited state energy: {fci_energy_1}")
    print(f"SPA Singlet excited state energy: {result.energy}")
    print("SPA/FCI error: {:+2.5f}".format(result.energy-fci_energy_1))
    print(result.variables)
    print(f"Excited State Circuit: {circuit_ex}")

    spa_energy_1 = result.energy

    dist_end = time.perf_counter()
    dist_time = dist_end - dist_start
    print(f"Distance {reported_distance:.3f} took {dist_time:.2f} s")
    
    with open("spa_h4.dat", "a") as f:
            f.write(f"{reported_distance:.3f} {dist_time:.2f} {fci_energy_0: .15f} {spa_energy_0: .15f} {fci_energy_1: .15f} {spa_energy_1: .15f}" + "\n")

    del integrals
    del madpno
    del ti
    del world
    del molecule
