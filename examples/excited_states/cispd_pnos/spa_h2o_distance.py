import numpy as np
import tequila as tq
import frayedends as fe
from pyscf import fci
from math import pi
import time
import sunrise as sun
import sys

sys.setrecursionlimit(1000000)

n_electrons = 10
box_size = 50.0
wavelet_order = 7
madness_thresh = 1.0e-6
econv = 1.0e-6

distance = np.round(np.arange(2.55, 0.54, -0.10, dtype=np.float64), 3).tolist() 

distance = distance[0:6]

with open("spa_h2o.dat", "w") as f:
    header = "distance dist_time_s fci_energy_0 spa_energy_0 fci_energy_1 spa_energy_1"
    f.write(header + "\n")

for d in distance:
    dist_start = time.perf_counter()
    reported_distance = d
    # H(d) = O + (H-O)*d = \vec{O} + d \vec(OH)
    geom = (    "O 0.0 0.0 0.1173\n"
                + "H 0.0 " + (0.7572*d).__str__() + " " + (0.1173+(-0.4692-0.1173)*d).__str__() + "\n"
                + "H 0.0 " + (-0.7572*d).__str__() + " " + (0.1173+(-0.4692-0.1173)*d).__str__() + "\n"
    )
    molecule = fe.MolecularGeometry(geometry=geom, units='angstrom')
    n_electrons_active = molecule.n_electrons - molecule.n_core_electrons
    print("Active Electron: ", n_electrons_active)

    world = fe.MadWorld(ndims=3, L=box_size, k=wavelet_order, thresh=madness_thresh)
    integrals = fe.Integrals(world)

    pno_start = time.perf_counter()
    madpno = fe.MadPNO(world, geom, n_orbitals=9, dft={"econv": 1.0e-5, "dconv": 1.0e-5})
    pno_end = time.perf_counter()
    pno_time = pno_end - pno_start
    print("Generating PNOs took %.2f seconds" % pno_time)

    # ---------- original orbital set ------------
    gs_orbs_original = madpno.get_orbitals() # HF + MP2

    cis_start = time.perf_counter()
    cis_orbs_original = madpno.compute_cis(n_excitation=1, dominant_contribution=True) # CIS X Functions
    cis_end = time.perf_counter()
    cis_time = cis_end - cis_start
    print("Generating CIS took %.2f seconds" % cis_time)

    cispd_start = time.perf_counter()
    cispd_orbs_original = madpno.compute_cispd(n_orbitals=8, dominant_contribution=True) # CISPD PNO
    cispd_end = time.perf_counter()
    cispd_time = cispd_end - cispd_start
    print("Generating CISPD took %.2f seconds" % cispd_time)

    print(f"PNO basis: {len(gs_orbs_original)}")
    print(f"CIS basis: {len(cis_orbs_original)}")
    print(f"CISPD basis: {len(cispd_orbs_original)}")


    # ----------- symmetric orthonormalized orbital set -----------
    cis_orbs = integrals.project_out(gs_orbs_original, cis_orbs_original)
    cis_orbs = integrals.orthonormalize(cis_orbs)
    cispd_orbs = integrals.project_out(gs_orbs_original + cis_orbs, cispd_orbs_original) # Cispd only project out the ground state orbitals

    orbitals_sym = gs_orbs_original + cis_orbs_original + cispd_orbs_original
    orbitals_sym = integrals.orthonormalize(orbitals_sym)

    orbitals_sym_active = orbitals_sym[1:]
    frozen = orbitals_sym[:1]
    n_orbitals_active = len(orbitals_sym_active)
    print("number of active orbitals: ", n_orbitals_active)
    print("nuclear repulsion: ", madpno.get_nuclear_repulsion())

    H_eff = integrals.compute_effective_hamiltonian(core_orbitals=frozen, active_orbitals=orbitals_sym_active, V=madpno.get_nuclear_potential(), energy_offset=madpno.get_nuclear_repulsion())
    c = H_eff[0]
    h = H_eff[1]
    g = H_eff[2]
    print("c: ", c)
    mol = tq.Molecule(geometry=geom, one_body_integrals=h, two_body_integrals=g, nuclear_repulsion=c, n_electrons= n_electrons_active, units='a', frozen_core=False)
    H_gs = mol.make_hamiltonian()

    print("len(active orbitals):", len(orbitals_sym_active))
    print("h shape:", h.shape) 
    print("g shape:", g.shape)
    print("mol.n_orbitals:", mol.n_orbitals)
    print("n_electrons_active: ", n_electrons_active)

    g_chem = g.transpose(0,2,1,3)
    e_roots, fcivecs = fci.direct_spin0.kernel(h, g_chem, n_orbitals_active, n_electrons_active, nroots=2)
    print("gs: ", e_roots[0])
    print("ex: ", e_roots[1])
    fci_energy_0 = e_roots[0] + c 
    fci_energy_1 = e_roots[1] + c 

    spa_edges = madpno.get_spa_edges()
    print("SPA edges: ", spa_edges)

    print("\n=============== SPA Calculation GS ===============\n")
    U = mol.make_ansatz(name="spa", edges=spa_edges)

    edge1 = spa_edges[0]
    edge2 = spa_edges[1]
    edge3 = spa_edges[2]
    edge4 = spa_edges[3]

    for i in range(len(edge1)):
        print(f"first edge i: {edge1[i]} ")

    for i in range(len(edge2)):
        print(f"second edge i: {edge2[i]} ")

    for i in range(len(edge3)):
        print(f"third edge i: {edge3[i]} ")

    for i in range(len(edge4)):
        print(f"fourth edge i: {edge4[i]} ")

    U += mol.UR(edge3[1], edge3[2], (tq.Variable('a') + 0.5) * pi)
    U += mol.UR(edge3[2], edge3[3], (tq.Variable('b') + 0.5) * pi)
    U += mol.UR(edge3[1], edge3[3], (tq.Variable('c') + 0.5) * pi)

    U += mol.UR(edge4[1], edge4[2], (tq.Variable('d') + 0.5) * pi)
    U += mol.UR(edge4[2], edge4[3], (tq.Variable('e') + 0.5) * pi)
    U += mol.UR(edge4[1], edge4[3], (tq.Variable('f') + 0.5) * pi)

    E = sun.SPAFP.decompose(H=H_gs, U=U)
    result = tq.minimize(E, silent=True, gradient="2-point", method_options={"finite_diff_rel_step":1.e-4})

    circuit_gs = tq.simulate(U, result.variables)


    print(f"FCI Ground state: {fci_energy_0}")
    print(f"SPA + UR GS energy: {result.energy}")
    print("SPA/FCI error: {:+2.5f}".format(result.energy-fci_energy_0))
    print(result.variables)
    print(f"Ground State Circuit: {circuit_gs}")

    gs_circuit = U.map_variables(result.variables)
    spa_energy_0 = result.energy


    # ----------- cholesky orthonormalized orbital set ------------------
    orbitals_ch = gs_orbs_original[:5] + cis_orbs_original + cispd_orbs_original + gs_orbs_original[5:] 
    orbitals_ch = integrals.orthonormalize(orbitals_ch, method="cholesky")

    overlap_frozen = integrals.compute_overlap_integrals([orbitals_sym[0]], [orbitals_ch[0]])
    print("frozen orbital overlap:", overlap_frozen)

    orbitals_ch_active = orbitals_ch[1:]

    print(fe.get_function_info(orbitals_ch_active))

    
    # ---------- rotate the circuit into excited state orbitals basis (cholesky orthonormalized set) ----------
    S = integrals.compute_overlap_integrals(orbitals_sym_active, orbitals_ch_active)
    rotation = mol.get_givens_circuit(S)

    # ----------- SPA excited state with cholesky orthonormalized orbital set-----------
    print("\n=============== SPA Calculation ES ===============\n")
    circuit_list = [gs_circuit]
    constants = [5.0]

    ex_spa_edges = madpno.get_spa_edges(orbitals=orbitals_ch)
    print("SPA edges: ", ex_spa_edges)

    U_ex = mol.make_ansatz(name="spa", edges=ex_spa_edges)

    ex_edge1 = ex_spa_edges[0]
    ex_edge2 = ex_spa_edges[1]
    ex_edge3 = ex_spa_edges[2]
    ex_edge4 = ex_spa_edges[3]

    for i in range(len(ex_edge1)):
        print(f"first ex_edge i: {ex_edge1[i]} ")

    for i in range(len(ex_edge2)):
        print(f"second ex_edge i: {ex_edge2[i]} ")

    for i in range(len(ex_edge3)):
        print(f"third edge i: {ex_edge3[i]} ")

    for i in range(len(ex_edge4)):
        print(f"fourth edge i: {ex_edge4[i]} ")

    U_ex += mol.UR(edge3[0], edge3[1], (tq.Variable('w') + 0.5) * pi)
    U_ex += mol.UR(edge3[0], edge3[2], (tq.Variable('x') + 0.5) * pi)

    U_ex += mol.UR(edge4[0], edge4[1], (tq.Variable('y') + 0.5) * pi)
    U_ex += mol.UR(edge4[0], edge4[2], (tq.Variable('z') + 0.5) * pi)

    ti = fe.TequilaInterface(mol=mol)
    E = ti.fastSPA_expectation_value_orthogonality_constraint(
        H=H_gs, # use ground state Hamiltonian and rotate the circuit into the different basis
        U=U_ex + rotation,
        circuit_list=circuit_list, 
        constant_list=constants
    )

    minimize_start = time.perf_counter()
    result = tq.minimize(E, silent=True)
    circuit_ex = tq.simulate(U_ex + rotation, result.variables)
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
    
    with open("spa_h2o.dat", "a") as f:
            f.write(f"{reported_distance:.3f} {dist_time:.2f} {fci_energy_0: .15f} {spa_energy_0: .15f} {fci_energy_1: .15f} {spa_energy_1: .15f}" + "\n")

    del integrals
    del madpno
    del ti
    del world
    del molecule

