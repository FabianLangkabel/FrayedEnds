import numpy as np
import tequila as tq
import frayedends as fe
from math import pi
import time


iterations = 6
box_size = 50.0
wavelet_order = 7
madness_thresh = 1.0e-6
econv = 1.0e-6


molecule = fe.MolecularGeometry(units="angstrom")
molecule.add_atom(0.0, 0.0, -0.5, "H")
molecule.add_atom(0.0, 0.0, 0.5, "H")
geom = "H 0.0 0.0 -0.5\nH 0.0 0.0 0.5"

world = fe.MadWorld(ndims=3, L=box_size, k=wavelet_order, thresh=madness_thresh)
integrals = fe.Integrals(world)

pno_start = time.perf_counter()
madpno = fe.MadPNO(world, geom, n_orbitals=2)
pno_end = time.perf_counter()
pno_time = pno_end - pno_start
print("Generating PNOs took %.2f seconds" % pno_time)

# ---------- original orbital set ------------
gs_orbs_original = madpno.get_orbitals() # HF + MP2
for i in range(len(gs_orbs_original)):
    gs_orbs_original[i].save_to_file(f"gs_orbs_original{i}.data")

cis_start = time.perf_counter()
cis_orbs_original = madpno.compute_cis(n_excitation=1) # CIS X Functions
cis_end = time.perf_counter()
cis_time = cis_end - cis_start
print("Generating CIS took %.2f seconds" % cis_time)
for i in range(len(cis_orbs_original)):
    cis_orbs_original[i].save_to_file(f"cis_orbs_original{i}.data")

cispd_start = time.perf_counter()
cispd_orbs_original = madpno.compute_cispd(n_orbitals=2) # CISPD PNO
cispd_end = time.perf_counter()
cispd_time = cispd_end - cispd_start
print("Generating CISPD took %.2f seconds" % cispd_time)
for i in range(len(cis_orbs_original)):
    cispd_orbs_original[i].save_to_file(f"cispd_orbs_original{i}.data")


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

mol = tq.Molecule(geometry=geom, one_body_integrals=T+V, two_body_integrals=G, nuclear_repulsion=c)
H_gs = mol.make_hamiltonian()

print("--- FCI Ground State with H_gs----")
energies, eivect = np.linalg.eigh(H_gs.to_matrix())
for i in range(20):#range(len(eivect)):
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
result_gs = tq.minimize(E, silent=True)
circuit_gs = tq.simulate(U, result_gs.variables)

print(f"FCI Ground state: {energies[0]}")
print(f"SPA + UR GS energy: {result_gs.energy}")
print("SPA/FCI error: {:+2.5f}".format(result_gs.energy-energies[0]))
print(result_gs.variables)
print(f"Ground State Circuit: {circuit_gs}")

gs_circuit = U.map_variables(result_gs.variables)

# ----------- cholesky orthonormalized orbital set ------------------
orbitals_ch = gs_orbs_original[:1] + cis_orbs_original + cispd_orbs_original + gs_orbs_original[1:] # 0: HF, 1: CIS, 2: CISPD, 3: MP2 PNO
orbitals_ch = integrals.orthonormalize(orbitals_ch, method="cholesky")

# ---------- rotate the circuit into excited state orbitals basis (cholesky orthonormalized set) ----------
S = integrals.compute_overlap_integrals(orbitals_sym, orbitals_ch)
rotation = mol.get_givens_circuit(S)

# ----------- SPA excited state with cholesky orthonormalized orbital set-----------
print("\n=============== SPA Calculation ES ===============\n")
circuit_list = [gs_circuit]
constants = [5.0]

U_ex = mol.make_ansatz(name="spa", edges=[(0,1,2,3)])

ti = fe.TequilaInterface(mol=mol)

UR = mol.UR(0, 1, (tq.Variable('u') + 0.5) * pi)
# UR += mol.UR(1, 2, (tq.Variable('v') + 0.5) * pi)
# UR += mol.UR(2, 3, (tq.Variable('w') + 0.5) * pi)
# UR += mol.UR(0, 3, (tq.Variable('x') + 0.5) * pi)
# UR += mol.UR(1, 3, (tq.Variable("y") + 0.5) * pi)
UR += mol.UR(0, 2, (tq.Variable("z") + 0.5) * pi)


E = ti.expectation_value_orthogonality_constraint(
    H=H_gs, # use ground state Hamiltonian and rotate the circuit into the different basis
    U=U_ex + UR + rotation,
    circuit_list=circuit_list, 
    constant_list=constants
)
result_ex = tq.minimize(E, silent=True)
circuit_ex = tq.simulate(U_ex + UR + rotation, result_ex.variables)
print("--------------- SPA + UR:  ES -----------------")
print(f"FCI Singlet excited state energy: {energies[9]}")
print(f"SPA Singlet excited state energy: {result_ex.energy}")
print("SPA/FCI error: {:+2.5f}".format(result_ex.energy-energies[9]))
print(result_ex.variables)
print(f"Excited State Circuit: {circuit_ex}")

ex_circuit = (U_ex + UR + rotation).map_variables(result_ex.variables)

tq.circuit.export_to(gs_circuit, filename="gs_spa_ur_opt.pdf")
tq.circuit.export_to(ex_circuit, filename="ex_spa_ur_rotation_opt.pdf")

combined_circuit = gs_circuit + ex_circuit

tq.circuit.export_to(combined_circuit, filename="gs_and_ex_spa_ur_rotation_opt.pdf")