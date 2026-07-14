import numpy as np
import tequila as tq
import frayedends as fe
from math import pi
import time

"""
#test givens
mol = tq.Molecule(geometry="H 0.0 0.0 0.0\nH 0.0 0.0 1.0", basis_set="sto-3g").use_native_orbitals()
U = mol.make_ansatz(name="SPA", edges=[(0,1)])
opt = tq.quantumchemistry.optimize_orbitals(molecule=mol, circuit=U, silent=True)
c = opt.mo_coeff
HT = opt.molecule.make_hamiltonian()
H = mol.make_hamiltonian()
UT = U + mol.get_givens_circuit(unitary=c)

HT2 = mol.transform_orbitals(c).make_hamiltonian()
print(" operator = " , HT - HT2)
E1 = tq.ExpectationValue(H=HT, U=U)
E2 = tq.ExpectationValue(H=H, U=UT)
f1 = tq.compile(E1)
f2 = tq.compile(E2)
variables = {k:1.0 for k in U.extract_variables()}
print(f1(variables) - f2(variables))

for _ in range(100):
    opt = tq.quantumchemistry.optimize_orbitals(molecule=mol, circuit=U, initial_guess="random",silent=True)
    if not np.isclose(opt.energy, result.energy, atol=1.e-3):
        print("opt.energy = ", opt.energy)
        print(opt.mo_coeff)
"""

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
# gs_orbs_original = madpno.get_orbitals() # HF + MP2
# for i in range(len(gs_orbs_original)):
#     gs_orbs_original[i].save_to_file(f"gs_orbs_original{i}.data")
# 
# cis_start = time.perf_counter()
# cis_orbs_original = madpno.compute_cis(n_excitation=1) # CIS X Functions
# cis_end = time.perf_counter()
# cis_time = cis_end - cis_start
# print("Generating CIS took %.2f seconds" % cis_time)
# for i in range(len(cis_orbs_original)):
#     cis_orbs_original[i].save_to_file(f"cis_orbs_original{i}.data")
# 
# cispd_start = time.perf_counter()
# cispd_orbs_original = madpno.compute_cispd(n_orbitals=2) # CISPD PNO
# cispd_end = time.perf_counter()
# cispd_time = cispd_end - cispd_start
# print("Generating CISPD took %.2f seconds" % cispd_time)
# for i in range(len(cis_orbs_original)):
#     cispd_orbs_original[i].save_to_file(f"cispd_orbs_original{i}.data")

gs_orbs_original = []
for i in range(2):
    gs_orbs_original.append(fe.SavedFct3D(f"gs_orbs_original{i}.data"))

cis_orbs_original = []
cispd_orbs_original = []
for i in range(1):
   cis_orbs_original.append(fe.SavedFct3D(f"cis_orbs_original{i}.data"))
   cispd_orbs_original.append(fe.SavedFct3D(f"cispd_orbs_original{i}.data"))

print("Orbitals loaded!")

# ----------- symmetric orthonormalized orbital set -----------
cis_orbs = integrals.project_out(gs_orbs_original, cis_orbs_original)
cis_orbs = integrals.orthonormalize(cis_orbs)
cispd_orbs = integrals.project_out(gs_orbs_original + cis_orbs, cispd_orbs_original) # Cispd only project out the ground state orbitals

orbitals_sym = gs_orbs_original + cis_orbs_original + cispd_orbs_original
orbitals_sym = integrals.orthonormalize(orbitals_sym)

# for i in range(len(orbitals_sym)):
#     world.cube_plot(f"orb_symmetric{i}", orbitals_sym[i], molecule, zoom=4.0)

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
# print(energies)

# ----------- SPA ground state with symmetric orthonormalized orbital set for projection -----------
print("\n=============== SPA Calculation GS ===============\n")
U = mol.make_ansatz(name="spa", edges=[(0,1,2,3)])

E = tq.ExpectationValue(U=U, H=H_gs)
result = tq.minimize(E, silent=True)
circuit_gs = tq.simulate(U, result.variables)

print("--------------- SPA without UR:  GS -----------------")
print(f"FCI Ground state: {energies[0]}")
print(f"SPA GS energy: {result.energy}")
print("SPA/FCI error: {:+2.5f}".format(result.energy-energies[0]))
print(f"Ground State Circuit: {circuit_gs}")

# ------------------------------------------------------------------

# U += mol.UR(0, 1, (tq.Variable('a') + 0.5) * pi)
U += mol.UR(1, 2, (tq.Variable("b") + 0.5) * pi)
U += mol.UR(2, 3, (tq.Variable("c") + 0.5) * pi)
# U += mol.UR(0, 3, (tq.Variable("d") + 0.5) * pi)
U += mol.UR(1, 3, (tq.Variable("e") + 0.5) * pi)
# U += mol.UR(0, 2, (tq.Variable("f") + 0.5) * pi)

E = tq.ExpectationValue(U=U, H=H_gs)
result = tq.minimize(E, silent=True)
circuit_gs = tq.simulate(U, result.variables)

print("--------------- SPA + UR:  GS -----------------")
print(f"FCI Ground state: {energies[0]}")
print(f"SPA + UR GS energy: {result.energy}")
print("SPA/FCI error: {:+2.5f}".format(result.energy-energies[0]))
print(result.variables)
print(f"Ground State Circuit: {circuit_gs}")

gs_circuit = U.map_variables(result.variables)

# ----------- cholesky orthonormalized orbital set ------------------
orbitals_ch = gs_orbs_original[:1] + cis_orbs_original + cispd_orbs_original + gs_orbs_original[1:] # 0: HF, 1: CIS, 2: CISPD, 3: MP2 PNO
orbitals_ch = integrals.orthonormalize(orbitals_ch, method="cholesky")

# for i in range(len(orbitals_ch)):
#     world.cube_plot(f"orb_cholesky{i}", orbitals_ch[i], molecule, zoom=4.0)


# ---------- rotate the circuit into excited state orbitals basis (cholesky orthonormalized set) ----------
S = integrals.compute_overlap_integrals(orbitals_sym, orbitals_ch)
rotation = mol.get_givens_circuit(S)

# ----------- SPA excited state with cholesky orthonormalized orbital set-----------
print("\n=============== SPA Calculation ES ===============\n")
circuit_list = [gs_circuit]
constants = [5.0]

U_ex = mol.make_ansatz(name="spa", edges=[(0,1,2,3)])

ti = fe.TequilaInterface(mol=mol)
E = ti.expectation_value_orthogonality_constraint(
    H=H_gs, # use ground state Hamiltonian and rotate the circuit into the different basis
    U=U_ex + rotation,
    circuit_list=circuit_list, 
    constant_list=constants
)
result = tq.minimize(E, silent=True)
circuit_ex = tq.simulate(U_ex + rotation, result.variables)

print("--------------- SPA without UR:  ES -----------------")
print(f"FCI Singlet excited state energy: {energies[9]}")
print(f"SPA Singlet excited state energy: {result.energy}")
print("SPA/FCI error: {:+2.5f}".format(result.energy-energies[9]))
print(result.variables)
print(f"Excited State Circuit: {circuit_ex}")

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
result = tq.minimize(E, silent=True)
circuit_ex = tq.simulate(U_ex + UR + rotation, result.variables)
print("--------------- SPA + UR:  ES -----------------")
print(f"FCI Singlet excited state energy: {energies[9]}")
print(f"SPA Singlet excited state energy: {result.energy}")
print("SPA/FCI error: {:+2.5f}".format(result.energy-energies[9]))
print(result.variables)
print(f"Excited State Circuit: {circuit_ex}")

# ----------- Consistency Test for rotation -----------
print("\n=============== Consistency Test for rotation ===============\n")
H_check = mol.transform_orbitals(S).make_hamiltonian()
E1 = tq.ExpectationValue(H=H_check, U=U_ex)
E2 = tq.ExpectationValue(H=H_gs, U=U_ex + rotation)
f1 = tq.compile(E1)
f2 = tq.compile(E2)
variables = {k:1.0 for k in U_ex.extract_variables()}
print("Consistency Test difference: ", f1(variables) - f2(variables))

print("\n--------------- SPA H_gs & U_ex + UR + rotation:  ES -----------------")
E = ti.expectation_value_orthogonality_constraint(
    H=H_gs, # use ground state Hamiltonian and rotate the circuit into the different basis
    U=U_ex + UR + rotation,
    circuit_list=circuit_list, 
    constant_list=constants
)

result = tq.minimize(E, silent=True)
print(f"FCI Singlet EX: {energies[9]}")
print(f"SPA Singlet EX: {result.energy}")
circuit = tq.simulate(U_ex + UR + rotation, result.variables)
print(f"Circuit: {circuit}")

print("--------------- SPA H_check & U_ex + UR:  ES -----------------")
gs_circuit_ch = gs_circuit + rotation # add rotation to groundstate circuit 
E = ti.expectation_value_orthogonality_constraint(
    H=H_check, 
    U=U_ex + UR, 
    circuit_list=[gs_circuit_ch], 
    constant_list=constants
)
result = tq.minimize(E, silent=True)
print(f"FCI Singlet EX: {energies[9]}")
print(f"SPA Singlet EX: {result.energy}")

circuit = tq.simulate(U_ex + UR, result.variables)
print(f"Circuit: {circuit}")

fe.cleanup(globals())
