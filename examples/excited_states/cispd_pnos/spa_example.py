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
"""

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

# ----------- symmetric orthonormalized orbital set -----------
cis_orbs = integrals.project_out(gs_orbs_original, cis_orbs_original)
cis_orbs = integrals.orthonormalize(cis_orbs)
cispd_orbs = integrals.project_out(gs_orbs_original + cis_orbs, cispd_orbs_original) # Cispd only project out the ground state orbitals

orbitals_sym = gs_orbs_original + cis_orbs + cispd_orbs
orbitals_sym = integrals.orthonormalize(orbitals_sym)

for i in range(len(orbitals_sym)):
    world.cube_plot(f"orb_symmetric{i}", orbitals_sym[i], molecule, zoom=4.0)

# ------------- create Hamiltonian from symmetric orbital set
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
print(energies)

# ----------- SPA ground state with symmetric orthonormalized orbital set for projection -----------
print("--------------- SPA Calculation GS -----------------")
U = mol.make_ansatz(name="spa", edges=[(0,1)])
E = tq.ExpectationValue(U=U, H=H_gs)
result = tq.minimize(E, silent=True)
print(f"FCI Ground state: {energies[0]}")
print(f"SPA GS energy: {result.energy}")
gs_circuit = U.map_variables(result.variables)

# ----------- cholesky orthonormalized orbital set ------------------
orbitals_ch = gs_orbs_original[:1] + cis_orbs + cispd_orbs + gs_orbs_original[1:] # 0: HF, 1: CIS, 2: CISPD, 3: MP2 PNO
orbitals_ch = integrals.orthonormalize(orbitals_ch, method="cholesky")

for i in range(len(orbitals_ch)):
    world.cube_plot(f"orb_cholesky{i}", orbitals_ch[i], molecule, zoom=4.0)


# ---------- rotate the circuit into excited state orbitals basis (cholesky orthonormalized set) ----------
S = integrals.compute_overlap_integrals(orbitals_sym, orbitals_ch)
rotation = mol.get_givens_circuit(S)

# ----------- SPA excited state with cholesky orthonormalized orbital set-----------
print("--------------- SPA Calculation EX -----------------")

U_ex = mol.make_ansatz(name="spa", edges=[(0,1,2,3)])#spa_edges)
U_ex = U_ex + rotation

UR = mol.UR(0, 3, (tq.Variable('a') + 0.25) * pi)
UR+= mol.UR(3, 1, (tq.Variable("b") + 0.25) *pi)
UR+= mol.UR(1, 2, (tq.Variable("c") + 0.25) *pi)

ansatz = U_ex + UR

circuit_list = [gs_circuit]
constants = [2.0]

# ----------- Consistency Test for rotation -----------
H_check = mol.transform_orbitals(S).make_hamiltonian()
E1 = tq.ExpectationValue(H=H_check, U=U)
E2 = tq.ExpectationValue(H=H_gs, U=U + rotation)
f1 = tq.compile(E1)
f2 = tq.compile(E2)
variables = {k:1.0 for k in U.extract_variables()}
print("difference: ", f1(variables) - f2(variables))

E = expectation_value_orthogonality_constraint(
    H=H_gs, # use ground state Hamiltonian and rotate the circuit into the different basis
    U=ansatz,
    circuit_list=circuit_list, 
    constant_list=constants
)

result = tq.minimize(E, silent=True)
print(f"FCI Singlet excited state energy: {energies[11]}")
print(f"SPA singlet excited state energy: {result.energy}")
circuit = tq.simulate(ansatz, result.variables)
print(f"Circuit: {circuit}")

E = expectation_value_orthogonality_constraint(H_check, U=U_ex + UR, circuit_list=circuit_list, constant_list=constants)
result = tq.minimize(E, silent=True)
print(f"FCI Singlet excited state energy: {energies[11]}")
print(f"SPA singlet excited state energy: {result.energy}")

circuit = tq.simulate(ansatz, result.variables)
print(f"Circuit: {circuit}")


# r0mol = deepcopy(mol)
# mUR = UR.map_variables({d:0 for d in UR.extract_variables()})
# core = len(mol.integral_manager.orbital_coefficients)-mol.n_orbitals
# rot = sun.measurement.gates_to_orb_rot(mUR,len(mol.integral_manager.orbital_coefficients),core=core)
# r0mol = r0mol.transform_orbitals(rot.T,ignore_active_space=True)
# sun.plot_MO(r0mol,filename='post')

fe.cleanup(globals())
