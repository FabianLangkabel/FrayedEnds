import numpy as np
import tequila as tq
import frayedends as fe
from pyscf import fci
from math import pi
import time

from pyscf import mcscf

n_electrons = 4
box_size = 50.0
wavelet_order = 7
madness_thresh = 1.0e-6
econv = 1.0e-6

geom = "Li 0.0 0.0 0.4028\nH 0.0 0.0 -1.2085"
molecule = fe.MolecularGeometry(geometry=geom, units='angstrom')
n_electrons_active = molecule.n_electrons - molecule.n_core_electrons
print("Active Electron: ", n_electrons_active)

world = fe.MadWorld(ndims=3, L=box_size, k=wavelet_order, thresh=madness_thresh)
integrals = fe.Integrals(world)

pno_start = time.perf_counter()
madpno = fe.MadPNO(world, geom, n_orbitals=4)
pno_end = time.perf_counter()
pno_time = pno_end - pno_start
print("Generating PNOs took %.2f seconds" % pno_time)

# ---------- original orbital set ------------
gs_orbs_original = madpno.get_orbitals() # HF + MP2
for i in range(len(gs_orbs_original)):
    # world.cube_plot(f"gs_orbs{i}", gs_orbs_original[i], molecule, zoom=4.0)
    gs_orbs_original[i].save_to_file(f"gs_orbs_original{i}.data")

cis_start = time.perf_counter()
cis_orbs_original = madpno.compute_cis(n_excitation=1) # CIS X Functions
for i in range(len(cis_orbs_original)):
    # world.cube_plot(f"cis_orbs{i}", cis_orbs_original[i], molecule, zoom=4.0)
    cis_orbs_original[i].save_to_file(f"cis_orbs_original{i}.data")
cis_end = time.perf_counter()
cis_time = cis_end - cis_start
print("Generating CIS took %.2f seconds" % cis_time)

cispd_start = time.perf_counter()
cispd_orbs_original = madpno.compute_cispd(n_orbitals=4) # CISPD PNO
for i in range(len(cispd_orbs_original)):
    # world.cube_plot(f"cispd_orbs{i}", cispd_orbs_original[i], molecule, zoom=4.0)
    cispd_orbs_original[i].save_to_file(f"cispd_orbs_original{i}.data")
cispd_end = time.perf_counter()
cispd_time = cispd_end - cispd_start
print("Generating CISPD took %.2f seconds" % cispd_time)

# gs_orbs_original = []
# for i in range(4):
#     gs_orbs_original.append(fe.SavedFct3D(f"gs_orbs_original{i}.data"))
# 
# cis_orbs_original = []
# cispd_orbs_original = []
# for i in range(1):
#    cis_orbs_original.append(fe.SavedFct3D(f"cis_orbs_original{i}.data"))
# for i in range(2):   
#    cispd_orbs_original.append(fe.SavedFct3D(f"cispd_orbs_original{i}.data"))
# 
# print("Orbitals loaded!")


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
e_roots, fcivecs = fci.direct_spin0.kernel(h, g_chem, n_orbitals_active, n_electrons_active, nroots=3)
print("gs: ", e_roots[0])
print("ex: ", e_roots[1])
e_ground_tot = e_roots[0] + c 
e_excited_tot = e_roots[1] + c 

spa_edges = madpno.get_spa_edges()
print("SPA edges: ", spa_edges)

print("\n=============== SPA Calculation GS ===============\n")
U = mol.make_ansatz(name="spa", edges=[(0,1,2,3,4,5)])

# U += mol.UR(1, 2, (tq.Variable('a') + 0.5) * pi)
U += mol.UR(1, 3, (tq.Variable('b') + 0.5) * pi)
# U += mol.UR(2, 3, (tq.Variable('c') + 0.5) * pi)
# U += mol.UR(1, 4, (tq.Variable('d') + 0.5) * pi)
# U += mol.UR(1, 5, (tq.Variable('e') + 0.5) * pi)
# U += mol.UR(2, 4, (tq.Variable('f') + 0.5) * pi)
# U += mol.UR(2, 5, (tq.Variable('g') + 0.5) * pi)
U += mol.UR(3, 4, (tq.Variable('h') + 0.5) * pi)
U += mol.UR(3, 5, (tq.Variable('i') + 0.5) * pi)
# U += mol.UR(4, 5, (tq.Variable('j') + 0.5) * pi)

E = tq.ExpectationValue(U=U, H=H_gs)
result = tq.minimize(E, silent=True)
circuit_gs = tq.simulate(U, result.variables)

print(f"FCI Ground state: {e_ground_tot}")
print(f"SPA + UR GS energy: {result.energy}")
print("SPA/FCI error: {:+2.5f}".format(result.energy-e_ground_tot))
print(result.variables)
print(f"Ground State Circuit: {circuit_gs}")

gs_circuit = U.map_variables(result.variables)

# ----------- cholesky orthonormalized orbital set ------------------
orbitals_ch = gs_orbs_original[:2] + cis_orbs_original + cispd_orbs_original + gs_orbs_original[2:] # 0: HF, 1: CIS, 2: CISPD, 3: MP2 PNO
orbitals_ch = integrals.orthonormalize(orbitals_ch, method="cholesky")
 
# ---------- rotate the circuit into excited state orbitals basis (cholesky orthonormalized set) ----------
S = integrals.compute_overlap_integrals(orbitals_sym, orbitals_ch)
rotation = mol.get_givens_circuit(S)

# ----------- SPA excited state with cholesky orthonormalized orbital set-----------
print("\n=============== SPA Calculation ES ===============\n")
circuit_list = [gs_circuit]
constants = [5.0]

U_ex = mol.make_ansatz(name="spa", edges=[(0,1,2,3,4,5)])

UR = mol.UR(0, 1, (tq.Variable('x') + 0.5) * pi)
UR += mol.UR(0, 2, (tq.Variable('y') + 0.5) * pi)
UR += mol.UR(0, 3, (tq.Variable('z') + 0.5) * pi)

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
print(f"minimize & simulate time: {minimize_end - minimize_start}")

print(f"FCI Singlet excited state energy: {e_excited_tot}")
print(f"SPA Singlet excited state energy: {result.energy}")
print("SPA/FCI error: {:+2.5f}".format(result.energy-e_excited_tot))
print(result.variables)
print(f"Excited State Circuit: {circuit_ex}")

fe.cleanup(globals())

