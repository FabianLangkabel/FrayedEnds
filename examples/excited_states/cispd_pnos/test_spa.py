import numpy as np
import tequila as tq
import frayedends as fe
import sunrise as sun
from math import pi
from copy import deepcopy

n_electrons = 4
number_roots = 3
iterations = 6
box_size = 50.0
wavelet_order = 7
madness_thresh = 0.0001
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

# molecule = fe.MolecularGeometry(units="bohr")
# molecule.add_atom(0.0, 0.0, -1.0, "H")
# molecule.add_atom(0.0, 0.0, 0.0, "Be")
# molecule.add_atom(0.0, 0.0, 1.0, "H")
# geom = "H 0.0 0.0 -1.0\nBe 0.0 0.0 0.0\nH 0.0 0.0 1.0"

molecule = fe.MolecularGeometry(units="angstrom")
molecule.add_atom(0.0, 0.0, -0.5, "H")
molecule.add_atom(0.0, 0.0, 0.5, "H")
geom = "H 0.0 0.0 -0.5\nH 0.0 0.0 0.5"

world = fe.MadWorld3D(L=box_size, k=wavelet_order, thresh=madness_thresh)
madpno = fe.MadPNO(world, geom, n_orbitals=2)
integrals = fe.Integrals3D(world)
orbitals = madpno.get_orbitals()
#print(madpno.get_pno_groupings())
# print(madpno.get_spa_edges())

cis_orbs = madpno.compute_cis(n_excitation=1) 

cispd_orbs = madpno.compute_cispd(n_orbitals=2)
#cispd_orbs = madpno.orthonormalize_cispd(integrals_obj=integrals)

# orbitals = orbitals[:1] + cis_orbs + cispd_orbs  + orbitals[1:] 
orbitals = orbitals + cis_orbs + cispd_orbs
orbitals = integrals.orthonormalize(orbitals, method="cholesky")
T = integrals.compute_kinetic_integrals(orbitals)
V = integrals.compute_potential_integrals(orbitals, V=madpno.get_nuclear_potential())
G = integrals.compute_two_body_integrals(orbitals)
c = madpno.get_nuclear_repulsion()

mol = tq.Molecule(geometry=geom, one_body_integrals=T+V, two_body_integrals=G, nuclear_repulsion=c)
H = mol.make_hamiltonian()

print(madpno.get_pno_groupings())
print(madpno.get_spa_edges())

#mol = tq.Molecule(geometry=geom, basis_set="sto-3g")
#sun.plot_MO(mol, 'pre')

# H = mol.make_hamiltonian()
energies, eivect = np.linalg.eigh(H.to_matrix())
for i in range(len(eivect)):
    w = tq.QubitWaveFunction.from_array(eivect[:,i])
    t = [i for i in w.items()]
    if [i for i in w.items()][0][0].binary.count('1') == mol.n_electrons:
        print(energies[i],'-->',w)
print(energies)
# Ergebnisse ausgeben
print(f"FCI Ground state: {energies[0]}")

# SPA Groundstate for projection
spa_edges = madpno.get_spa_edges() # (0,1,2,3) 0: HF, 1: MP2 PNO, 2: CIS X, 3: CISPD 

U = mol.make_ansatz(name="spa", edges=[(0,1)])
E = tq.ExpectationValue(U=U, H=H)
result = tq.minimize(E, silent=True)
# print(f"FCI Ground state: {energies[0]}")
# print(f"SPA GS energy: {result.energy}")
gs_circuit = U.map_variables(result.variables)

print("--------------- SPA Calculation EX -----------------")

U = mol.make_ansatz(name="spa", edges=[(0,1,2,3)])#spa_edges)
a = tq.Variable('a')
UR = mol.UR(0, 1, (a + 0.25) * pi)
UR+= mol.UR(1,2, (tq.Variable("b") + 0.25) *pi)
#UR+= mol.UR(2,3, (tq.Variable("c") + 0.1) *pi)


ansatz = U + UR

circuit_list = [gs_circuit]
constants = [2.0]

E = expectation_value_orthogonality_constraint(
    H=H, 
    U=ansatz, 
    circuit_list=circuit_list, 
    constant_list=constants
)

result = tq.minimize(E, silent=True)
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
