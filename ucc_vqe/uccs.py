import numpy as np
from pyscf import gto, scf
from pyblock2._pyscf.ao2mo import integrals as itg
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import UCC, HartreeFock
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.operators import FermionicOp

def fermionic_one_rdm_op(p, q, num_spin_orbitals):
    # FermionicOp Label: p^ q = creation at p, annihilation at q
    label = {f"+_{p} -_{q}": 1.0}
    return FermionicOp(label, num_spin_orbitals=num_spin_orbitals)

def fermionic_two_rdm_op(p, q, r, s, num_spin_orbitals):
    # FermionicOp Label: p^ q^ r s = creation at p and at q, annihilation at r and s
    label = {f"+_{p} +_{q} -_{r} -_{s}": 2.0}
    return FermionicOp(label, num_spin_orbitals=num_spin_orbitals)

def define_HeH_mole_object(number, basisset): 
    x = float(number)
    print(x)
    mol = gto.Mole()
    mol.atom = [["He", 0, 0, 0], ["H", x, 0, 0]]
    mol.unit = 'Bohr'
    mol.basis = basisset
    mol.charge = 0
    mol.spin = 1
    mol.build()
    return mol

def get_coefficients_and_integrals(mol):
    uhf=scf.UHF(mol)
    uhf.kernel()
    alpha_coeff=uhf.mo_coeff[0]
    beta_coeff=uhf.mo_coeff[1]
    ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_uhf_integrals(uhf,
    ncore=0, ncas=None, g2e_symm=1)
    return alpha_coeff, beta_coeff, ncas, n_elec, spin, ecore, h1e, g2e, orb_sym

#h1_a = np.load("/workspaces/MRA-OrbitalOptimization/H3lin_test/alpha_hmatrix.npy")
#h1_b = np.load("/workspaces/MRA-OrbitalOptimization/H3lin_test/beta_hmatrix.npy")

#h2_aa = np.load("/workspaces/MRA-OrbitalOptimization/H3lin_test/alpha_alpha_gtensor.npy")
#h2_bb = np.load("/workspaces/MRA-OrbitalOptimization/H3lin_test/beta_beta_gtensor.npy")
#h2_ba = np.load("/workspaces/MRA-OrbitalOptimization/H3lin_test/alpha_beta_gtensor.npy")

#=== Initial Integrale aus UHF holen =====
HeH_geom = define_HeH_mole_object(2.2, 'sto-3g')
alpha_coeff, beta_coeff, ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = get_coefficients_and_integrals(HeH_geom)
h1_a = h1e[0]
h1_b = h1e[1]
h2_aa = g2e[0]
h2_bb = g2e[2]
h2_ba = g2e[1]

# === Hamiltonian erstellen ===
hamiltonian = ElectronicEnergy.from_raw_integrals(h1_a, h2_aa, h1_b, h2_bb, h2_ba)
hamiltonian.nuclear_repulsion_energy = 0.9090909090909086 #for H3lin 1.8 Bohr

# === Problem erzeugen ===
problem = ElectronicStructureProblem(hamiltonian)
problem.num_particles = (2, 1)


# === Mapper ===
mapper = JordanWignerMapper()

# === Ansatz ===
ansatz = UCC(
    qubit_mapper=mapper,
    num_particles=(2, 1),
    num_spatial_orbitals=2,
    excitations='s',
    initial_state=HartreeFock(
        problem.num_spatial_orbitals,
        problem.num_particles,
        mapper
    )
)
estimator = Estimator()
optimizer = SLSQP()
vqe = VQE(estimator, ansatz, optimizer)
vqe.initial_point = np.zeros(ansatz.num_parameters)


qubit_op = mapper.map(problem.second_q_ops()[0])
vqe_result = vqe.compute_minimum_eigenvalue(operator=qubit_op)
optimal_params = vqe_result.optimal_point
print("Optimale VQE-Parameter:", optimal_params)
total_energy = vqe_result.eigenvalue.real + problem.nuclear_repulsion_energy
print("Gesamtenergie (VQE):", total_energy)



rdm1 = np.zeros((problem.num_spin_orbitals, problem.num_spin_orbitals), dtype=complex)
estimator = Estimator()
 #Schleife über alle Spin-Orbitale
for p in range(problem.num_spin_orbitals):
    for q in range(problem.num_spin_orbitals):
        ferm_op = fermionic_one_rdm_op(p, q, problem.num_spin_orbitals)
        qubit_op = mapper.map(ferm_op)
        # Erwartungswert messen
        exp_val = estimator.run([ansatz], [qubit_op], [optimal_params]).result().values[0]
        rdm1[p, q] = exp_val
print(rdm1)

rdm1_alpha = np.zeros((problem.num_spatial_orbitals, problem.num_spatial_orbitals), dtype=complex)
rdm1_beta = np.zeros((2, 2), dtype=complex)
for i in range(0, 2, 1):
    for j in range(0, 2, 1):
        rdm1_alpha[i, j] = rdm1[i, j]
        rdm1_beta[i, j] = rdm1[i+2, j+2]

print(rdm1_alpha)
print(np.trace(rdm1_alpha))
print(rdm1_beta)
print(np.trace(rdm1_beta))
