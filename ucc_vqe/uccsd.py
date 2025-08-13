import numpy as np
from pyscf import gto, scf
from pyblock2._pyscf.ao2mo import integrals as itg
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
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
    label = {f"+_{p} +_{q} -_{s} -_{r}": 1.0}
    return FermionicOp(label, num_spin_orbitals=num_spin_orbitals)

def define_H3lin_mole_object(number, basisset): 
    x = float(number) * -1
    y = float(number)
    mol = gto.Mole()
    mol.atom = [["H", 0, 0, 0], ["H", y, 0, 0], ["H", x, 0, 0]]
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
H3lin_geom = define_H3lin_mole_object(1.8, 'sto-3g')
alpha_coeff, beta_coeff, ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = get_coefficients_and_integrals(H3lin_geom)
h1_a = h1e[0]
h1_b = h1e[1]
h2_aa = g2e[0]
h2_bb = g2e[2]
h2_ab = g2e[1]
h2_ba = h2_ab.transpose(2,3,0,1)


# === Hamiltonian erstellen ===
hamiltonian = ElectronicEnergy.from_raw_integrals(h1_a, h2_aa, h1_b, h2_bb, h2_ba)
hamiltonian.nuclear_repulsion_energy = 1.3888888888888888 #for H3lin 1.8 Bohr

# === Problem erzeugen ===
problem = ElectronicStructureProblem(hamiltonian)
problem.num_particles = (2, 1)


# === Mapper ===
mapper = JordanWignerMapper()

# === Ansatz ===
ansatz = UCCSD(
    problem.num_spatial_orbitals,
    problem.num_particles,
    mapper,
    reps=1,
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



rdm1 = np.zeros((problem.num_spin_orbitals, problem.num_spin_orbitals), dtype=float)
estimator = Estimator()
 #Schleife über alle Spin-Orbitale
for p in range(problem.num_spin_orbitals):
    for q in range(problem.num_spin_orbitals):
        ferm_op = fermionic_one_rdm_op(p, q, problem.num_spin_orbitals)
        qubit_op = mapper.map(ferm_op)
        # Erwartungswert messen
        exp_val = estimator.run([ansatz], [qubit_op], [optimal_params]).result().values[0]
        rdm1[p, q] = exp_val
#print(rdm1)

rdm1_alpha = np.zeros((problem.num_spatial_orbitals, problem.num_spatial_orbitals), dtype=float)
rdm1_beta = np.zeros((problem.num_spatial_orbitals, problem.num_spatial_orbitals), dtype=float)
for i in range(0, 3, 1):
    for j in range(0, 3, 1):
        rdm1_alpha[i, j] = rdm1[i, j]
        rdm1_beta[i, j] = rdm1[i+3, j+3]

print(rdm1_alpha)
print(np.trace(rdm1_alpha))
print(rdm1_beta)
print(np.trace(rdm1_beta))

rdm2 = np.zeros((problem.num_spin_orbitals, problem.num_spin_orbitals, problem.num_spin_orbitals, problem.num_spin_orbitals), dtype=float)
estimator = Estimator()
for p in range(problem.num_spin_orbitals):
    for q in range(problem.num_spin_orbitals):
        for r in range(problem.num_spin_orbitals):
            for s in range(problem.num_spin_orbitals):
                ferm_op_rdm2 = fermionic_two_rdm_op(p, q, r, s, problem.num_spin_orbitals)
                qubit_op_rdm2 = mapper.map(ferm_op_rdm2)
                exp_val_2 = estimator.run([ansatz], [qubit_op_rdm2], [optimal_params]).result().values[0]
                rdm2[p, q, r, s] = exp_val_2
#print(rdm2)

rdm2_aa = np.zeros((problem.num_spatial_orbitals, problem.num_spatial_orbitals, problem.num_spatial_orbitals, problem.num_spatial_orbitals), dtype=float)
rdm2_bb = np.zeros((problem.num_spatial_orbitals, problem.num_spatial_orbitals, problem.num_spatial_orbitals, problem.num_spatial_orbitals), dtype=float)
rdm2_ba = np.zeros((problem.num_spatial_orbitals, problem.num_spatial_orbitals, problem.num_spatial_orbitals, problem.num_spatial_orbitals), dtype=float)
for i in range(0, 3, 1):
    for j in range(0, 3, 1):
        for k in range(0, 3, 1):
            for l in range(0, 3, 1):
                rdm2_aa[i, j, l, k] = rdm2[i, j, k, l]
                rdm2_ba[j, i, k, l] = rdm2[i+3, j, k+3, l]
                rdm2_bb[i, j, l, k] = rdm2[i+3, j+3, k+3, l+3]

#rdm2_ab = rdm2_ba.transpose(2,3,0,1)
#print("Alpha alpha 2-rdm:")
#print(rdm2_aa)
print("Alpha Beta 2-rdm:")
print(rdm2_ba)
#print("Beta beta 2-rdm:")
#print(rdm2_bb)

#====rdms auf Platte schreiben=====
#np.save("/workspaces/MRA-OrbitalOptimization/H3lin_vqe_test/1_8_alpha_1rdm_vqe.npy", rdm1_alpha)
#np.save("/workspaces/MRA-OrbitalOptimization/H3lin_vqe_test/1_8_beta_1rdm_vqe.npy", rdm1_beta)
#np.save("/workspaces/MRA-OrbitalOptimization/H3lin_vqe_test/1_8_alpha_alpha_2rdm_vqe.npy", rdm2_aa)
#np.save("/workspaces/MRA-OrbitalOptimization/H3lin_vqe_test/1_8_alpha_beta_2rdm_vqe.npy", rdm2_ab)
#np.save("/workspaces/MRA-OrbitalOptimization/H3lin_vqe_test/1_8_beta_beta_2rdm_vqe.npy", rdm2_bb)


#====Überprüfe die Energie per Hand=====
aa_2rdm_phys = rdm2_aa.transpose(0, 3, 1, 2)
aa_2rdm_phys = np.swapaxes(aa_2rdm_phys, 1, 2)
ba_2rdm_phys = rdm2_ba.transpose(0, 3, 1, 2)
ba_2rdm_phys = np.swapaxes(ba_2rdm_phys, 1, 2)
bb_2rdm_phys = rdm2_bb.transpose(0, 3, 1, 2)
bb_2rdm_phys = np.swapaxes(bb_2rdm_phys, 1, 2)

one_body_en = np.einsum('ij,ij->', rdm1_alpha, h1_a) + np.einsum('ij,ij->', rdm1_beta, h1_b)
two_body_en = 0.5 * (np.einsum('ijkl,ikjl->', aa_2rdm_phys, h2_aa) + 2 * np.einsum('ijkl,ikjl->', ba_2rdm_phys, h2_ab) + 
                     np.einsum('ijkl,ikjl->', bb_2rdm_phys, h2_bb))
tot_en = one_body_en + two_body_en + 1.3888888888888888
print("1-body energy: ", one_body_en)
print("2 body energy: ", two_body_en)
print("Total energy: ", tot_en)