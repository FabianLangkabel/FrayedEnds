import os
import numpy as np
import frayedends as fe

# ---------------------------------------------------------------------------
# Parallelisierung
# ---------------------------------------------------------------------------
os.environ["MAD_NUM_THREADS"] = "28"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "28"
DMRG_THREADS = 28
DMRG_STACK_MEM = int(30 * 1024**3)   # 30 GB

# ---------------------------------------------------------------------------
# MRA-Parameter
# ---------------------------------------------------------------------------
box_size = 54.686
wavelet_order = 7
madness_thresh = 1e-7

# ---------------------------------------------------------------------------
# Aktivraum-Konfiguration
# ---------------------------------------------------------------------------
input_orb_path = "initial_orbitals"
translated_orb_path = "translated"
output_orb_path = "it_1"

n_total_occupied = 21                                       # 42 / 2
active_occupied_indices = [16, 19, 20]
active_virtual_indices  = [6, 7, 32]
core_occupied_indices = [i for i in range(n_total_occupied)
                         if i not in active_occupied_indices]

n_active_orbitals = len(active_occupied_indices) + len(active_virtual_indices)
n_active_electrons = 2 * len(active_occupied_indices)       # closed shell

# ---------------------------------------------------------------------------
# Geometrie (Cabana 1974, D6h)
# ---------------------------------------------------------------------------
world = fe.MadWorld(ndims=3, L=box_size, k=wavelet_order, thresh=madness_thresh)

mol = fe.MolecularGeometry("""
C    0.98740391     0.98740391     0.00000000
C   -0.36141491     1.34881882     0.00000000
C   -1.34881882     0.36141491     0.00000000
C   -0.98740391    -0.98740391     0.00000000
C    0.36141491    -1.34881882     0.00000000
C    1.34881882    -0.36141491     0.00000000
H    1.75327126     1.75327126     0.00000000
H   -0.64174182     2.39501309     0.00000000
H   -2.39501309     0.64174182     0.00000000
H   -1.75327126    -1.75327126     0.00000000
H    0.64174182    -2.39501309     0.00000000
H    2.39501309    -0.64174182     0.00000000
""", units="angstrom", silent=True)

Vnuc = mol.get_vnuc(world)
nuclear_repulsion_energy = mol.get_nuclear_repulsion()

# ---------------------------------------------------------------------------
# Orbitale laden
# ---------------------------------------------------------------------------
def load(path):
    return world.load_savedfct_from_m_file(path)

core_orbs = []

for i in core_occupied_indices:
    print(f"Core  occupied_{i}")
    orb = load(os.path.join(input_orb_path, f"occupied_{i}"))
    core_orbs.append(orb)

active_orbs = []
for i in active_occupied_indices:
    print(f"Active occupied_{i}")
    orb = load(os.path.join(input_orb_path, f"occupied_{i}"))
    active_orbs.append(orb)

for i in active_virtual_indices:
    print(f"Active virtual_{i}")
    orb = load(os.path.join(input_orb_path, f"virtual_{i}"))
    active_orbs.append(orb)

# ---------------------------------------------------------------------------
# Aktivraum-Integrale (FC fuer h1 -- wie Ethan-Beispiel)
# ---------------------------------------------------------------------------
integrals = fe.Integrals(world)
T_act = integrals.compute_kinetic_integrals(active_orbs)
V_act = integrals.compute_potential_integrals(active_orbs, Vnuc)
G_act = integrals.compute_two_body_integrals(active_orbs, ordering="chem").elems
FC_act = integrals.compute_frozen_core_interaction(core_orbs, active_orbs)
h1 = T_act + V_act + FC_act

# ---------------------------------------------------------------------------
# c (= E_nuc + E_core) via OrbitalRefinement.get_c() -- Ethan-Style.
# Dazu einmal optimize_orbs(maxiter=0) mit HF-RDM des Aktivraums ([2,2,2,0,0,0])
# aufrufen; calculate_core_energy() wird intern getriggert und get_c() liefert
# das skalare c. Die rotate_orbitals_back-Phase stellt die urspruengliche
# Orbital-Reihenfolge wieder her, c ist rotationsinvariant.
# ---------------------------------------------------------------------------
n_doubly_occ = len(active_occupied_indices)
occ = np.zeros(n_active_orbitals)
occ[:n_doubly_occ] = 1.0
rdm1_HF = 2.0 * np.diag(occ)
rdm2_HF = (np.einsum("ik,jl->ijkl", rdm1_HF, rdm1_HF)
           - 0.5 * np.einsum("il,jk->ijkl", rdm1_HF, rdm1_HF))

opti = fe.OrbitalRefinement(world, Vnuc, nuclear_repulsion_energy)
_, _ = opti.refine_orbitals(
    orbitals=[core_orbs, active_orbs],
    rdm1=rdm1_HF,
    rdm2=rdm2_HF,
    opt_thresh=1.0,
    occ_thresh=1.0,
    maxiter=0,
    orthonormalization_method="cd",
    refine_core=False,
)
c = opti.get_c()
print(f"\nE_nuc      = {nuclear_repulsion_energy:.10f}")
print(f"c (ecore)  = {c:.10f}   (= E_nuc + E_core)")
print(f"E_core     = {c - nuclear_repulsion_energy:.10f}")

# ---------------------------------------------------------------------------
# DMRG (SU2, Singulett)
# ---------------------------------------------------------------------------
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

driver = DMRGDriver(
    scratch="./tmp",
    symm_type=SymmetryTypes.SU2,
    n_threads=DMRG_THREADS,
    stack_mem=DMRG_STACK_MEM,
)
driver.initialize_system(
    n_sites=n_active_orbitals,
    n_elec=n_active_electrons,
    spin=0,
)

# bond_dim konstant 500; Noise nur fuer die ersten ~10 Sweeps
bond_dims = [500]
noises = [1e-4] * 4 + [1e-5] * 4 + [1e-6] * 2 + [0]
thrds = [1e-10]
n_sweeps = 50

mpo = driver.get_qc_mpo(h1e=h1, g2e=G_act, ecore=c, iprint=1)
ket = driver.get_random_mps(tag="GS", bond_dim=50, nroots=1)
energy = driver.dmrg(mpo, ket, n_sweeps=n_sweeps, bond_dims=bond_dims,
                     noises=noises, thrds=thrds, iprint=1)
print(f"\nDMRG energy = {energy:.15f}")

# ---------------------------------------------------------------------------
# 1-/2-RDM extrahieren, Energie aus RDMs als Sanity-Check
# ---------------------------------------------------------------------------
rdm_1 = driver.get_1pdm(ket)
rdm_2 = driver.get_2pdm(ket)

# pyblock2 -> "phys" (1212) Ordering, wie es Optimization3D braucht
rdm_2_phys = rdm_2.transpose(0, 1, 3, 2)

one_body_en = np.einsum("ij,ij->", rdm_1, h1)
two_body_en = 0.5 * np.einsum("ijkl,ikjl->", rdm_2_phys, G_act)
rdm_energy = one_body_en + two_body_en + c
print(f"Energy from rdms = {rdm_energy:.15f}")

# ---------------------------------------------------------------------------
# Orbital-Optimierung mit DMRG-RDM (analog zu p450_open_shell).
# Wir verwenden dasselbe opti-Objekt wieder; optimize_orbs ruft intern
# give_initial_orbitals() auf, was den internen Zustand zuruecksetzt.
# ---------------------------------------------------------------------------
new_orbs, converged = opti.refine_orbitals(
    orbitals=core_orbs + active_orbs,
    rdm1=rdm_1,
    rdm2=rdm_2_phys,
    opt_thresh=0.001,
    occ_thresh=0.001,
    maxiter=11,
    orthonormalization_method="cd",
    refine_core=True,
)

# ---------------------------------------------------------------------------
# Orbitale speichern
# ---------------------------------------------------------------------------
os.makedirs(output_orb_path, exist_ok=True)
core_count = 0
active_count = 0
for orb in new_orbs:
    if getattr(orb, "type", "") == "frozen_occ":
        orb.save_to_file(os.path.join(output_orb_path, f"core_{core_count}.fe"))
        core_count += 1
    else:
        orb.save_to_file(os.path.join(output_orb_path, f"active_{active_count}.fe"))
        active_count += 1

print(f"Saved {core_count} core + {active_count} active orbitals to '{output_orb_path}/'")
print(f"Converged: {converged}")

