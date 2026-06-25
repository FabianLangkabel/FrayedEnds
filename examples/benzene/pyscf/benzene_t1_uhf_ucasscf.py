"""
UCASSCF(6,6)/6-311++G(2d,2p) fuer Benzol T1 -- UHF-Start, UCASSCF.

Unterschied zu benzene_t1_uhf.py: hier wird mcscf.UCASSCF verwendet, das
alpha- und beta-Orbitalsaetze SEPARAT optimiert. sort_mo erhaelt deshalb
einen Tupel (alpha_indices, beta_indices) -- beide aus AVAS-Diagnose extrahiert.

SPEICHER-HINWEIS: UCASSCF mit 6-311++G(2d,2p) (222 Basisfkt.) braucht viel RAM
fuer den 4-Index AO->MO Tensor (alle Spin-Kombinationen aa, ab, bb). max_memory
ist auf 400 GB gesetzt -- nur auf HPC mit entsprechender Ausstattung sinnvoll.

Referenz: Karadakov 2008, Tab. 1
  CASSCF(6,6) T1:              -230.696661   (ROHF-basiert)
  T1 <- S0 vertikale Anregung: 3.86 eV
"""
import numpy as np
from pyscf import gto, scf, mcscf
from pyscf.mcscf import avas
from pyscf.symm import irrep_id2name

mol = gto.M(
    atom="""
    C    1.39640000   0.00000000   0.00000000
    C    0.69820000   1.20931787   0.00000000
    C   -0.69820000   1.20931787   0.00000000
    C   -1.39640000   0.00000000   0.00000000
    C   -0.69820000  -1.20931787   0.00000000
    C    0.69820000  -1.20931787   0.00000000
    H    2.47950000   0.00000000   0.00000000
    H    1.23975000   2.14730999   0.00000000
    H   -1.23975000   2.14730999   0.00000000
    H   -2.47950000   0.00000000   0.00000000
    H   -1.23975000  -2.14730999   0.00000000
    H    1.23975000  -2.14730999   0.00000000
    """,
    basis="6-311++G(2d,2p)",
    max_memory=400000,               # 400 GB (HPC)
    spin=2,                          # Triplett: 2S = nalpha - nbeta = 2
    symmetry=True,
    verbose=4,
)

mf = scf.UHF(mol).run(conv_tol=1e-10)
s2_uhf, _ = mf.spin_square()
print(f"\n<S^2> UHF = {s2_uhf:.4f}  (rein: 2.0000; >2 -> Spin-Kontamination)")

ncas, nelecas, mo_avas = avas.avas(
    mf, aolabels=['C 2pz'], minao='ano', canonicalize=True, openshell_option=2,
)

# AVAS-Analyse: dominante UHF-MO-Indizes pro Slot fuer ALPHA und BETA.
# Beide gehen in sort_mo ein, weil UCASSCF separate Spinsaetze optimiert.
S = mol.intor("int1e_ovlp")
ncore = (mol.nelectron - np.sum(nelecas)) // 2

mo_arr = np.asarray(mf.mo_coeff)
e_arr = np.asarray(mf.mo_energy)
o_arr = np.asarray(mf.mo_occ)
sym_arr = np.asarray(mf.orbsym)
avas_arr = np.asarray(mo_avas)


def dominant_indices(coeffs, avas_block):
    overlap = coeffs.T @ S @ avas_block
    picked = []
    for j in range(overlap.shape[1]):
        for idx in np.argsort(-np.abs(overlap[:, j])):
            if int(idx) not in picked:
                picked.append(int(idx))
                break
    return sorted(picked)


indices_per_spin = {}
for label, spin in (('alpha', 0), ('beta', 1)):
    coef = mo_arr[spin] if mo_arr.ndim == 3 else mo_arr
    ener = e_arr[spin] if e_arr.ndim == 2 else e_arr
    occ = o_arr[spin] if o_arr.ndim == 2 else o_arr
    symm = sym_arr[spin] if sym_arr.ndim == 2 else sym_arr
    avas_spin = avas_arr[spin] if avas_arr.ndim == 3 else avas_arr

    idxs = dominant_indices(coef, avas_spin[:, ncore:ncore + ncas])
    indices_per_spin[label] = idxs

    print(f"\nGewaehlte UHF-{label}-MO-Indizes fuer Aktivraum (dominanteste pro AVAS-Slot):")
    for idx in idxs:
        sym = irrep_id2name(mol.groupname, symm[idx])
        print(f"  MO{idx:>3}  E={ener[idx]:+.5f}  occ={occ[idx]:.1f}  {sym}")

# UCASSCF: separate Alpha/Beta-Aktivraeume
mc = mcscf.UCASSCF(mf, ncas, nelecas)
mc.max_memory = 400000           # 400 GB (HPC)
mc.natorb = True
caslst = (indices_per_spin['alpha'], indices_per_spin['beta'])
e_cas = mc.kernel(mc.sort_mo(caslst, base=0))[0]

# Spin-aufgeloeste 1-RDMs -> Alpha- und Beta-NOON
dm1a, dm1b = mc.fcisolver.make_rdm1s(mc.ci, mc.ncas, mc.nelecas)
nat_occ_a = np.linalg.eigvalsh(dm1a)[::-1]
nat_occ_b = np.linalg.eigvalsh(dm1b)[::-1]
s2, _ = mc.fcisolver.spin_square(mc.ci, mc.ncas, mc.nelecas)

print(f"\nUHF              = {mf.e_tot:.6f}")
print(f"<S^2> UHF        = {s2_uhf:.4f}")
print(f"UCASSCF(6,6) T1  = {e_cas:.6f}   (Paper ROHF-CASSCF: -230.696661)")
print(f"NOON alpha       = {np.array2string(nat_occ_a, precision=4)}")
print(f"NOON beta        = {np.array2string(nat_occ_b, precision=4)}")
print(f"<S^2> UCASSCF    = {s2:.4f}   (Triplett: 2.0000)")
