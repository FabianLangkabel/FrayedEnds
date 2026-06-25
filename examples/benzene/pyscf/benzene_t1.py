"""
CASSCF(6,6)/6-311++G(2d,2p) fuer Benzol T1 (1^3B1u, D6h).
Referenz: Karadakov, J. Phys. Chem. A 2008, 112, 7303, Tab. 1.
  CASSCF(6,6) T1:              -230.696661
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
    spin=2,                          # Triplett: 2S = nalpha - nbeta = 2
    symmetry=True,
    verbose=4,
)

mf = scf.ROHF(mol).run(conv_tol=1e-10)

# AVAS waehlt das valente pi-System (6 C-2pz) als Aktivraum; openshell_option=2
# verteilt die SOMOs sauber auf besetzten/virtuellen Teil des Aktivraums.
ncas, nelecas, mo_avas = avas.avas(
    mf, aolabels=['C 2pz'], minao='ano', canonicalize=True, openshell_option=2,
)

# AVAS-Analyse: dominante ROHF-MO-Indizes pro Aktivraum-Slot identifizieren,
# Aktivraum aber aus unvermischten ROHF-Orbitalen aufbauen (via sort_mo).
S = mol.intor("int1e_ovlp")
ncore = (mol.nelectron - np.sum(nelecas)) // 2
overlap = mf.mo_coeff.T @ S @ mo_avas[:, ncore:ncore + ncas]

hf_indices = []
for j in range(ncas):
    abs_ov = np.abs(overlap[:, j])
    for idx in np.argsort(-abs_ov):
        if int(idx) not in hf_indices:
            hf_indices.append(int(idx))
            break

hf_indices.sort()
print("\nGewaehlte ROHF-MO-Indizes fuer Aktivraum (dominanteste pro AVAS-Slot):")
for idx in hf_indices:
    sym = irrep_id2name(mol.groupname, mf.orbsym[idx])
    print(f"  MO{idx:>3}  E={mf.mo_energy[idx]:+.5f}  occ={mf.mo_occ[idx]:.1f}  {sym}")

# Falls PySCF einen anderen Triplett-Zustand findet (NOON-Muster (1.5,1.5,0.5,0.5)
# oder Anregung ~5 eV statt 3.86 eV), D2h-Symmetrie explizit fixieren:
#   mc.fcisolver.wfnsym = 'B2u'   # oder 'B3u'
mc = mcscf.CASSCF(mf, ncas, nelecas)
mc.natorb = True
e_cas = mc.kernel(mc.sort_mo(hf_indices, base=0))[0]

dm1 = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)
nat_occ = np.linalg.eigvalsh(dm1)[::-1]
s2, _ = mc.fcisolver.spin_square(mc.ci, mc.ncas, mc.nelecas)

print(f"\nROHF           = {mf.e_tot:.6f}")
print(f"CASSCF(6,6) T1 = {e_cas:.6f}   (Paper: -230.696661)")
print(f"NOON           = {np.array2string(nat_occ, precision=4)}")
print(f"<S^2>          = {s2:.4f}   (Triplett: 2.0000)")
