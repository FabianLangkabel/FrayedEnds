"""
CASSCF(6,6)/6-311++G(2d,2p) fuer Benzol S0 (D6h).
Referenz: Karadakov, J. Phys. Chem. A 2008, 112, 7303, Tab. 1.
  RHF:         -230.766572
  CASSCF(6,6): -230.838459
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
    symmetry=True,
    verbose=4,
)

mf = scf.RHF(mol).run(conv_tol=1e-10)

# AVAS waehlt das valente pi-System (6 C-2pz) als Aktivraum; filtert
# diffuse Rydberg-pz dank Projektion auf minimale valente AOs raus.
ncas, nelecas, mo_avas = avas.avas(
    mf, aolabels=['C 2pz'], minao='ano', canonicalize=True,
)

# AVAS-Analyse: dominante HF-MO-Indizes pro Aktivraum-Slot identifizieren,
# Aktivraum aber aus unvermischten HF-Orbitalen aufbauen (via sort_mo).
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
print("\nGewaehlte HF-MO-Indizes fuer Aktivraum (dominanteste pro AVAS-Slot):")
for idx in hf_indices:
    sym = irrep_id2name(mol.groupname, mf.orbsym[idx])
    print(f"  MO{idx:>3}  E={mf.mo_energy[idx]:+.5f}  occ={mf.mo_occ[idx]:.1f}  {sym}")

mc = mcscf.CASSCF(mf, ncas, nelecas)
mc.natorb = True
e_cas = mc.kernel(mc.sort_mo(hf_indices, base=0))[0]

dm1 = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)
nat_occ = np.linalg.eigvalsh(dm1)[::-1]

print(f"\nRHF         = {mf.e_tot:.6f}   (Paper: -230.766572)")
print(f"CASSCF(6,6) = {e_cas:.6f}   (Paper: -230.838459)")
print(f"NOON        = {np.array2string(nat_occ, precision=4)}")
