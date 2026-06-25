"""
CASSCF(6,6)/6-311++G(2d,2p) fuer Benzol T1 (1^3B1u, D6h) -- UHF-Start.
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
    spin=2,                          # Triplett: 2S = nalpha - nbeta = 2
    symmetry=True,
    verbose=4,
)

mf = scf.UHF(mol).run(conv_tol=1e-10)
s2_uhf, mult_uhf = mf.spin_square()
print(f"\n<S^2> UHF = {s2_uhf:.4f}  (rein: 2.0000; >2 -> Spin-Kontamination)")

ncas, nelecas, mo_avas = avas.avas(
    mf, aolabels=['C 2pz'], minao='ano', canonicalize=True, openshell_option=2,
)

# AVAS-Analyse: dominante alpha-UHF-MO-Indizes pro Aktivraum-Slot identifizieren,
# Aktivraum aber aus unvermischten UHF-Orbitalen aufbauen (via sort_mo).
# mcscf.CASSCF auf UHF nutzt intern nur den alpha-Satz.
S = mol.intor("int1e_ovlp")
ncore = (mol.nelectron - np.sum(nelecas)) // 2

mo_arr = np.asarray(mf.mo_coeff)
e_arr = np.asarray(mf.mo_energy)
o_arr = np.asarray(mf.mo_occ)
sym_arr = np.asarray(mf.orbsym)
avas_arr = np.asarray(mo_avas)

hf_coeffs = mo_arr[0] if mo_arr.ndim == 3 else mo_arr
hf_energy = e_arr[0] if e_arr.ndim == 2 else e_arr
hf_occ = o_arr[0] if o_arr.ndim == 2 else o_arr
orbsym_a = sym_arr[0] if sym_arr.ndim == 2 else sym_arr
mo_avas_a = avas_arr[0] if avas_arr.ndim == 3 else avas_arr

overlap = hf_coeffs.T @ S @ mo_avas_a[:, ncore:ncore + ncas]

hf_indices = []
for j in range(ncas):
    abs_ov = np.abs(overlap[:, j])
    for idx in np.argsort(-abs_ov):
        if int(idx) not in hf_indices:
            hf_indices.append(int(idx))
            break

hf_indices.sort()
print("\nGewaehlte UHF-alpha-MO-Indizes fuer Aktivraum (dominanteste pro AVAS-Slot):")
for idx in hf_indices:
    sym = irrep_id2name(mol.groupname, orbsym_a[idx])
    print(f"  MO{idx:>3}  E={hf_energy[idx]:+.5f}  occ={hf_occ[idx]:.1f}  {sym}")

mc = mcscf.CASSCF(mf, ncas, nelecas)
mc.fcisolver.wfnsym = 'B3u'
mc.natorb = True
e_cas = mc.kernel(mc.sort_mo(hf_indices, base=0))[0]

dm1 = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)
nat_occ = np.linalg.eigvalsh(dm1)[::-1]
s2, _ = mc.fcisolver.spin_square(mc.ci, mc.ncas, mc.nelecas)

print(f"\nUHF            = {mf.e_tot:.6f}")
print(f"<S^2> UHF      = {s2_uhf:.4f}")
print(f"CASSCF(6,6) T1 = {e_cas:.6f}   (Paper ROHF: -230.696661)")
print(f"NOON           = {np.array2string(nat_occ, precision=4)}")
print(f"<S^2> CASSCF   = {s2:.4f}   (Triplett: 2.0000)")
