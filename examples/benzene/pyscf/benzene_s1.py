"""
CASSCF(6,6)/6-311++G(2d,2p) fuer Benzol S1 (1^1B2u, D6h).
Referenz: Karadakov, J. Phys. Chem. A 2008, 112, 7303, Tab. 1.
  CASSCF(6,6) S1:              -230.656017
  S1 <- S0 vertikale Anregung: 4.96 eV
"""
import numpy as np
from pyscf import gto, scf, mcscf
from pyscf.mcscf import avas

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
)  # spin=0 default (Singulett)

mf = scf.RHF(mol).run(conv_tol=1e-10)

ncas, nelecas, mo_avas = avas.avas(
    mf, aolabels=['C 2pz'], minao='ano', canonicalize=True,
)

# S1 = 1B2u (D6h). Analog zu T1 (3B1u -> B3u in D2h) waehlen wir hier B2u
# als komplementaere D2h-Irrep. wfnsym zwingt den FCI-Solver auf den
# niedrigsten Singulett dieser Symmetrie (nicht S0 = Ag).
# Falls Energie / NOON unplausibel: 'B3u' testen.
mc = mcscf.CASSCF(mf, ncas, nelecas)
mc.fcisolver.wfnsym = 'B2u'
mc.natorb = True
e_cas = mc.kernel(mo_avas)[0]

dm1 = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)
nat_occ = np.linalg.eigvalsh(dm1)[::-1]
s2, _ = mc.fcisolver.spin_square(mc.ci, mc.ncas, mc.nelecas)

print(f"\nRHF            = {mf.e_tot:.6f}")
print(f"CASSCF(6,6) S1 = {e_cas:.6f}   (Paper: -230.656017)")
print(f"NOON           = {np.array2string(nat_occ, precision=4)}")
print(f"<S^2>          = {s2:.4f}   (Singulett: 0.0000)")
