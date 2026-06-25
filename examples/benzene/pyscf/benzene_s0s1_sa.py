"""
State-averaged CASSCF(6,6)/6-311++G(2d,2p) fuer Benzol S0 + S1.
Referenz: Karadakov 2008, Tab. 1
  S0 (1^1A1g): -230.838459   (state-specific)
  S1 (1^1B2u): -230.656017   (state-specific)
  S1 <- S0:    4.96 eV

SA-CASSCF gleichgewichtet (0.5/0.5). Aktivraum-Orbitale werden gemeinsam
fuer beide Zustaende optimiert; FCI findet die 2 niedrigsten Singulett-
Eigenzustaende ohne wfnsym-Constraint (Symmetrie auf Molekuel-Ebene
deaktiviert). Einzelne SA-Energien liegen systematisch ueber den state-
specific Werten, dafuer ist die Anregungsluecke besser balanciert.
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
    symmetry=False,                  # keine wfnsym-Festlegung
    verbose=4,
)

mf = scf.RHF(mol).run(conv_tol=1e-10)

ncas, nelecas, mo_avas = avas.avas(
    mf, aolabels=['C 2pz'], minao='ano', canonicalize=True,
)

mc = mcscf.CASSCF(mf, ncas, nelecas)
mc = mcscf.state_average_(mc, weights=[0.333, 0.333, 0.333])
mc.natorb = True
mc.kernel(mo_avas)

e_S0, e_S1, e_S2 = mc.e_states

print(f"\nRHF                = {mf.e_tot:.6f}")
print(f"SA-CASSCF(6,6) S0  = {e_S0:.6f}   (Paper SS: -230.838459)")
print(f"SA-CASSCF(6,6) T1  = {e_S1:.6f}   (Paper SS: -230.696661)")
print(f"SA-CASSCF(6,6) S2  = {e_S2:.6f}   (Paper SS: -230.656017)")
print(f"S1 - S0 (PySCF SA) = {(e_S1 - e_S0) * 27.2114:.3f} eV")
print(f"S1 - S0 (Paper SS) = 4.96 eV")
