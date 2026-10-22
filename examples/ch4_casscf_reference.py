"""PySCF CASSCF and FCI reference calculations for the CH4 ``minbas.py`` example."""

from math import sqrt
from time import time
from pyscf import gto, mcscf, scf

BASIS = "6-31g"
ACTIVE_ELECTRONS = 8
ACTIVE_ORBITALS = 8


def make_ch4_geometry(coordinate: float) -> str:
    """Return the tetrahedral CH4 geometry used by ``minbas.py``.

    ``coordinate`` is the magnitude of each hydrogen Cartesian coordinate in
    angstrom, so the carbon-hydrogen distance is ``sqrt(3) * coordinate``.
    """
    return f"""C  0.0          0.0          0.0
H  {coordinate}  {coordinate}  {coordinate}
H  {coordinate} -{coordinate} -{coordinate}
H -{coordinate}  {coordinate} -{coordinate}
H -{coordinate} -{coordinate}  {coordinate}"""


def calculate_casscf_energy(coordinate: float = 1.15) -> float:
    """Calculate the CH4 ground-state energy with CASSCF(8, 8)."""
    geometry = make_ch4_geometry(coordinate)
    molecule = gto.M(
        atom=geometry,
        basis=BASIS,
        unit="angstrom",
        charge=0,
        spin=0,
        verbose=4,
    )

    mean_field = scf.RHF(molecule).run()
    if not mean_field.converged:
        raise RuntimeError("The Hartree-Fock calculation did not converge")

    casscf = mcscf.CASSCF(
        mean_field,
        ncas=ACTIVE_ORBITALS,
        nelecas=ACTIVE_ELECTRONS,
    ).run()
    if not casscf.converged:
        raise RuntimeError("The CASSCF calculation did not converge")

    # Frozen-core FCI: C 1s frozen, 8 electrons in the remaining 16 HF orbitals
    n_active = mean_field.mo_coeff.shape[1] - 1
    fc_fci = mcscf.CASCI(mean_field, ncas=n_active, nelecas=ACTIVE_ELECTRONS).run()
    if not fc_fci.converged:
        raise RuntimeError("The FCI calculation did not converge")
    e_fci = fc_fci.e_tot

    bond_length = sqrt(3.0) * coordinate
    print(f"CH bond length: {bond_length:.8f} a")
    print(f"RHF energy:     {mean_field.e_tot:.12f} Hartree")
    print(f"CASSCF energy:  {casscf.e_tot:.12f} Hartree")
    print(f"FC-FCI energy (1 frozen, {n_active} active): {e_fci:.12f} Hartree")
    return mean_field.e_tot, casscf.e_tot, e_fci

print("basis:", BASIS)
t0 = time()
energies = []
fci_energies = []
for coord in [0.2 + 0.05 * i for i in range(20)]:
    e_hf, _, e_fci = calculate_casscf_energy(coord)
    energies.append(e_hf)
    fci_energies.append(e_fci)
t1 = time()
print(f"Total time for all bond lengths: {t1 - t0:.2f} seconds")
print("Energies:", energies)
print("FCI energies:", fci_energies)
