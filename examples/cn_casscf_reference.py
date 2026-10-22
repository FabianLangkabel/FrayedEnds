"""PySCF frozen-core open-shell CASSCF reference calculation for CN."""

from time import time

from pyscf import gto, mcscf, scf

BASIS = "cc-pvqz"
N_FROZEN_CORE_ORBITALS = 2
ACTIVE_ELECTRONS = (5, 4)
ACTIVE_ORBITALS = 8


def make_cn_geometry(bond_length: float) -> str:
    """Return a CN geometry with ``bond_length`` in angstrom."""
    return f"""C  0.0  0.0  0.0
N  0.0  0.0  {bond_length}"""


def calculate_casscf_energy(bond_length: float = 1.1718) -> float:
    """Calculate the CN doublet energy with frozen-core CASSCF(9, 8)."""
    molecule = gto.M(
        atom=make_cn_geometry(bond_length),
        basis=BASIS,
        unit="angstrom",
        charge=0,
        spin=1,
        verbose=4,
    )

    mean_field = scf.ROHF(molecule).run()
    if not mean_field.converged:
        initial_density = mean_field.make_rdm1()
        mean_field = mean_field.newton()
        mean_field.kernel(dm0=initial_density)
    if not mean_field.converged:
        raise RuntimeError("The Hartree-Fock calculation did not converge")

    # Keep the first two ROHF orbitals doubly occupied and exclude them from
    # all CASSCF orbital rotations, so their coefficients remain unchanged.
    casscf = mcscf.CASSCF(
        mean_field,
        ncas=ACTIVE_ORBITALS,
        nelecas=ACTIVE_ELECTRONS,
        ncore=N_FROZEN_CORE_ORBITALS,
    ).run()
    
    print(f"CN bond length: {bond_length:.8f} a")
    print(f"ROHF energy:     {mean_field.e_tot:.12f} Hartree")
    print(f"CASSCF energy:   {casscf.e_tot:.12f} Hartree")
    return casscf.e_tot


def main() -> None:
    print("basis:", BASIS)
    t0 = time()
    energies = []
    for distance in [0.9, 1.0, 1.08, 1.1, 1.12, 1.14, 1.16, 1.18, 1.2, 1.22, 1.24, 1.3, 1.4, 1.55, 1.75, 2.0, 2.3, 2.6, 3.0, 3.5, 4.0, 5.0]:
        energies.append(calculate_casscf_energy(distance))
    t1 = time()
    print(f"Total time for all bond lengths: {t1 - t0:.2f} seconds")
    print("Energies:", energies)


if __name__ == "__main__":
    main()
