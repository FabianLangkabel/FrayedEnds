import numpy
import tequila as tq

import frayedends

world = frayedends.MadWorld3D(thresh=1.0e-6, k=9)
method = "fci"
geom = """
H 0.0 0.0 0.0
H 0.0 0.0 1.0
"""
orbitals = "sto-3g"
energy, orbitals, rdm1, rdm2 = frayedends.optimize_basis_3D(
    world=world, many_body_method=method, geometry=geom, econv=1.0e-7, orbitals=orbitals
)
print("2 MRA orbitals: ", energy)

for basis in ["sto-3g", "sto-6g", "6-31G", "cc-pVDZ", "cc-pVTZ", "cc-pVQZ", "cc-pV5Z"]:
    mol = tq.Molecule(geometry=geom, basis_set=basis)
    print(
        "{} orbitals gives: {:2.5f}".format(mol.n_orbitals, mol.compute_energy("fci"))
    )

    from pyscf import gto, mcscf, scf

    mol = gto.Mole()
    mol.atom = [
        ["H", (0.0, 0.0, 0.0)],
        ["H", (0.0, 0.0, 1.0)],  # Bond length of 0.74 Angstroms
    ]
    mol.basis = basis  # Basis set
    mol.spin = 0  # Singlet state
    mol.build()

    # Perform a Hartree-Fock calculation
    mf = scf.RHF(mol)
    mf.kernel()

    # Perform a CAS-SCF calculation
    # Define the active space: 2 electrons in 2 orbitals
    cas = mcscf.CASSCF(mf, 2, 2)
    cas.kernel()

    # Print the CAS-SCF energy
    print("CAS-SCF energy:", cas.e_tot)
