from pyscf import gto, scf, mcscf, lo

def calculate_casscf_energy_and_gradient(geometry, basis='cc-pV5Z', unit='bohr', charge=0, spin=0, nactorb=4, nactelec=2):
    """
    Calculate the ground state energy of a molecule using CASSCF.

    Parameters:
    - geometry: List of tuples, e.g., [('H', (0, 0, 0)), ('H', (0, 0, 0.74))]
    - basis: Basis set (default: 'cc-pVDZ')
    - charge: Molecular charge (default: 0)
    - spin: Spin multiplicity (default: 0)
    - nactorb: Number of active orbitals (default: 2)
    - nactelec: Number of active electrons (default: 2)

    Returns:
    - Ground state energy (float)
    """

    # Define the molecule
    mol = gto.Mole(atom=geometry, unit=unit, basis=basis, charge=charge, spin=spin)
    mol.build()
    print(mol.atom_coords())
    # Perform Hartree-Fock calculation
    mf = scf.RHF(mol)
    mf.kernel()

    #loc_orbs = lo.Boys(mol, mf.mo_coeff).kernel()

    # Perform CASSCF calculation
    mc = mcscf.CASSCF(mf, nactorb, nactelec)
    #mc.conv_tol = 1e-10          # Stricter convergence tolerance (default ~1e-7)
    #mc.max_cycle_macro = 200      # More macro iterations (default 100)
    #mc.max_cycle_micro = 5       # More micro iterations (default 4)
    #mc.max_stepsize = 0.005
    #mc.kernel(mo_coeff=loc_orbs)
    mc.kernel()

    # Calculate the gradient
    mc_grad = mc.nuc_grad_method().kernel()

    return mc.e_tot, mc_grad

distance_list=[0.02+0.02*i for i in range(200)]
energy_list = []
gradient_list = []
for i in range(len(distance_list)):
    # Example usage: H2 molecule
    geometry = f'H 0.0 0.0 {-distance_list[i]/2}\nH 0.0 0.0 {distance_list[i]/2}'
    energy, gradient = calculate_casscf_energy_and_gradient(geometry)

    print(f"Ground state energy (CASSCF): {energy} Hartree")
    print(f"Energy gradient (CASSCF):\n{gradient}")
    energy_list.append(energy)
    gradient_list.append(gradient[1, 2])

print("Distance list =", distance_list)
print("Energy list =", energy_list)
print("Gradient list =", gradient_list)