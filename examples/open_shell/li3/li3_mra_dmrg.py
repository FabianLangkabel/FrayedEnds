import time

import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

import frayedends as fe

molecule_name = "li3"
n_elec = 3
n_core = 3
iterations = 15
box_size = 50.0
wavelet_order = 7
madness_thresh = 1.0e-6
econv = 1.0e-4 

basisset = "sto-3g"
nwchem_command = "nwchem" 

# DRMG parameters
bond_dims = [250] * 4 + [500]
noises = [1e-4] * 4 + [1e-5] * 4 + [0]
thrds = [1e-10] * 8

# Create .dat file for the results
output_file = f"{molecule_name}_nwchem_dmrg_oo.dat"
 
with open(output_file, "w") as f:
    f.write("iteration iteration_time_s energy energy_rdm\n")
 
total_start = time.perf_counter()

# Define the linear Li3 geometry
geom = f"Li 0.0 0.0 1.5489\nLi 0.0 1.6710 -0.7745\nLi 0.0 -1.6710 -0.7745\n"

# Run NWChem calculation (doublet, UHF)
nwchem = fe.run_nwchem(geometry=geom, basis=basisset, name=molecule_name, units="angstrom", multiplicity=2, nwchem_command=nwchem_command)

# Initalize world
world = fe.MadWorld(ndims=3, L=box_size, k=wavelet_order, thresh=madness_thresh)

# Convert NWChem MOs to MRA-Orbitals
converter = fe.NWChem_Converter_open_shell(world)
converter.read_nwchem_file(nwchem)
alpha_mos, beta_mos = converter.get_mos()
Vnuc = converter.get_Vnuc()
nuclear_repulsion_energy = converter.get_nuclear_repulsion_energy()

# Calculate Integrals
core_alpha_orbitals = alpha_mos[:n_core]
core_beta_orbitals = beta_mos[:n_core]

active_alpha_orbitals = alpha_mos[n_core:]
active_beta_orbitals = beta_mos[n_core:]

n_active = len(active_alpha_orbitals)

integrals = fe.Integrals_open_shell(world)
c, h1, g2 = integrals.compute_effective_hamiltonian(
    core_alpha_orbitals, core_beta_orbitals, active_alpha_orbitals, active_beta_orbitals, Vnuc, nuclear_repulsion_energy
)
g2[0] = g2[0].transpose(0, 2, 1, 3) # aa
g2[1] = g2[1].transpose(0, 2, 1, 3) # ab
g2[2] = g2[2].transpose(0, 2, 1, 3) # bb

driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SZ, n_threads=8)
driver.initialize_system(n_sites=n_active, n_elec=n_elec, spin=1)

mpo = driver.get_qc_mpo(h1e=h1, g2e=g2, ecore=c, iprint=0)
ket = driver.get_random_mps(tag="GS", bond_dim=50, nroots=1)
energy = driver.dmrg(mpo, ket, n_sweeps=20, bond_dims=bond_dims, noises=noises, thrds=thrds, iprint=0)

# Extract rdms
rdm_1 = driver.get_1pdm(ket)
rdm_2 = driver.get_2pdm(ket)

rdm_2_phys_aa = rdm_2[0].transpose(0, 1, 3, 2)
rdm_2_phys_ab = rdm_2[1].transpose(0, 1, 3, 2)
rdm_2_phys_bb = rdm_2[2].transpose(0, 1, 3, 2)

one_body_en = np.einsum("ij,ij->", rdm_1[0], h1[0]) + np.einsum("ij,ij->", rdm_1[1], h1[1])
two_body_en = 0.5 * (
    np.einsum("ijkl,ikjl->", rdm_2_phys_aa, g2[0])
    + 2 * np.einsum("ijkl,ikjl->", rdm_2_phys_ab, g2[1])
    + np.einsum("ijkl,ikjl->", rdm_2_phys_bb, g2[2])
)
rdm_energy = one_body_en + two_body_en + c

with open(output_file, "a") as f:
    f.write(f"0 0.00 {energy:.15f} {rdm_energy:.15f}\n")

print(f"Orbitals: {n_core} core + {n_active} active (per spin), {n_elec} active electrons")
print(f"{'Iter':<6}{'Energy':<18}{'E_rdm':<18}dE")
print(f"{0:<6}{energy:<18.10f}{rdm_energy:<18.10f}-")

current = energy
converged = False

for iter in range(iterations):
    iter_start = time.perf_counter()

    # Orbital Refinement (core and active orbitals)
    opti = fe.OrbitalRefinement_open_shell(world, Vnuc, nuclear_repulsion_energy)
    new_core_orbs, new_as_orbs, converged_flag = opti.refine_orbitals(
        orbitals=[core_alpha_orbitals, core_beta_orbitals, active_alpha_orbitals, active_beta_orbitals],
        rdm1=rdm_1,
        rdm2=[rdm_2_phys_aa, rdm_2_phys_ab, rdm_2_phys_bb],
        opt_thresh=0.001,
        occ_thresh=0.001,
        maxiter=1,
        orthonormalization_method="cd",
        refine_core=True,
    )

    core_alpha_orbitals, core_beta_orbitals = new_core_orbs
    active_alpha_orbitals, active_beta_orbitals = new_as_orbs

    # Update Integrals with new orbitals
    c, h1, g2 = opti.get_effective_hamiltonian()
    g2[0] = g2[0].transpose(0, 2, 1, 3) # aa
    g2[1] = g2[1].transpose(0, 2, 1, 3) # ab
    g2[2] = g2[2].transpose(0, 2, 1, 3) # bb

    driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SZ, n_threads=8)
    driver.initialize_system(n_sites=n_active, n_elec=n_elec, spin=1)

    mpo = driver.get_qc_mpo(h1e=h1, g2e=g2, ecore=c, iprint=0)
    ket = driver.get_random_mps(tag="GS", bond_dim=50, nroots=1)
    energy = driver.dmrg(mpo, ket, n_sweeps=20, bond_dims=bond_dims, noises=noises, thrds=thrds, iprint=0)

    # Extract rdms
    rdm_1 = driver.get_1pdm(ket)
    rdm_2 = driver.get_2pdm(ket)

    rdm_2_phys_aa = rdm_2[0].transpose(0, 1, 3, 2)
    rdm_2_phys_ab = rdm_2[1].transpose(0, 1, 3, 2)
    rdm_2_phys_bb = rdm_2[2].transpose(0, 1, 3, 2)

    one_body_en = np.einsum("ij,ij->", rdm_1[0], h1[0]) + np.einsum("ij,ij->", rdm_1[1], h1[1])
    two_body_en = 0.5 * (
        np.einsum("ijkl,ikjl->", rdm_2_phys_aa, g2[0])
        + 2 * np.einsum("ijkl,ikjl->", rdm_2_phys_ab, g2[1])
        + np.einsum("ijkl,ikjl->", rdm_2_phys_bb, g2[2])
    )
    rdm_energy = one_body_en + two_body_en + c

    print(f"{iter + 1:<6}{energy:<18.10f}{rdm_energy:<18.10f}{energy - current:+.1e}")
 
    iter_time = time.perf_counter() - iter_start
    with open(output_file, "a") as f:
        f.write(f"{iter + 1} {iter_time:.2f} {energy:.15f} {rdm_energy:.15f}\n")
 
    if abs(energy - current) < econv:
        converged = True
        print(f"Converged after {iter + 1} iterations (|dE| < {econv:.0e})")
        break
    current = energy
 
if not converged:
    print(f"Not converged after {iterations} iterations")
 
with open(output_file, "a") as f:
    status = "converged" if converged else "not converged"
    f.write(f"# {status}, econv = {econv:.0e}, total time = {time.perf_counter() - total_start:.2f} s\n")
 
fe.cleanup(globals())