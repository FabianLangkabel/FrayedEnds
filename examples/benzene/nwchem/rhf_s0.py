import os
os.environ["MAD_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"


box_size = 50.0
wavelet_order = 7
madness_thresh = 0.0001
basisset = '6-311++G(2d,2p)'

# Run NWChem calculation
import subprocess as sp
nwchem_input = '''
title "molecule"
memory stack 3000 mb heap 200 mb global 2800 mb
charge 0  
geometry units angstroms noautoz nocenter
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
end
basis spherical
  * library ''' + basisset + '''
end
scf
 maxiter 200
 direct
 vectors atomic
 thresh 1.0e-6
end
task scf

property
 moldenfile molden_norm janpa
end
task scf property
'''

with open("nwchem", "w") as f:
    f.write(nwchem_input)

programm = sp.call("/opt/conda/bin/nwchem nwchem", stdout=open('nwchem.out', 'w'), stderr=open('nwchem_err.log', 'w'), shell = True)

# Alle Orbitale plotten (2D-Schnitt z=+1 Ang ueber Molekuelebene)
from plot_mos import plot_all_mos, find_molden_file
molden_path = find_molden_file(".")
if molden_path:
    print(f"\nPlotte Orbitale aus {molden_path} ...")
    plot_all_mos(molden_path, outdir="mo_plots_rhf_s0", z_above=1.0)
else:
    print("\nWarnung: Keine .molden Datei gefunden -- Plot uebersprungen.")

'''
#Initalize world
import frayedends as fe
world = fe.MadWorld3D(L=box_size, k=wavelet_order, thresh=madness_thresh)

# Convert NWChem AOs and MOs to MRA-Orbitals
converter = fe.NWChem_Converter_open_shell(world)
converter.read_nwchem_file("nwchem")
aos = converter.get_normalized_aos()
alpha_mos, beta_mos = converter.get_mos()
Vnuc = converter.get_Vnuc()
nuclear_repulsion_energy = converter.get_nuclear_repulsion_energy()
del converter


# Calculate Integrals
core_alpha_orbitals = []
core_beta_orbitals = []
active_alpha_orbitals = []
active_beta_orbitals = []

for i in range(len(alpha_mos)):
    if i < 3:
        core_alpha_orbitals.append(alpha_mos[i])
        core_beta_orbitals.append(beta_mos[i])
    elif i >=3:
        active_alpha_orbitals.append(alpha_mos[i])
        active_beta_orbitals.append(beta_mos[i])

import numpy as np

integrals = fe.Integrals_open_shell_3D(world)
c, h1, g2 = integrals.compute_effective_hamiltonian(core_alpha_orbitals, core_beta_orbitals, active_alpha_orbitals, active_beta_orbitals, Vnuc, nuclear_repulsion_energy)
g2[0] = g2[0].transpose(0,2,1,3)
g2[1] = g2[1].transpose(0,2,1,3)
g2[2] = g2[2].transpose(0,2,1,3)


from pyblock2._pyscf.ao2mo import integrals as itg
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SZ, n_threads=4)
driver.initialize_system(n_sites=len(active_alpha_orbitals), n_elec=3, spin=1)
bond_dims = [250] * 4 + [500]
noises = [1e-4] * 4 + [1e-5] * 4 + [0]
thrds = [1e-10] * 8
mpo = driver.get_qc_mpo(h1e=h1, g2e=g2, ecore=c, iprint=1)
ket = driver.get_random_mps(tag="GS", bond_dim=50, nroots=1)
energy = driver.dmrg(mpo, ket, n_sweeps=20, bond_dims=bond_dims, noises=noises,
    thrds=thrds, iprint=1)
print('DMRG energy = %20.15f' % energy)


# Extract rdms
rdm_1 = driver.get_1pdm(ket)
rdm_2 = driver.get_2pdm(ket) 

rdm_2_phys_aa = rdm_2[0].transpose(0, 1, 3, 2)
rdm_2_phys_ab = rdm_2[1].transpose(0, 1, 3, 2)
rdm_2_phys_bb = rdm_2[2].transpose(0, 1, 3, 2)


one_body_en = np.einsum('ij,ij->', rdm_1[0], h1[0]) + np.einsum('ij,ij->', rdm_1[1], h1[1])
two_body_en = 0.5 * (np.einsum('ijkl,ikjl->', rdm_2_phys_aa, g2[0]) 
                     + 2 * np.einsum('ijkl,ikjl->', rdm_2_phys_ab, g2[1]) 
                     + np.einsum('ijkl,ikjl->', rdm_2_phys_bb, g2[2]))
rdm_energy = one_body_en + two_body_en + c
print('Energy from rdms = %20.15f' % rdm_energy)


opti = fe.Optimization_open_shell_3D(world, Vnuc, nuclear_repulsion_energy)
new_core_orbs, new_as_orbs, converged = opti.optimize_orbs(orbitals=[core_alpha_orbitals, core_beta_orbitals, active_alpha_orbitals, active_beta_orbitals], rdm1=rdm_1, rdm2=[rdm_2_phys_aa, rdm_2_phys_ab, rdm_2_phys_bb], opt_thresh=0.001, occ_thresh=0.001, maxiter=1, orthonormalization_method="cd", refine_core=True)


# Update Integrals with new orbitals
c, h1, g2 = opti.get_effective_hamiltonian()
g2[0] = g2[0].transpose(0,2,1,3)
g2[1] = g2[1].transpose(0,2,1,3)
g2[2] = g2[2].transpose(0,2,1,3)

driver.initialize_system(n_sites=len(new_as_orbs[0]), n_elec=3, spin=1)
bond_dims = [250] * 4 + [500]
noises = [1e-4] * 4 + [1e-5] * 4 + [0]
thrds = [1e-10] * 8
mpo = driver.get_qc_mpo(h1e=h1, g2e=g2, ecore=c, iprint=1)
ket = driver.get_random_mps(tag="GS", bond_dim=50, nroots=1)
energy = driver.dmrg(mpo, ket, n_sweeps=20, bond_dims=bond_dims, noises=noises,
    thrds=thrds, iprint=1)
print('DMRG energy = %20.15f' % energy)
'''