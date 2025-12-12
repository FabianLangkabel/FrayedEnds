distance = 1.0
iteration_energies = []
iterations = 6
molecule_name = "li3"
box_size = 50.0
wavelet_order = 7
madness_thresh = 0.0001
basisset = 'sto-3g'

# Run NWChem calculation
import subprocess as sp
nwchem_input = '''
title "molecule"
memory stack 3000 mb heap 200 mb global 2800 mb
charge 0  
geometry noautoz noautosym nocenter
  Li  0.0 0.0 0.0
  Li 0.0 0.0 ''' + distance.__str__() + '''
  Li 0.0 0.0 ''' + (-distance).__str__() + '''
end
basis  
  * library ''' + basisset + '''
end
scf  
 doublet; uhf
 maxiter 200
end   
task scf  
'''

with open("nwchem", "w") as f:
    f.write(nwchem_input)

programm = sp.call("/opt/conda/bin/nwchem nwchem", stdout=open('nwchem.out', 'w'), stderr=open('nwchem_err.log', 'w'), shell = True)

#Initalize world
import madpy as mad
world = mad.MadWorld3D(L=box_size, k=wavelet_order, thresh=madness_thresh)

# Convert NWChem AOs and MOs to MRA-Orbitals
converter = mad.NWChem_Converter_open_shell(world)
converter.read_nwchem_file("nwchem")
aos = converter.get_normalized_aos()
alpha_mos, beta_mos = converter.get_mos()
Vnuc = converter.get_Vnuc()
nuclear_repulsion_energy = converter.get_nuclear_repulsion_energy()
del converter


# Visualization of initial orbitals
'''
import py3Dmol
def orbital_view(molecule, orbital, label):

    world.cube_plot("orbital", orbital, molecule.impl, 1.0, 81, [0.0, 0.0, 0.0])
    orbdata = open("orbital.cube", "r").read()
    v = py3Dmol.view(width=300, height=300)  # kleine Breite für nebeneinander
    v.addVolumetricData(orbdata, "cube", {'isoval': -0.001, 'color': "red", 'opacity': 0.75})
    v.addVolumetricData(orbdata, "cube", {'isoval': 0.001, 'color': "blue", 'opacity': 0.75})
    v.addModel(orbdata, "cube")
    v.setStyle({'sphere':{}})
    v.zoomTo()
    html = f"""
    <div style="display:inline-block; text-align:center; margin:5px;">
        <div style="font-weight:bold; margin-bottom:4px;">{label}</div>
        {v._make_html()}
    </div>
    """
    return html

molecule = mad.MadMolecule()
molecule.add_atom(0, 0, 0, "H")
molecule.add_atom(0, 0, distance, "H")
molecule.add_atom(0, 0, -distance, "H")

alpha_0_plot = orbital_view(molecule, alpha_mos[0], "alpha 0")
beta_0_plot = orbital_view(molecule, beta_mos[0], "beta 0")
alpha_1_plot = orbital_view(molecule, alpha_mos[1], "alpha 1")

from IPython.display import display, HTML
display(HTML(f"<div style='display:flex; gap:10px'>{alpha_0_plot}{beta_0_plot}{alpha_1_plot}</div>"))
'''

# Calculate Integrals
core_alpha_orbitals = []
core_beta_orbitals = []
active_alpha_orbitals = []
active_beta_orbitals = []

for i in range(len(alpha_mos)):
    if i < 3:
        core_alpha_orbitals.append(alpha_mos[i])
        core_beta_orbitals.append(beta_mos[i])
    elif i >=3 and i < 6:
        active_alpha_orbitals.append(alpha_mos[i])
        active_beta_orbitals.append(beta_mos[i])

import numpy as np

integrals = mad.Integrals_open_shell_3D(world)
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
rdm_energy = one_body_en + two_body_en + nuclear_repulsion_energy
print('Energy from rdms = %20.15f' % rdm_energy)

opti = mad.Optimization_open_shell_3D(world, Vnuc, nuclear_repulsion_energy)
opti.optimize_orbs(orbitals=[core_alpha_orbitals, core_beta_orbitals, active_alpha_orbitals, active_beta_orbitals], rdm1=rdm_1, rdm2=[rdm_2_phys_aa, rdm_2_phys_ab, rdm_2_phys_bb], opt_thresh=0.001, occ_thresh=0.001, maxiter=1)