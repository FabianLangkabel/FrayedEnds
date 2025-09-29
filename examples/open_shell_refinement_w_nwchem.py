distance = 1.0
iteration_energies = []
iterations = 6
molecule_name = "h3"
box_size = 50.0
wavelet_order = 7
madness_thresh = 0.0001
basisset = 'sto-3g'

# Run NWChem calculation
import subprocess as sp
nwchem_input = '''
title "molecule"
memory stack 1500 mb heap 100 mb global 1400 mb
charge 0  
geometry noautosym nocenter
  H 0.0 0.0 0.0
  H 0.0 0.0 ''' + distance.__str__() + '''
  H 0.0 0.0 ''' + (-distance).__str__() + '''
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

'''
# Visualization of initial orbitals
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
integrals = mad.Integrals_open_shell_3D(world)
T_aa, T_bb = integrals.compute_kinetic_integrals(alpha_mos, beta_mos)
V_aa, V_bb = integrals.compute_potential_integrals(alpha_mos, beta_mos, Vnuc)
G_aa, G_ab, G_bb = integrals.compute_two_body_integrals(alpha_mos, beta_mos)

G_aa = G_aa.transpose(0,2,1,3)
G_ab = G_ab.transpose(0,2,1,3)
G_bb = G_bb.transpose(0,2,1,3)

#Run DMRG
from pyblock2._pyscf.ao2mo import integrals as itg
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SZ, n_threads=4)
driver.initialize_system(n_sites=3, n_elec=3, spin=1, orb_sym=[0, 0, 0])

bond_dims = [250] * 4 + [500]
noises = [1e-4] * 4 + [1e-5] * 4 + [0]
thrds = [1e-10] * 8

mpo = driver.get_qc_mpo(h1e=[T_aa + V_aa, T_bb + V_bb], g2e=[G_aa, G_ab, G_bb], ecore=nuclear_repulsion_energy, iprint=1)
ket = driver.get_random_mps(tag="GS", bond_dim=50, nroots=1)
energy = driver.dmrg(mpo, ket, n_sweeps=20, bond_dims=bond_dims, noises=noises,
    thrds=thrds, iprint=1)
print('DMRG energy = %20.15f' % energy)



# Rdm
# aa, ab, bb
# ijlk
# Wenn eigentlich ab|ab -> ab|ba
# alpha 1 * beta 1 | alpha 2 * beta 2

'''
# Calculate rdms
from pyblock2._pyscf.ao2mo import integrals as itg
from pyblock2.driver.core import DMRGDriver, SymmetryTypes




# Refine orbitals
for i in range(len(alpha_mos)):
    alpha_mos[i].type="active"
for i in range(len(beta_mos)):
  beta_mos[i].type="active"

opti = mad.Optimization_open_shell_3D(world, Vnuc, nuclear_repulsion_energy)
#orbs = opti.get_orbitals(orbitals=[alpha_mos, beta_mos], rdm1=[], rdm2=[], opt_thresh=0.001, occ_thresh=0.001)
opti.optimize_orbs(orbitals=[alpha_mos, beta_mos], rdm1=[], rdm2=[], opt_thresh=0.001, occ_thresh=0.001)
'''