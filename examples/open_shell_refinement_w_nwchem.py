distance = 2.5
iteration_energies = []
iterations = 6
molecule_name = "h3"
box_size = 50.0
wavelet_order = 7
madness_thresh = 0.0001
basisset = 'cc-pvdz'

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
del converter






# Visualization of orbitals
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