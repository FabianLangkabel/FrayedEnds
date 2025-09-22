distance = 2.5
iteration_energies = []
iterations = 6
molecule_name = "beh2"
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
  Be 0.0 0.0 0.0
  H 0.0 0.0 ''' + distance.__str__() + '''
  H 0.0 0.0 ''' + (-distance).__str__() + '''
end
basis  
  * library ''' + basisset + '''
end
scf  
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
converter = mad.NWChem_Converter(world)
converter.read_nwchem_file("nwchem")
aos = converter.get_normalized_aos()
mos = converter.get_mos()
del converter

# Visualization of orbitals
molecule = mad.MadMolecule()
molecule.add_atom(0, 0, 0, "Be")
molecule.add_atom(0, 0, distance, "H")
molecule.add_atom(0, 0, -distance, "H")
world.cube_plot("orbital", mos[5], molecule.impl, 1.0, 81, [0.0, 0.0, 0.0])

import py3Dmol
orbdata = open("orbital.cube", "r").read()
v = py3Dmol.view()
v.addVolumetricData(orbdata, "cube", {'isoval': -0.001, 'color': "red", 'opacity': 0.75})
v.addVolumetricData(orbdata, "cube", {'isoval': 0.001, 'color': "blue", 'opacity': 0.75})
v.addModel(orbdata, "cube")
v.setStyle({'sphere':{}})
v.zoomTo()
v.show()