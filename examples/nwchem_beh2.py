import subprocess as sp

import py3Dmol

import frayedends as fe

distance = 2.5
iteration_energies = []
iterations = 6
molecule_name = "beh2"
box_size = 50.0
wavelet_order = 7
madness_thresh = 0.0001

basisset = "cc-pvdz"
nwchem_command = "nwchem"
nwchem_localize = None

geom = "Be 0.0 0.0 0.0\nH 0.0 0.0 2.5\nH 0.0 0.0 -2.5\n"

# Run NWChem calculation
nwchem = fe.run_nwchem(
    geometry=geom,
    basis=basisset,
    name=molecule_name,
    units="angstrom",
    nwchem_command=nwchem_command,
    localize=nwchem_localize,
)

# Initalize world
world = fe.MadWorld(ndims=3, L=box_size, k=wavelet_order, thresh=madness_thresh)

# Convert NWChem AOs and MOs to MRA-Orbitals
converter = fe.NWChem_Converter(world)
converter.read_nwchem_file(nwchem)
aos = converter.get_normalized_aos()
mos = converter.get_mos()
del converter

# Visualization of orbitals
molecule = fe.MolecularGeometry()
molecule.add_atom(0, 0, 0, "Be")
molecule.add_atom(0, 0, distance, "H")
molecule.add_atom(0, 0, -distance, "H")
world.cube_plot("orbital", mos[5], molecule, 1.0, 81, [0.0, 0.0, 0.0])

orbdata = open("orbital.cube", "r").read()
v = py3Dmol.view()
v.addVolumetricData(orbdata, "cube", {"isoval": -0.001, "color": "red", "opacity": 0.75})
v.addVolumetricData(orbdata, "cube", {"isoval": 0.001, "color": "blue", "opacity": 0.75})
v.addModel(orbdata, "cube")
v.setStyle({"sphere": {}})
v.zoomTo()
v.show()
