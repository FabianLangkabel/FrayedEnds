import numpy as np
import pyscf
import json
import subprocess as sp
from pyscf.tools import cubegen
from pyblock2._pyscf.ao2mo import integrals as itg
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

'''
Definitions/Parameters
'''                       
distance = 1
iteration_energies = []
iterations = 6
molecule_name = "h2"
box_size = 50.0
wavelet_order = 7
madness_thresh = 0.0001
number_occ_in_as = 1
number_virt_in_as = 3
number_occupied_orbitals = 1
basisset = 'cc-pvdz'

# GEOMETRY IN ANGSTROM
nwchem_input = '''
title "molecule"
memory stack 1500 mb heap 100 mb global 1400 mb
charge 0  
geometry noautosym nocenter
  H 0.0 0.0 ''' + (distance / 2).__str__() + '''
  H 0.0 0.0 ''' + (-distance / 2).__str__() + '''
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

###### Convert nwchem MOs to MRA-orbitals ######
first_occ_as_orbital = number_occupied_orbitals - number_occ_in_as
last_virt_as_orbital = number_occupied_orbitals + number_virt_in_as
active_space = list(range(first_occ_as_orbital, last_virt_as_orbital))
nwchem_to_orbopt_input = json.loads('{}')
nwchem_to_orbopt_input["box_size"] = box_size
nwchem_to_orbopt_input["wavelet_order"] = wavelet_order
nwchem_to_orbopt_input["madness_thresh"] = madness_thresh
nwchem_to_orbopt_input["nwchem_file"] = "nwchem"
nwchem_to_orbopt_input["output_folder"] = "output"
nwchem_to_orbopt_input["number_occupied_orbitals"] = number_occupied_orbitals
nwchem_to_orbopt_input["active_orbitals"] = active_space
nwchem_to_orbopt_input_file = open("nwchem_to_orbopt_input.json", 'w')
json.dump(nwchem_to_orbopt_input, nwchem_to_orbopt_input_file)
nwchem_to_orbopt_input_file.close()

programm = sp.call("/workspaces/MRA-OrbitalOptimization/build/nwchem_to_orbopt/nwchem_to_orbopt nwchem_to_orbopt_input.json", stdout=open('converter_log', 'w'), stderr=open('converter_err_log', 'w'), shell = True)

################### DMRG ###################
threads = 4
ram_gb = 4
bd_pre = 100
bd_reordered = 200

with open("output" + '/c.txt') as f:
    c = float(f.readline())
h1 = np.load("output" + "/htensor.npy")
g2 = np.load("output" + "/gtensor.npy")
g2 = np.swapaxes(g2,1,2) # Integrals are in physics notation, Block2 uses chemistry notation

#### First DMRG calculation
driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SU2, n_threads=threads, stack_mem=int(ram_gb*1024**3), restart_dir="./restart")
driver.initialize_system(n_sites=len(active_space), n_elec=number_occ_in_as * 2, spin=0)
mpo = driver.get_qc_mpo(h1e=h1, g2e=g2, ecore=c, iprint=1)
ket = driver.get_random_mps(tag="GS", bond_dim=250, nroots=1)
energy = driver.dmrg(mpo, ket, n_sweeps=500, bond_dims=[bd_pre], iprint=1)
print('DMRG energy = %20.15f' % energy)