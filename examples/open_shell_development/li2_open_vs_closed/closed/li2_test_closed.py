distance = 1.0
iteration_energies = []
iterations = 6
molecule_name = "li2"
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
  Li 0.0 0.0 ''' + distance.__str__() + '''
  Li 0.0 0.0 ''' + (-distance).__str__() + '''
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
Vnuc = converter.get_Vnuc()
nuclear_repulsion_energy = converter.get_nuclear_repulsion_energy()
del converter

orbs = []
for i in range(2):
    orbs.append(mos[i])
    orbs[i].type="frozen_occ"

orbs.append(mos[2])
orbs.append(mos[3])
orbs.append(mos[4])

orbs[2].type = "active"
orbs[3].type = "active"
orbs[4].type = "active"

import numpy as np

opti = mad.Optimization3D(world, Vnuc, nuclear_repulsion_energy)
orbs = opti.get_orbitals(orbitals=orbs, rdm1=np.zeros((2,2)), rdm2=np.zeros((2,2,2,2)), opt_thresh=0.001, occ_thresh=0.001)

c, h1, g2 = opti.get_integrals()
h1 = np.array(h1)
g2 = np.array(g2)
h1.resize((3,3))
g2.resize((3, 3, 3, 3))
g2 = g2.transpose(0,2,1,3)

from pyblock2._pyscf.ao2mo import integrals as itg
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SU2, n_threads=4)
driver.initialize_system(n_sites=3, n_elec=2, spin=0)
mpo = driver.get_qc_mpo(h1e=h1, g2e=g2, ecore=c, iprint=0)
ket = driver.get_random_mps(tag="KET", bond_dim=100, nroots=1)
energy = driver.dmrg(mpo, ket, n_sweeps=10, bond_dims=[100], noises=[1e-5] * 4 + [0], thrds=[1e-10] * 8, iprint=1)
print('DMRG energy = %20.15f' % energy)