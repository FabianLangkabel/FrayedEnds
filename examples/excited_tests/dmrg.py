# SA-Excited states with DMRG + NWChem + Orbital refinement

### Parameters
distance = 2.5
iteration_energies = []
iterations = 6
molecule_name = "h2"
box_size = 50.0
wavelet_order = 7
madness_thresh = 0.0001
basisset = '6-31g'

'''
### Run NWChem calculation
Create NWChem input and run NWChem calculation. If the MadPy devcontainer or the singularity image is used, NWChem is already installed. Otherwise, NWChem has to be installed and the path has to be adjusted.
'''

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


'''
### Convert NWChem AOs and MOs to MRA-Orbitals
Read the atomic orbitals (AOs) and molecular orbitals (MOs) from a NWChem calculation and translate them into multiwavelets.
'''

import madpy as mad
world = mad.MadWorld(L=box_size, k=wavelet_order, thresh=madness_thresh)

converter = mad.NWChem_Converter(world)
converter.read_nwchem_file("nwchem")
mos = converter.get_mos()
Vnuc = converter.get_Vnuc()
nuclear_repulsion_energy = converter.get_nuclear_repulsion_energy()
del converter

#opti = mad.Optimization(world, Vnuc, nuc_repulsion)


'''
from pyscf import gto, scf
import numpy as np
from pyblock2._pyscf.ao2mo import integrals as itg
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="sto3g", symmetry="d2h", verbose=0)
mf = scf.RHF(mol).run(conv_tol=1E-14)
ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(mf, ncore=0, ncas=None, g2e_symm=8)

driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SU2, n_threads=4)
driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym)

mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=0)

ket = driver.get_random_mps(tag="KET", bond_dim=100, nroots=3)
energies = driver.dmrg(mpo, ket, n_sweeps=10, bond_dims=[100], noises=[1e-5] * 4 + [0], thrds=[1e-10] * 8, iprint=1)

print('State-averaged MPS energies = [%s]' % " ".join("%20.15f" % x for x in energies))
'''