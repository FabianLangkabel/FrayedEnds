# SA-Excited states with DMRG + NWChem + Orbital refinement

### Parameters
distance = 1.0
iteration_energies = []
iterations = 6
molecule_name = "h2"
box_size = 50.0
wavelet_order = 7
madness_thresh = 0.0001
basisset = "6-31g"
n_elec = 2
number_roots = 3

"""
### Run NWChem calculation
Create NWChem input and run NWChem calculation. If the MadPy devcontainer or the singularity image is used, NWChem is already installed. Otherwise, NWChem has to be installed and the path has to be adjusted.
"""

import subprocess as sp

nwchem_input = (
    """
title "molecule"
memory stack 1500 mb heap 100 mb global 1400 mb
charge 0  
geometry noautosym nocenter
  H 0.0 0.0 """
    + distance.__str__()
    + """
  H 0.0 0.0 """
    + (-distance).__str__()
    + """
end
basis  
  * library """
    + basisset
    + """
end
scf  
 maxiter 200
end   
task scf  
"""
)
with open("nwchem", "w") as f:
    f.write(nwchem_input)
programm = sp.call(
    "/opt/conda/bin/nwchem nwchem",
    stdout=open("nwchem.out", "w"),
    stderr=open("nwchem_err.log", "w"),
    shell=True,
)


"""
### Convert NWChem AOs and MOs to MRA-Orbitals
Read the atomic orbitals (AOs) and molecular orbitals (MOs) from a NWChem calculation and translate them into multiwavelets.
"""

import madpy as mad

world = mad.MadWorld3D(L=box_size, k=wavelet_order, thresh=madness_thresh)

converter = mad.NWChem_Converter(world)
converter.read_nwchem_file("nwchem")
orbs = converter.get_mos()
Vnuc = converter.get_Vnuc()
nuclear_repulsion_energy = converter.get_nuclear_repulsion_energy()
del converter

for i in range(len(orbs)):
    orbs[i].type = "active"

"""
Calculate initial integrals
"""
integrals = mad.Integrals3D(world)
G = integrals.compute_two_body_integrals(orbs).elems  # Physics Notation
T = integrals.compute_kinetic_integrals(orbs)
V = integrals.compute_potential_integrals(orbs, Vnuc)
S = integrals.compute_overlap_integrals(orbs)
G_chem = G.transpose(0, 2, 1, 3)
# del integrals

"""
Performe SA DMRG calculation and extract rdms
"""

ncas = len(orbs)

import numpy as np
from pyblock2._pyscf.ao2mo import integrals as itg
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SU2, n_threads=4)
driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=0)
mpo = driver.get_qc_mpo(h1e=T + V, g2e=G_chem, ecore=nuclear_repulsion_energy, iprint=0)
ket = driver.get_random_mps(tag="KET", bond_dim=100, nroots=number_roots)
energies = driver.dmrg(
    mpo,
    ket,
    n_sweeps=10,
    bond_dims=[100],
    noises=[1e-5] * 4 + [0],
    thrds=[1e-10] * 8,
    iprint=1,
)
print("State-averaged MPS energies = [%s]" % " ".join("%20.15f" % x for x in energies))

"""
Extract rdms
"""
kets = [driver.split_mps(ket, ir, tag="KET-%d" % ir) for ir in range(ket.nroots)]
sa_1pdm = np.mean([driver.get_1pdm(k) for k in kets], axis=0)
sa_2pdm = np.mean([driver.get_2pdm(k) for k in kets], axis=0).transpose(0, 3, 1, 2)
print(
    "Energy from SA-pdms = %20.15f"
    % (
        np.einsum("ij,ij->", sa_1pdm, T + V)
        + 0.5 * np.einsum("ijkl,ijkl->", sa_2pdm, G_chem)
        + nuclear_repulsion_energy
    )
)
sa_2pdm_phys = sa_2pdm.swapaxes(1, 2)  # Physics Notation

np.savetxt("initial_energies.txt", energies)

for iter in range(iterations):

    """
    Refine orbitals
    """
    opti = mad.Optimization3D(world, Vnuc, nuclear_repulsion_energy)
    orbs = opti.get_orbitals(
        orbitals=orbs,
        rdm1=sa_1pdm,
        rdm2=sa_2pdm_phys,
        opt_thresh=0.001,
        occ_thresh=0.001,
    )

    """
  DMRG with refined orbitals
  """
    G = integrals.compute_two_body_integrals(orbs).elems  # Physics Notation
    T = integrals.compute_kinetic_integrals(orbs)
    V = integrals.compute_potential_integrals(orbs, Vnuc)
    S = integrals.compute_overlap_integrals(orbs)
    G_chem = G.transpose(0, 2, 1, 3)

    driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SU2, n_threads=4)
    driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=0)
    mpo = driver.get_qc_mpo(
        h1e=T + V, g2e=G_chem, ecore=nuclear_repulsion_energy, iprint=0
    )
    ket = driver.get_random_mps(tag="KET", bond_dim=100, nroots=number_roots)
    energies = driver.dmrg(
        mpo,
        ket,
        n_sweeps=10,
        bond_dims=[100],
        noises=[1e-5] * 4 + [0],
        thrds=[1e-10] * 8,
        iprint=1,
    )
    print(
        "State-averaged MPS energies after refinement = [%s]"
        % " ".join("%20.15f" % x for x in energies)
    )
    np.savetxt("energies_it_" + str(iter) + ".txt", energies)

    kets = [driver.split_mps(ket, ir, tag="KET-%d" % ir) for ir in range(ket.nroots)]
    sa_1pdm = np.mean([driver.get_1pdm(k) for k in kets], axis=0)
    sa_2pdm = np.mean([driver.get_2pdm(k) for k in kets], axis=0).transpose(0, 3, 1, 2)
    print(
        "Energy from SA-pdms = %20.15f"
        % (
            np.einsum("ij,ij->", sa_1pdm, T + V)
            + 0.5 * np.einsum("ijkl,ijkl->", sa_2pdm, G_chem)
            + nuclear_repulsion_energy
        )
    )
    sa_2pdm_phys = sa_2pdm.swapaxes(1, 2)  # Physics Notation

del integrals
del opti
