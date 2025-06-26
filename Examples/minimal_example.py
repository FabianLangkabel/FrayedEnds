import tequila as tq
import numpy as np
import madpy as mad
from pyscf import fci
import OrbOpt_helper

iterations=6

box_size = 50.0 # the system is in a volume of dimensions (box_size*2)^3 (Bohr units)
wavelet_order = 7
madness_thresh = 0.00001

optimization_thresh = 0.0001
NO_occupation_thresh = 0.0001

all_orbitals = [0,1]
frozen_occupied_orbitals = []
active_orbitals = [0,1]
as_dim=len(active_orbitals)

dummy_molecule="He 0.0 0.0 0.0" # Specify the number of electrons in the system by changing the atom (number of e needs to be even)

def potential(x,y,z):
    a = -5.0
    r = np.array([x,y,z])

    return a * np.exp(- 0.5 * np.linalg.norm(r)**2)


factory=mad.PyFuncFactory(box_size, wavelet_order, madness_thresh, potential)
mra_pot=factory.GetMRAFunction()
del factory

opti=mad.Optimization(box_size, wavelet_order, madness_thresh)
opti.plot("custom_pot.dat",mra_pot, axis=0)
opti.plane_plot("pot.dat",mra_pot,plane="yz",zoom=10.0,datapoints=71,origin=[0.0,0.0,0.0])
del opti


eigensolver = mad.Eigensolver(box_size, wavelet_order, madness_thresh)
eigensolver.solve(mra_pot, 2, 5)
all_orbs=eigensolver.GetOrbitals(len(frozen_occupied_orbitals),as_dim,0)
del eigensolver

integrals = mad.Integrals(box_size, wavelet_order, madness_thresh)
G_elems = integrals.compute_two_body_integrals(all_orbs)
G = tq.quantumchemistry.NBodyTensor(elems=G_elems, ordering="phys").elems
T = integrals.compute_kinetic_integrals(all_orbs)
V = integrals.compute_potential_integrals(all_orbs, mra_pot)
S = integrals.compute_overlap_integrals(all_orbs)
del integrals
h1=T+V
g2=G
c=0.0
print("done")
for it in range(iterations):
    mol = tq.Molecule(dummy_molecule, one_body_integrals=h1, two_body_integrals=g2, nuclear_repulsion=c)
    e, fcivec = fci.direct_spin0.kernel(h1, g2, mol.n_orbitals, mol.n_electrons)
    rdm1, rdm2 = fci.direct_spin0.make_rdm12(fcivec, mol.n_orbitals, mol.n_electrons)
    rdm2 = np.swapaxes(rdm2, 1, 2)
    print("FCI energy: " + str(e))

    rdm1_list = rdm1.reshape(-1).tolist()
    rdm2_list = rdm2.reshape(-1).tolist()

    # Orbital-Optimization
    red2 = mad.RedirectOutput("OrbOpt" + str(it) + ".log")
    opti = mad.Optimization(box_size, wavelet_order, madness_thresh)
    opti.nocc = 2;  # spatial orbital = 2; spin orbitals = 1
    opti.truncation_tol = 1e-6
    opti.coulomb_lo = 0.001
    opti.coulomb_eps = 1e-6
    opti.BSH_lo = 0.01
    opti.BSH_eps = 1e-6

    print("Read rdms, create initial guess and calculate initial energy")
    opti.GiveCustomPotential(mra_pot)
    opti.GiveInitialOrbitals(all_orbs)
    opti.GiveRDMsAndRotateOrbitals(rdm1_list, rdm2_list)
    opti.CalculateAllIntegrals()
    opti.CalculateCoreEnergy()
    opti.CalculateEnergies()

    print("---------------------------------------------------")
    print("Start orbital optimization")
    opti.OptimizeOrbitals(optimization_thresh, NO_occupation_thresh)

    opti.RotateOrbitalsBackAndUpdateIntegrals()

    all_orbs = opti.GetOrbitals()
    c = opti.GetC()
    h1_elements = opti.GetHTensor()
    g2_elements = opti.GetGTensor()
    for i in range(len(all_orbs)):
        opti.plot("orbital_" + i.__str__()+".dat", all_orbs[i])
        opti.plane_plot(i.__str__()+"orb.dat",all_orbs[i],plane="yz",zoom=10.0,datapoints=71,origin=[0.0,0.0,0.0])
    del opti
    del red2









