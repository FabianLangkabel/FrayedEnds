import numpy
import madpy
import tequila as tq

# initialize the PNO interface
geom = "H 0.0 0.0 0.0\nH 0.0 0.0 15.0"
madpno = madpy.MadPNO(geom, maxrank=1, pnoint={"n_pno":1})
orbitals = madpno.get_orbitals()
param = madpno.madness_parameters
Vnuc = madpno.get_nuclear_potential()
nuclear_repulsion = madpno.get_nuclear_repulsion()
del madpno

for iteration in range(2):

    integrals = madpy.Integrals(param)
    G = integrals.compute_two_body_integrals(orbitals).elems
    T = integrals.compute_kinetic_integrals(orbitals)
    V = integrals.compute_potential_integrals(orbitals, Vnuc)
    S = integrals.compute_overlap_integrals(orbitals)
    del integrals

    mol = tq.Molecule(geometry=geom, one_body_integrals=T+V, two_body_integrals=G, nuclear_repulsion=nuclear_repulsion)
    U = mol.make_ansatz(name="UpCCGD")
    H = mol.make_hamiltonian()
    E = tq.ExpectationValue(H=H, U=U)
    result = tq.minimize(E, silent=True)
    rdm1, rdm2 = mol.compute_rdms(U, variables=result.variables)

    print("iteration {} energy {:+2.5f}".format(iteration, result.energy))

    opt = madpy.Optimization(param)
    orbitals = opt(orbitals=orbitals, rdm1=rdm1, rdm2=rdm2)
    del opt






