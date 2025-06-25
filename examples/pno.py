import numpy as np
import madpy
import tequila as tq
from time import time

true_start=time()
start = time()
# initialize the PNO interface
geom = "H 0.0 0.0 -1.25\nH 0.0 0.0 1.25"
madpno = madpy.MadPNO(geom, maxrank=1, pnoint={"n_pno":1})
orbitals = madpno.get_orbitals()
param = madpno.madness_parameters
c, h, g = madpno.get_integrals()
del madpno
#todo: add cleanup

end=time()
print("PNO time: ", end-start)

for iteration in range(6):
    start = time()
    mol = tq.Molecule(geometry=geom, one_body_integrals=h, two_body_integrals=g, nuclear_repulsion=c)
    U = mol.make_ansatz(name="UpCCGD")
    H = mol.make_hamiltonian()
    E = tq.ExpectationValue(H=H, U=U)
    result = tq.minimize(E, silent=True)
    rdm1, rdm2 = mol.compute_rdms(U, variables=result.variables)
    end = time()
    print("VQE iteration {} time: {:.2f} seconds".format(iteration, end-start))

    print("iteration {} energy {:+2.5f}".format(iteration, result.energy))
    
    start = time()
    opt = madpy.Optimization(param)
    orbitals = opt(orbitals=orbitals, rdm1=rdm1, rdm2=rdm2)
    c,h_el,g_el = opt.get_integrals()
    del opt

    h=np.array(h_el).reshape(2,2)
    g=np.array(g_el).reshape(2,2,2,2)
    g=tq.quantumchemistry.NBodyTensor(g, ordering="dirac")
    g=g.reorder(to="openfermion")
    end = time()
    print("It "+str(iteration)+" Optimization time: ", end-start)
true_end=time()
print("Total time: ", true_end-true_start)




