import numpy as np
import madpy
import tequila as tq
from time import time


def get_best_initial_values(mol):
    tries = 20
    U = mol.make_ansatz(name="HCB-UpCCGD")
    best_opt = tq.quantumchemistry.optimize_orbitals(molecule=mol, circuit=U, silent=True, use_hcb=True, initial_guess="random")
    opt = tq.quantumchemistry.optimize_orbitals(molecule=mol, circuit=U, silent=True, use_hcb=True)
    if opt.energy < best_opt.energy:
        best_opt = opt
    
    for _ in range(tries):
        #opt = tq.quantumchemistry.optimize_orbitals(molecule=mol, circuit=U, silent=True, use_hcb=True, initial_guess="random")
        initial_guess = np.eye(mol.n_orbitals) + np.random.normal(scale=1.0, loc=0.0, size=mol.n_orbitals**2).reshape(mol.n_orbitals, mol.n_orbitals)
        opt = tq.quantumchemistry.optimize_orbitals(molecule=mol, circuit=U, silent=True, use_hcb=True, initial_guess=initial_guess)
        if opt.energy < best_opt.energy:
            best_opt = opt
            
    return best_opt


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

for iteration in range(1):
    start = time()
    mol = tq.Molecule(geometry=geom, one_body_integrals=h, two_body_integrals=g, nuclear_repulsion=c)
    U = mol.make_ansatz(name="HCB-UpCCGD")
    if iteration == 0:
        opt = get_best_initial_values(mol)
    else:
        opt = tq.quantumchemistry.optimize_orbitals(molecule=mol, circuit=U, silent=True, use_hcb=True, initial_guess=opt.mo_coeff)
    
    mol_new = opt.molecule
    H = mol_new.make_hardcore_boson_hamiltonian()
    U = mol_new.make_ansatz(name="HCB-UpCCGD")
    E = tq.ExpectationValue(H=H, U=U)
    result = tq.minimize(E, silent=True, use_hcb=True)
    rdm1, rdm2 = mol.compute_rdms(U, variables=result.variables)
    end = time()
    print("VQE iteration {} time: {:.2f} seconds".format(iteration, end-start))

    print("iteration {} energy {:+2.5f}".format(iteration, result.energy))
    
    start = time()
    opti = madpy.Optimization(param)
    orbitals = opti(orbitals=orbitals, rdm1=rdm1, rdm2=rdm2)
    c,h_el,g_el = opti.get_integrals()
    del opti

    h=np.array(h_el).reshape(2,2)
    g=np.array(g_el).reshape(2,2,2,2)
    g=tq.quantumchemistry.NBodyTensor(g, ordering="dirac")
    g=g.reorder(to="openfermion")
    end = time()
    print("It "+str(iteration)+" Optimization time: ", end-start)
true_end=time()
print("Total time: ", true_end-true_start)




