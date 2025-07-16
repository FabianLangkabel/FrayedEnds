import tequila as tq
import sunrise as hb
import pickle
import sys
sys.path.insert(1,'/Users/jakobkottmann/devel/spafastprototype/')
from decompose import decompose
from spa import run_spa, SPASolver, make_decomposed_clusters
from time import time
import numpy as np

geometry = '''
C 0.3247070 1.8174900 0.0000000
C -0.3247070 0.6505420 0.0000000
C 0.3247070 -0.6505420 0.0000000
C -0.3247070 -1.8174900 0.0000000
H -1.4120610 0.6458230 0.0000000
H 1.4120610 -0.6458230 0.0000000
H -0.2040530 2.7618040 0.0000000
H 1.4083260 1.8629110 0.0000000
H 0.2040530 -2.7618040 0.0000000
H -1.4083260 -1.8629110 0.0000000'''

def init_buta():
    mol = hb.Molecule(geometry=geometry, basis_set='sto-3g', backend='pyscf').use_native_orbitals()
    tqmol = tq.Molecule(geometry=geometry, basis_set='sto-3g', backend='pyscf').use_native_orbitals()
    print(tqmol)
    print(tqmol.n_electrons)
    print(tqmol.n_orbitals)
    edges = mol.get_spa_edges()
    guess = mol.get_spa_guess()
    beg = time()
    opt = run_spa(mol=tqmol, edges=edges,initial_guess=guess.T, decompose=True, silent=True,grouping=12, fast_rdm=True)
    # opt = mol.optimize_orbitals(tqmol,circuit=tqmol.make_ansatz('HCB-SPA',edges=edges), initial_guess=guess.T, silent=True,use_hcb=True)
    hb.plot_MO(opt.molecule,filename='buthadiene')
    print('Time ',time()-beg)
    with open('Buthadiene_SPA_plane.data', 'wb') as file:
        pickle.dump(opt.molecule, file)

    print("finished orbital optimization")
    print(opt.energy)

    U = mol.make_ansatz(name="HCB-SPA", edges=edges)
    return opt.molecule, U

def solve_spa(mol, U):
    connections = U.to_networkx()
    grouping = make_decomposed_clusters(grouping=12, connections=connections)
    vqe_solver = SPASolver(decompose=True,grouping=grouping, fast_rdm=True)
    result = vqe_solver(circuit=U, molecule=mol, H=mol.make_hardcore_boson_hamiltonian())
    print(result.energy)
    rdm1, rdm2 = vqe_solver.compute_rdms(U, variables=result.variables, molecule=mol, use_hcb=True)
    return result, rdm1, rdm2

mol, U= init_buta()
print("yes")
c,h,g = mol.get_integrals(ignore_active_space=True)
mol = tq.Molecule(geometry=geometry, one_body_integrals=h, two_body_integrals=g, nuclear_repulsion=c)
print(mol)
print(mol.n_electrons)
print(mol.n_orbitals)
result, rdm1, rdm2 = solve_spa(mol, U)
c = mol.integral_manager.orbital_coefficients
print(c)