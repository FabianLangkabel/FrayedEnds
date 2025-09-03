import madpy
import numpy as np
import tequila as tq
import time


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

def transform_rdms(TransformationMatrix, rdm1, rdm2):
    new_rdm1 = np.dot(np.dot(TransformationMatrix.transpose(), rdm1),TransformationMatrix)
    n = rdm2.shape[0]
    
    temp1 = np.zeros(shape=(n, n, n, n))
    for i in range(n):
        for j in range(n):
            for k2 in range(n):
                for l in range(n):
                    k_value = 0
                    for k in range(n):
                        k_value += TransformationMatrix[k][k2] * rdm2[i][j][k][l]
                    temp1[i][j][k2][l] = k_value
                    
    temp2 = np.zeros(shape=(n, n, n, n))
    for i2 in range(n):
        for j in range(n):
            for k2 in range(n):
                for l in range(n):
                    i_value = 0
                    for i in range(n):
                        i_value += TransformationMatrix[i][i2] * temp1[i][j][k2][l]
                    temp2[i2][j][k2][l] = i_value
                    
    temp3 = np.zeros(shape=(n, n, n, n))
    for i2 in range(n):
        for j in range(n):
            for k2 in range(n):
                for l2 in range(n):
                    l_value = 0
                    for l in range(n):
                        l_value += TransformationMatrix[l][l2] * temp2[i2][j][k2][l]
                    temp3[i2][j][k2][l2] = l_value
                    
    new_rdm2 = np.zeros(shape=(n, n, n, n))
    for i2 in range(n):
        for j2 in range(n):
            for k2 in range(n):
                for l2 in range(n):
                    j_value = 0
                    for j in range(n):
                        j_value += TransformationMatrix[j][j2] * temp3[i2][j][k2][l2]
                    new_rdm2[i2][j2][k2][l2] = j_value
                    
    return new_rdm1, new_rdm2


world = madpy.MadWorld3D(thresh=1e-4)

distance_list = [30.1631+0.025 * i for i in range(1)]
Energy_list=[]
Gradient_list=[]

for distance in distance_list:
    true_start = time.time()
    geometry = "Li 0.0 0.0 " + str(-distance/2) + "\nH 0.0 0.0 " + str(distance/2)

    pno_start = time.time()
    madpno = madpy.MadPNO(world, geometry, n_orbitals=3)
    orbitals = madpno.get_orbitals()

    nuc_repulsion = madpno.get_nuclear_repulsion()
    Vnuc = madpno.get_nuclear_potential()

    integrals = madpy.Integrals3D(world)
    orbitals = integrals.orthonormalize(orbitals=orbitals)
    

    c = nuc_repulsion
    current=0.0
    for iteration in range(10):
        print(c)
        integrals = madpy.Integrals3D(world)
        if iteration==0:
            G = integrals.compute_two_body_integrals(orbitals)
            T = integrals.compute_kinetic_integrals(orbitals)
            V = integrals.compute_potential_integrals(orbitals, Vnuc)
            S = integrals.compute_overlap_integrals(orbitals)
            print("h1:",T+V)
            print("g2:",G.elems)
            print("s:",S)
            mol = tq.Molecule(geometry, one_body_integrals=T + V, two_body_integrals=G, nuclear_repulsion=c)
            params0=mol.parameters
        else:
            G = integrals.compute_two_body_integrals(active_orbitals)
            T = integrals.compute_kinetic_integrals(active_orbitals)
            V = integrals.compute_potential_integrals(active_orbitals, Vnuc)
            S = integrals.compute_overlap_integrals(active_orbitals)
            print("h1:",T+V)
            print("g2:",G.elems)
            params0.frozen_core=False
            mol = tq.Molecule(parameters=params0, one_body_integrals=T + V, two_body_integrals=G, nuclear_repulsion=c, 
                              n_electrons=2, active_orbitals=[0,1], frozen_orbitals=[])
            
        U = mol.make_ansatz(name="HCB-UpCCGD")
        if iteration==0:
            opt = get_best_initial_values(mol)
        else:
            opt = tq.quantumchemistry.optimize_orbitals(molecule=mol, circuit=U, silent=True, use_hcb=True, initial_guess=opt.mo_coeff)
        mol_new=opt.molecule
        H = mol_new.make_hardcore_boson_hamiltonian()
        U = mol_new.make_ansatz(name="HCB-UpCCGD")
        E = tq.ExpectationValue(H=H, U=U)
        result = tq.minimize(E, silent=True, use_hcb=True)
        rdm1, rdm2 = mol.compute_rdms(U=U, variables=result.variables, use_hcb=True)
        rdm1, rdm2 = transform_rdms(opt.mo_coeff.transpose(), rdm1, rdm2)
    
        print("1-RDM:", rdm1)
        print("2-RDM:", rdm2)

        print("iteration {} energy {:+2.7f}".format(iteration, result.energy))
        if abs(result.energy-current)<1e-6:
            break
        current = result.energy
        

        opti = madpy.Optimization3D(world, Vnuc, nuc_repulsion)
        orbitals = opti.get_orbitals(orbitals=orbitals, rdm1=rdm1, rdm2=rdm2, opt_thresh=0.001, occ_thresh=0.001)
        active_orbitals = []
        for i in range(len(orbitals)):
            if orbitals[i].type == "active":
                active_orbitals.append(orbitals[i])
        c = opti.get_c() # if there are no frozen core electrons, this should always be equal to the nuclear repulsion

        for i in range(len(orbitals)):
            world.line_plot(f"orb{i}.dat", orbitals[i])
    Energy_list.append(result.energy)
    
    
    true_end = time.time()
    print("Total time: ", true_end - true_start)

print("distance_list=", distance_list)
print("Energy_list=", Energy_list)
print("Gradient_list=", Gradient_list)

del madpno
del integrals
del opti
del world

