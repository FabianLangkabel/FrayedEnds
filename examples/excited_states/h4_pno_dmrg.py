import frayedends as fe
import time
import numpy as np

distance = np.arange(1.5, 0.2, -0.03).tolist() # for H2 pair getting closer
# distance = np.arange(2.5, 0.45, -0.05).tolist()
iteration_energies = []
iterations = 6
molecule_name = "h4"
box_size = 50.0
wavelet_order = 7
madness_thresh = 0.0001
basisset = '6-31g'
n_elec = 4
number_roots = 3

results = []

with open("results_pno_dmrg.dat", "w") as f:
    header = "distance iteration iteration_time_s " + " ".join(f"energy_{i}" for i in range(number_roots))
    f.write(header + "\n")

with open("distance_times_pno_dmrg.dat", "w") as f:
    f.write("distance total_time_s\n")

total_start = time.perf_counter()

for d in distance:
    dist_start = time.perf_counter()
    # linear H4 molecule with equidistant spacing d
    '''
    geom = ("H 0.0 0.0 " + (-d - d / 2).__str__() + "\n"
            "H 0.0 0.0 " + (-d / 2).__str__() + "\n"
            "H 0.0 0.0 " + (d / 2).__str__() + "\n"
            "H 0.0 0.0 " + (d + d / 2).__str__() + "\n"
            )

    '''
    # for H2 molecules getting closer and closer to a H4 molecule
    geom = ("H 0.0 0.0 " + (-d - 2.55).__str__() + "\n"
            "H 0.0 0.0 " + (-d).__str__() + "\n"
            "H 0.0 0.0 " + d.__str__() + "\n"
            "H 0.0 0.0 " + (d + 2.55).__str__() + "\n"
            )


    world = fe.MadWorld3D(L=box_size, k=wavelet_order, thresh=madness_thresh)

    madpno = fe.MadPNO(world, geom, n_orbitals=8)
    orbs = madpno.get_orbitals()

    nuc_repulsion = madpno.get_nuclear_repulsion()
    Vnuc = madpno.get_nuclear_potential()

    integrals = fe.Integrals3D(world)
    orbitals = integrals.orthonormalize(orbitals=orbs)

    for i in range(len(orbs)):
        orbs[i].type="active"

    # for i in range(len(orbs)):
    #    world.line_plot(f"orb{i}_d{d}.dat", orbs[i])

    '''
    Calculate initial integrals
    '''
    integrals = fe.Integrals3D(world)
    G = integrals.compute_two_body_integrals(orbs, ordering="chem").elems #Physics Notation
    T = integrals.compute_kinetic_integrals(orbs)
    V = integrals.compute_potential_integrals(orbs, Vnuc)
    S = integrals.compute_overlap_integrals(orbs)
    G_chem = G
    #del integrals

    '''
    Performe SA DMRG calculation and extract rdms
    '''

    ncas = len(orbs)

    from pyblock2.driver.core import DMRGDriver, SymmetryTypes
    import numpy as np

    driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SU2, n_threads=8)
    driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=0)
    mpo = driver.get_qc_mpo(h1e=T+V, g2e=G_chem, ecore=nuc_repulsion, iprint=0)
    ket = driver.get_random_mps(tag="KET", bond_dim=100, nroots=number_roots)
    energies = driver.dmrg(mpo, ket, n_sweeps=10, bond_dims=[100], noises=[1e-5] * 4 + [0], thrds=[1e-10] * 8, iprint=1)
    print('State-averaged MPS energies = [%s]' % " ".join("%20.15f" % x for x in energies))

    '''
    Extract rdms
    '''
    kets = [driver.split_mps(ket, ir, tag="KET-%d" % ir) for ir in range(ket.nroots)]
    sa_1pdm = np.mean([driver.get_1pdm(k) for k in kets], axis=0)
    sa_2pdm = np.mean([driver.get_2pdm(k) for k in kets], axis=0).transpose(0,3,1,2)
    print('Energy from SA-pdms = %20.15f' % (np.einsum('ij,ij->', sa_1pdm, T+V) + 0.5 * np.einsum('ijkl,ijkl->', sa_2pdm, G_chem) + nuc_repulsion))
    sa_2pdm_phys = sa_2pdm.swapaxes(1,2) #Physics Notation

    # np.savetxt("initial_energies.txt", energies)

    with open("results_pno_dmrg.dat", "a") as f:
        f.write(f"{d:.6f} {-1} {0.00} " + " ".join(f"{x:.15f}" for x in energies) + "\n") # for H2 pair use 2*d

    for iter in range(iterations):
        iter_start = time.perf_counter()

        '''
        Refine orbitals
        '''
        opti = fe.Optimization3D(world, Vnuc, nuc_repulsion)
        orbs = opti.get_orbitals(orbitals=orbs, rdm1=sa_1pdm, rdm2=sa_2pdm_phys, opt_thresh=0.001, occ_thresh=0.001)
        # for i in range(len(orbs)):
        #    world.line_plot(f"orb{i}_d{d}.dat", orbs[i])

        '''
        DMRG with refined orbitals
        '''
        G = integrals.compute_two_body_integrals(orbs).elems #Physics Notation
        T = integrals.compute_kinetic_integrals(orbs)
        V = integrals.compute_potential_integrals(orbs, Vnuc)
        S = integrals.compute_overlap_integrals(orbs)
        G_chem = G.transpose(0,2,1,3)

        driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SU2, n_threads=8)
        driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=0)
        mpo = driver.get_qc_mpo(h1e=T+V, g2e=G_chem, ecore=nuc_repulsion, iprint=0)
        ket = driver.get_random_mps(tag="KET", bond_dim=100, nroots=number_roots)
        energies = driver.dmrg(mpo, ket, n_sweeps=10, bond_dims=[100], noises=[1e-5] * 4 + [0], thrds=[1e-10] * 8, iprint=1)
        print('State-averaged MPS energies after refinement = [%s]' % " ".join("%20.15f" % x for x in energies))
        # np.savetxt("energies_it_" + str(iter) + ".txt", energies)


        kets = [driver.split_mps(ket, ir, tag="KET-%d" % ir) for ir in range(ket.nroots)]
        sa_1pdm = np.mean([driver.get_1pdm(k) for k in kets], axis=0)
        sa_2pdm = np.mean([driver.get_2pdm(k) for k in kets], axis=0).transpose(0,3,1,2)
        print('Energy from SA-pdms = %20.15f' % (np.einsum('ij,ij->', sa_1pdm, T+V) + 0.5 * np.einsum('ijkl,ijkl->', sa_2pdm, G_chem) + nuc_repulsion))
        sa_2pdm_phys = sa_2pdm.swapaxes(1,2) #Physics Notation

        iter_end = time.perf_counter()
        iter_time = iter_end - iter_start
        print(f"Iteration {iter} time: {iter_time:.2f} s")
        with open("iteration_times_pno_dmrg.dat", "a") as f:
            f.write(f"{2*d:.6f} {iter} {iter_time:.6f}\n") # for H2 pair use 2*d

        with open("results_pno_dmrg_2.dat", "a") as f:
            f.write(f"{2*d:.6f} {iter} {iter_time:.6f} " + " ".join(f"{x:.15f}" for x in energies) + "\n") # for H2 pair use 2*d

        results.append({"distance": 2*d, "iteration": iter, "iteration_time": iter_time, "energies": energies}) # for H2 pair use 2*d

    dist_end = time.perf_counter()
    dist_time = dist_end - dist_start
    print(f"Distance {2*d:.6f} took {dist_time:.2f} s") # for H2 pair use 2*d
    with open("distance_times_pno_dmrg.dat", "a") as f:
        f.write(f"{2*d:.6f} {dist_time:.6f}\n") # for H2 pair use 2*d

    del integrals
    del opti
    del madpno
    del world

total_end = time.perf_counter()
total_time = total_end - total_start
print(f"Total runtime: {total_time:.2f} s")
with open("total_time_pno_dmrg.dat", "w") as f:
    f.write(f"total_runtime_s {total_time:.6f}\n")