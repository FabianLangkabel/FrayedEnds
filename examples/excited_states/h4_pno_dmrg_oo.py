import frayedends as fe
import time
import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

iterations = 10
molecule_name = "h4"
box_size = 50.0
wavelet_order = 7
madness_thresh = 0.0001
basisset = '6-31g'
n_elec = 4
number_roots = 3
energy_tol = 1e-6
prev_energies = None
iter = 0

iteration_results = []

geometry_mode = "equidistant"  # "equidistant" or "h2_pair"
print(f"Geometry mode: {geometry_mode}")

if geometry_mode == "equidistant": # linear H4 molecule with equidistant spacing d
    distance = np.arange(2.5, 0.45, -0.05).tolist()
elif geometry_mode == "h2_pair": # for H2 pair getting closer
    distance = np.arange(1.5, 0.2, -0.03).tolist()
else:
    raise ValueError("geometry_mode must be 'equidistant' or 'h2_pair'")

with open("iteration_pno_dmrg_oo.dat", "w") as f:
    header = "distance iteration iteration_time_s " + " ".join(f"energy_{i}" for i in range(number_roots))
    f.write(header + "\n")

with open("distance_times_pno_dmrg_oo.dat", "w") as f:
    f.write("distance total_time_s\n")

with open("results_pno_dmrg_oo.dat", "w") as f:
    header = "distance " + " ".join(f"energy_{i}" for i in range(number_roots))
    f.write(header + "\n")

total_start = time.perf_counter()

for d in distance:
    dist_start = time.perf_counter()
    reported_distance = 2 * d if geometry_mode == "h2_pair" else d

    if geometry_mode == "equidistant":  # for equidistant linear H4 molecule
        geom = ("H 0.0 0.0 " + (-d - d / 2).__str__() + "\n"
                "H 0.0 0.0 " + (-d / 2).__str__() + "\n"
                "H 0.0 0.0 " + (d / 2).__str__() + "\n"
                "H 0.0 0.0 " + (d + d / 2).__str__() + "\n")
    elif geometry_mode == "h2_pair":  # for H2 molecules getting closer and closer to a H4 molecule
        geom = ("H 0.0 0.0 " + (-d - 1.5).__str__() + "\n"
                "H 0.0 0.0 " + (-d).__str__() + "\n"
                "H 0.0 0.0 " + d.__str__() + "\n"
                "H 0.0 0.0 " + (d + 1.5).__str__() + "\n")
    else:
        raise ValueError("Invalid geometry mode selected.")

    world = fe.MadWorld3D(L=box_size, k=wavelet_order, thresh=madness_thresh)

    madpno = fe.MadPNO(world, geom, n_orbitals=8)
    orbs = madpno.get_orbitals()

    nuc_repulsion = madpno.get_nuclear_repulsion()
    Vnuc = madpno.get_nuclear_potential()

    integrals = fe.Integrals3D(world)
    orbs = integrals.orthonormalize(orbitals=orbs)

    for i in range(len(orbs)):
        orbs[i].type="active"

    # for i in range(len(orbs)):
    #    world.line_plot(f"orb{i}_d{d}.dat", orbs[i], axis="z", datapoints=2001)

    '''
    Calculate initial integrals
    '''
    integrals = fe.Integrals3D(world)
    G = integrals.compute_two_body_integrals(orbs, ordering="chem").elems
    T = integrals.compute_kinetic_integrals(orbs)
    V = integrals.compute_potential_integrals(orbs, Vnuc)
    S = integrals.compute_overlap_integrals(orbs)

    '''
    Performe SA DMRG calculation and extract rdms
    '''

    n_orbitals = len(orbs)

    driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SU2, n_threads=8)
    driver.initialize_system(n_sites=n_orbitals, n_elec=n_elec, spin=0)
    mpo = driver.get_qc_mpo(h1e=T+V, g2e=G, ecore=nuc_repulsion, iprint=0)
    ket = driver.get_random_mps(tag="KET", bond_dim=100, nroots=number_roots)
    energies = driver.dmrg(mpo, ket, n_sweeps=10, bond_dims=[100], noises=[1e-5] * 4 + [0], thrds=[1e-10] * 8, iprint=1)
    print('State-averaged MPS energies = [%s]' % " ".join("%20.15f" % x for x in energies))

    '''
    Extract rdms
    '''
    kets = [driver.split_mps(ket, ir, tag="KET-%d" % ir) for ir in range(ket.nroots)]
    sa_1pdm = np.mean([driver.get_1pdm(k) for k in kets], axis=0)
    sa_2pdm = np.mean([driver.get_2pdm(k) for k in kets], axis=0).transpose(0,3,1,2)
    print('Energy from SA-pdms = %20.15f' % (np.einsum('ij,ij->', sa_1pdm, T+V) + 0.5 * np.einsum('ijkl,ijkl->', sa_2pdm, G) + nuc_repulsion))
    sa_2pdm_phys = sa_2pdm.swapaxes(1,2) #Physics Notation

    with open("iteration_pno_dmrg_oo.dat", "a") as f:
        f.write(f"{reported_distance:.3f} {-1} {0.00} " + " ".join(f"{x:.15f}" for x in energies) + "\n")

    for iter in range(iterations):
        iter_start = time.perf_counter()

        '''
        Refine orbitals
        '''
        opti = fe.Optimization3D(world, Vnuc, nuc_repulsion)
        orbs = opti.get_orbitals(orbitals=orbs, rdm1=sa_1pdm, rdm2=sa_2pdm_phys, opt_thresh=0.001, occ_thresh=0.001)
        # for i in range(len(orbs)):
        #    world.line_plot(f"orb{i}_d{d}.dat", orbs[i], axis="z", datapoints=2001))

        '''
        DMRG with refined orbitals
        '''
        G = integrals.compute_two_body_integrals(orbs, ordering="chem").elems
        T = integrals.compute_kinetic_integrals(orbs)
        V = integrals.compute_potential_integrals(orbs, Vnuc)
        S = integrals.compute_overlap_integrals(orbs)

        driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SU2, n_threads=8)
        driver.initialize_system(n_sites=n_orbitals, n_elec=n_elec, spin=0)
        mpo = driver.get_qc_mpo(h1e=T+V, g2e=G, ecore=nuc_repulsion, iprint=0)
        ket = driver.get_random_mps(tag="KET", bond_dim=100, nroots=number_roots)
        energies = driver.dmrg(mpo, ket, n_sweeps=10, bond_dims=[100], noises=[1e-5] * 4 + [0], thrds=[1e-10] * 8, iprint=1)
        print('State-averaged MPS energies after refinement = [%s]' % " ".join("%20.15f" % x for x in energies))

        kets = [driver.split_mps(ket, ir, tag="KET-%d" % ir) for ir in range(ket.nroots)]
        sa_1pdm = np.mean([driver.get_1pdm(k) for k in kets], axis=0)
        sa_2pdm = np.mean([driver.get_2pdm(k) for k in kets], axis=0).transpose(0,3,1,2)
        print('Energy from SA-pdms = %20.15f' % (np.einsum('ij,ij->', sa_1pdm, T+V) + 0.5 * np.einsum('ijkl,ijkl->', sa_2pdm, G) + nuc_repulsion))
        sa_2pdm_phys = sa_2pdm.swapaxes(1,2) #Physics Notation

        iter_end = time.perf_counter()
        iter_time = iter_end - iter_start

        with open("iteration_pno_dmrg_oo.dat", "a") as f:
            f.write(f"{reported_distance:.3f} {iter} {iter_time:.2f} " + " ".join(f"{x:.15f}" for x in energies) + "\n")

        iteration_results.append({"distance": reported_distance, "iteration": iter, "iteration_time": iter_time, "energies": energies})

        if prev_energies is not None:
            delta_e = np.max(np.abs(energies - prev_energies))
            if delta_e < energy_tol:
                break

        prev_energies = energies

    with open("results_pno_dmrg_oo.dat", "a") as f:
        f.write(f"{reported_distance:.3f} " + " ".join(f"{x:.15f}" for x in energies) + "\n")

    dist_end = time.perf_counter()
    dist_time = dist_end - dist_start
    print(f"Distance {reported_distance:.3f} took {dist_time:.2f} s")
    with open("distance_times_pno_dmrg_oo.dat", "a") as f:
        f.write(f"{reported_distance:.3f} {dist_time:.2f}\n")

    del integrals
    del opti
    del madpno
    del world