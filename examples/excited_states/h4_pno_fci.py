import frayedends as fe
import time
import numpy as np
from pyscf import fci

distance = np.arange(1.5, 0.2, -0.03).tolist() # for H2 pair getting closer
# distance = np.arange(2.5, 0.50, -0.05).tolist()
iteration_energies = []
iterations = 15
molecule_name = "h4"
box_size = 50.0
wavelet_order = 7
madness_thresh = 0.0001
basisset = '6-31g'
n_electrons = 4
econv = 1.e-6 # Energy convergence threshold

iterations_results = []

with open("iterations_pno_fci.dat", "w") as f:
    header = "distance iteration iteration_time_s energy_0"
    f.write(header + "\n")

with open("distance_times_pno_fci.dat", "w") as f:
    f.write("distance total_time_s\n")

with open("results_pno_fci.dat", "w") as f:
    header = "distance energy_0"
    f.write(header + "\n")

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
    orbs = integrals.orthonormalize(orbitals=orbs)
    for i in range(len(orbs)):
        orbs[i].type="active"

    # for i in range(len(orbs)):
    #    world.line_plot(f"orb{i}_d{d}.dat", orbs[i])

    n_orbitals = len(orbs)

    current = 0.0
    for iteration in range(iterations):
        iter_start = time.perf_counter()

        integrals = fe.Integrals3D(world)
        G = integrals.compute_two_body_integrals(orbs, ordering="chem").elems #Physics Notation
        T = integrals.compute_kinetic_integrals(orbs)
        V = integrals.compute_potential_integrals(orbs, Vnuc)

        # FCI calculation
        e, fcivec = fci.direct_spin0.kernel(T + V, G, n_orbitals,
                                            n_electrons)  # Computes the energy and the FCI vector
        rdm1, rdm2 = fci.direct_spin0.make_rdm12(fcivec, n_orbitals,
                                                 n_electrons)  # Computes the 1- and 2- body reduced density matrices
        rdm2 = np.swapaxes(rdm2, 1, 2)

        e_tot = e + nuc_repulsion

        print("iteration {} FCI electronic energy {:+2.8f}, total energy {:+2.8f}".format(iteration, e, e_tot))

        opti = fe.Optimization3D(world, Vnuc, nuc_repulsion)
        orbs = opti.get_orbitals(orbitals=orbs, rdm1=rdm1, rdm2=rdm2, opt_thresh=0.001, occ_thresh=0.001)
        # for i in range(len(orbs)):
        #    world.line_plot(f"orb{i}_d{d}.dat", orbs[i])

        iter_end = time.perf_counter()
        iter_time = iter_end - iter_start

        with open("iterations_pno_fci.dat", "a") as f:
            f.write(f"{2*d:.6f} {iteration} {iter_time:.6f} {e_tot: .15f}" + "\n") # for H2 pair use 2*d

        iterations_results.append({"distance": 2 * d, "iteration": iteration, "iteration_time": iter_time, "energy": e_tot}) # for H2 pair use 2*d

        if np.isclose(e_tot, current, atol=econv, rtol=0.0):
            break  # The loop terminates as soon as the energy changes less than econv in one iteration step
        current = e_tot

    with open("results_pno_fci.dat", "a") as f:
        f.write(f"{2*d:.6f} {e_tot: .15f}" + "\n") # for H2 pair use 2*d

    dist_end = time.perf_counter()
    dist_time = dist_end - dist_start
    print(f"Distance {2*d:.6f} took {dist_time:.2f} s") # for H2 pair use 2*d
    with open("distance_times_pno_fci.dat", "a") as f:
        f.write(f"{2*d:.6f} {dist_time:.6f}\n") # for H2 pair use 2*d

    del integrals
    del opti
    del madpno
    del world

total_end = time.perf_counter()
total_time = total_end - total_start
print(f"Total runtime: {total_time:.2f} s")
with open("total_time_pno_fci.dat", "w") as f:
    f.write(f"total_runtime_s {total_time:.6f}\n")