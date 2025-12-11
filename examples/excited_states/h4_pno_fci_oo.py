import frayedends as fe
import time
import numpy as np
from pyscf import fci

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

geometry_mode = "equidistant"  # "equidistant" or "h2_pair"
print(f"Geometry mode: {geometry_mode}")

if geometry_mode == "equidistant": # linear H4 molecule with equidistant spacing d
    distance = np.arange(2.5, 0.45, -0.05).tolist()
elif geometry_mode == "h2_pair": # for H2 pair getting closer
    distance = np.arange(1.5, 0.2, -0.03).tolist()
else:
    raise ValueError("geometry_mode must be 'equidistant' or 'h2_pair'")

with open("iterations_pno_fci_oo.dat", "w") as f:
    header = "distance iteration iteration_time_s energy_0"
    f.write(header + "\n")

with open("distance_times_pno_fci_oo.dat", "w") as f:
    f.write("distance total_time_s\n")

with open("results_pno_fci_oo.dat", "w") as f:
    header = "distance energy_0"
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

    n_orbitals = len(orbs)

    current = 0.0
    for iteration in range(iterations):
        iter_start = time.perf_counter()

        integrals = fe.Integrals3D(world)
        G = integrals.compute_two_body_integrals(orbs, ordering="chem").elems
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
        #    world.line_plot(f"orb{i}_d{d}.dat", orbs[i], axis="z", datapoints=2001)

        iter_end = time.perf_counter()
        iter_time = iter_end - iter_start

        with open("iterations_pno_fci_oo.dat", "a") as f:
            f.write(f"{reported_distance:.3f} {iteration} {iter_time:.2f} {e_tot: .15f}" + "\n")

        iterations_results.append({"distance": reported_distance, "iteration": iteration, "iteration_time": iter_time, "energy": e_tot})

        if np.isclose(e_tot, current, atol=econv, rtol=0.0):
            break  # The loop terminates as soon as the energy changes less than econv in one iteration step
        current = e_tot

    with open("results_pno_fci_oo.dat", "a") as f:
        f.write(f"{reported_distance:.3f} {e_tot: .15f}" + "\n")

    dist_end = time.perf_counter()
    dist_time = dist_end - dist_start
    print(f"Distance {reported_distance:.3f} took {dist_time:.2f} s")
    with open("distance_times_pno_fci_oo.dat", "a") as f:
        f.write(f"{reported_distance:.3f} {dist_time:.2f}\n")

    del integrals
    del opti
    del madpno
    del world