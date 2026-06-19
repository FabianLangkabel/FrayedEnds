import time

import numpy as np
from pyscf import fci

import frayedends as fe

iterations = 1
molecule_name = "h4"
box_size = 50.0
wavelet_order = 7
madness_thresh = 0.0001
basisset = "6-31g"
n_elec = 4
number_roots = 3

iteration_results = []

geometry_mode = "h2_pair"  # "equidistant" or "h2_pair"
print(f"Geometry mode: {geometry_mode}")

if geometry_mode == "equidistant":  # linear H4 molecule with equidistant spacing d
    distance = np.arange(2.5, 0.45, -0.05).tolist()
elif geometry_mode == "h2_pair":  # for H2 pair getting closer
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
        geom = (
            "H 0.0 0.0 " + (-d - d / 2).__str__() + "\n"
            "H 0.0 0.0 " + (-d / 2).__str__() + "\n"
            "H 0.0 0.0 " + (d / 2).__str__() + "\n"
            "H 0.0 0.0 " + (d + d / 2).__str__() + "\n"
        )
    elif geometry_mode == "h2_pair":  # for H2 molecules getting closer and closer to a H4 molecule
        geom = (
            "H 0.0 0.0 " + (-d - 1.5).__str__() + "\n"
            "H 0.0 0.0 " + (-d).__str__() + "\n"
            "H 0.0 0.0 " + d.__str__() + "\n"
            "H 0.0 0.0 " + (d + 1.5).__str__() + "\n"
        )
    else:
        raise ValueError("Invalid geometry mode selected.")

    world = fe.MadWorld(ndims=3, L=box_size, k=wavelet_order, thresh=madness_thresh)
    integrals = fe.Integrals(world)

    pno_start = time.perf_counter()
    madpno = fe.MadPNO(world, geom, n_orbitals=4)
    pno_end = time.perf_counter()
    pno_time = pno_end - pno_start
    print("Generating PNOs took %.2f seconds" % pno_time)

    gs_orbs = madpno.get_orbitals()

    # for i in range(len(gs_orbs)):
    # world.cube_plot(f"gs_orb{i}", gs_orbs[i], molecule, zoom=4.0)

    cis_start = time.perf_counter()
    cis_orbs = madpno.compute_cis(n_excitation=2) # Compute CIS for 2 excitations (1st and 2nd excited states)
    cis_orbs = integrals.project_out(gs_orbs, cis_orbs)
    cis_orbs = integrals.orthonormalize(cis_orbs)
    cis_end = time.perf_counter()
    cis_time = cis_end - cis_start
    print("Generating CIS X Functions took %.2f seconds" % cis_time)

    # for i in range(len(cis_orbs)):
    #     world.cube_plot(f"cis_orb{i}", cis_orbs[i], molecule, zoom=4.0)

    cispd_start = time.perf_counter()
    cispd_orbs = madpno.compute_cispd(n_orbitals=4)
    cispd_orbs = integrals.project_out(gs_orbs + cis_orbs, cispd_orbs)
    cispd_end = time.perf_counter()
    cispd_time = cispd_end - cispd_start

    print("Generating CISPD PNOs took %.2f seconds" % cispd_time)

    # for i in range(len(cispd_orbs)):
    #     world.cube_plot(f"cispd_orb{i}", cispd_orbs[i], molecule, zoom=4.0)

    nuc_repulsion = madpno.get_nuclear_repulsion()
    Vnuc = madpno.get_nuclear_potential()

    orbs = gs_orbs + cis_orbs + cispd_orbs
    orbs = integrals.orthonormalize(orbitals=orbs)

    # for i in range(len(orbs)):
    #    world.line_plot(f"orb{i}_d{d}.dat", orbs[i], axis="z", datapoints=2001)

    n_orbitals = len(orbs)

    """
    Calculate initial integrals
    """
    integrals = fe.Integrals(world)
    G = integrals.compute_two_body_integrals(orbs, ordering="chem").elems
    T = integrals.compute_kinetic_integrals(orbs)
    V = integrals.compute_potential_integrals(orbs, Vnuc)
    h1 = T + V

    e_roots, fcivecs = fci.direct_spin0.kernel(h1, G, n_orbitals, n_elec, nroots=3)
    print("State-averaged FCI energies = [%s]" % " ".join("%20.15f" % (x + nuc_repulsion) for x in e_roots))

    all_1pdms = []
    all_2pdms = []

    for fcivec in fcivecs:
        rdm1_state, rdm2_state = fci.direct_spin0.make_rdm12(fcivec, n_orbitals, n_elec)
        all_1pdms.append(rdm1_state)
        all_2pdms.append(rdm2_state)

    sa_1pdm = np.mean(all_1pdms, axis=0)
    sa_2pdm_phys = np.mean(all_2pdms, axis=0)
    sa_2pdm_chem = sa_2pdm_phys.swapaxes(1, 2)  #chem notation
    print(
        "Energy from SA-pdms = %20.15f"
        % (np.einsum("ij,ij->", sa_1pdm, h1) + 0.5 * np.einsum("ijkl,ikjl->", sa_2pdm_chem, G) + nuc_repulsion)
    )

    with open("iteration_pno_dmrg_oo.dat", "a") as f:
        f.write(f"{reported_distance:.3f} {-1} {0.00} " + " ".join(f"{(x + nuc_repulsion):.15f}" for x in e_roots) + "\n")

    for iter in range(iterations):
        iter_start = time.perf_counter()

        """
        Refine orbitals
        """
        opti = fe.OrbitalRefinement(world, Vnuc, nuc_repulsion)
        orbs = opti.get_orbitals(orbitals=orbs, rdm1=sa_1pdm, rdm2=sa_2pdm_phys, opt_thresh=0.001, occ_thresh=0.001)
        # for i in range(len(orbs)):
        #    world.line_plot(f"orb{i}_d{d}.dat", orbs[i], axis="z", datapoints=2001))

        G = integrals.compute_two_body_integrals(orbs, ordering="chem").elems
        T = integrals.compute_kinetic_integrals(orbs)
        V = integrals.compute_potential_integrals(orbs, Vnuc)
        h1 = T + V

        e_roots, fcivecs = fci.direct_spin0.kernel(h1, G, n_orbitals, n_elec, nroots=3)
        print("State-averaged FCI energies = [%s]" % " ".join("%20.15f" % (x + nuc_repulsion) for x in e_roots))

        all_1pdms = []
        all_2pdms = []

        for fcivec in fcivecs:
            rdm1_state, rdm2_state = fci.direct_spin0.make_rdm12(fcivec, n_orbitals, n_elec)
            all_1pdms.append(rdm1_state)
            all_2pdms.append(rdm2_state)

        sa_1pdm = np.mean(all_1pdms, axis=0)
        sa_2pdm_phys = np.mean(all_2pdms, axis=0)
        sa_2pdm_chem = sa_2pdm_phys.swapaxes(1, 2)  # chem notation
        print(
            "Energy from SA-pdms = %20.15f"
            % (np.einsum("ij,ij->", sa_1pdm, h1) + 0.5 * np.einsum("ijkl,ikjl->", sa_2pdm_chem, G) + nuc_repulsion)
        )

        iter_end = time.perf_counter()
        iter_time = iter_end - iter_start

        with open("iteration_pno_dmrg_oo.dat", "a") as f:
            f.write(f"{reported_distance:.3f} {iter} {iter_time:.2f} " + " ".join(f"{(x + nuc_repulsion):.15f}" for x in e_roots) + "\n")

        iteration_results.append(
            {"distance": reported_distance, "iteration": iter, "iteration_time": iter_time, "energies": (e_roots + nuc_repulsion)}
        )

    with open("results_pno_dmrg_oo.dat", "a") as f:
        f.write(f"{reported_distance:.3f} " + " ".join(f"{(x + nuc_repulsion):.15f}" for x in e_roots) + "\n")

    dist_end = time.perf_counter()
    dist_time = dist_end - dist_start
    print(f"Distance {reported_distance:.3f} took {dist_time:.2f} s")
    with open("distance_times_pno_dmrg_oo.dat", "a") as f:
        f.write(f"{reported_distance:.3f} {dist_time:.2f}\n")

    del integrals
    del opti
    del madpno
    del world
