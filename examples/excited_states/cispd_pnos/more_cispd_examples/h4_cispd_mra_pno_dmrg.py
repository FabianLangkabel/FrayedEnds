import time

import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

import frayedends as fe

molecule_name = "h4"
n_elec = 4
number_roots = 3
iterations = 20
box_size = 50.0
wavelet_order = 7
madness_thresh = 1.0e-6
econv = 1.0e-4

geometry_mode = "h2_pair"  # "equidistant" or "h2_pair"
print(f"Geometry mode: {geometry_mode}")

if geometry_mode == "equidistant":  # linear H4 molecule with equidistant spacing d
    distance = np.arange(2.5, 0.45, -0.05).tolist()
elif geometry_mode == "h2_pair":  # for H2 pair getting closer
    distance = np.arange(1.5, 0.2, -0.03).tolist()
else:
    raise ValueError("geometry_mode must be 'equidistant' or 'h2_pair'")

# Create .dat file for the results
output_file = f"{molecule_name}_{geometry_mode}_cispd_pno_dmrg_oo.dat"
 
with open(output_file, "w") as f:
    f.write("distance iteration iteration_time_s " + " ".join(f"energy_{i}" for i in range(number_roots)) + " energy_SA\n")
 
total_start = time.perf_counter()

for d in distance:
    dist_start = time.perf_counter()
    reported_distance = 2 * d if geometry_mode == "h2_pair" else d

    # Define the H4 geometry for this distance
    if geometry_mode == "equidistant":  # equidistant linear H4 molecule
        z = [-1.5 * d, -0.5 * d, 0.5 * d, 1.5 * d]
    else:  # two H2 molecules getting closer
        z = [-d - 1.5, -d, d, d + 1.5]
    geom = "".join(f"H 0.0 0.0 {zi}\n" for zi in z)

    world = fe.MadWorld(ndims=3, L=box_size, k=wavelet_order, thresh=madness_thresh)
    integrals = fe.Integrals(world)

    pno_start = time.perf_counter()
    madpno = fe.MadPNO(world, geom, units="angstrom", n_orbitals=4)  # 2 HF + 2 MP2-PNOs
    pno_time = time.perf_counter() - pno_start
    print("Generating PNOs took %.2f seconds" % pno_time)

    gs_orbs = madpno.get_orbitals()

    # for i in range(len(gs_orbs)):
    # world.cube_plot(f"gs_orb{i}", gs_orbs[i], molecule, zoom=4.0)

    cis_start = time.perf_counter()
    cis_orbs = madpno.compute_cis(n_excitation=2)  # Compute CIS for 2 excitations (1st and 2nd excited states)
    cis_orbs = integrals.project_out(gs_orbs, cis_orbs)
    cis_orbs = integrals.orthonormalize(cis_orbs)
    cis_time =time.perf_counter() - cis_start
    print("Generating CIS took %.2f seconds" % cis_time)

    # for i in range(len(cis_orbs)):
    #     world.cube_plot(f"cis_orb{i}", cis_orbs[i], molecule, zoom=4.0)

    cispd_start = time.perf_counter()
    cispd_orbs = madpno.compute_cispd(n_orbitals=4)  # 2 CIS X Functions + 2 CIS(D)-PNOS
    cispd_orbs = integrals.project_out(gs_orbs + cis_orbs, cispd_orbs)
    cispd_time = time.perf_counter() - cispd_start
    print("Generating CISPD PNOs took %.2f seconds" % cispd_time)

    # for i in range(len(cispd_orbs)):
    #     world.cube_plot(f"cispd_orb{i}", cispd_orbs[i], molecule, zoom=4.0)

    nuc_repulsion = madpno.get_nuclear_repulsion()
    Vnuc = madpno.get_nuclear_potential()

    orbs = gs_orbs + cis_orbs + cispd_orbs
    orbs = integrals.orthonormalize(orbitals=orbs)

    # for i in range(len(orbs)):
    #    world.line_plot(f"orb{i}_d{d}.dat", orbs[i], axis="z", datapoints=2001)

    """
    Calculate initial integrals
    """
    G = integrals.compute_two_body_integrals(orbs, ordering="chem").elems
    T = integrals.compute_kinetic_integrals(orbs)
    V = integrals.compute_potential_integrals(orbs, Vnuc)
    h1 = T + V

    """
    Performe SA DMRG calculation and extract rdms
    """

    n_orbitals = len(orbs)

    driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SU2, n_threads=8)
    driver.initialize_system(n_sites=n_orbitals, n_elec=n_elec, spin=0)

    idx = driver.orbital_reordering(h1, G)
    h1_new = h1[idx][:, idx]
    g2_new = G[idx][:, idx][:, :, idx][:, :, :, idx]

    mpo = driver.get_qc_mpo(h1e=h1_new, g2e=g2_new, ecore=nuc_repulsion, iprint=0)
    ket = driver.get_random_mps(tag="KET", bond_dim=100, nroots=number_roots)
    energies = driver.dmrg(mpo, ket, n_sweeps=10, bond_dims=[100], noises=[1e-5] * 4 + [0], thrds=[1e-10] * 8, iprint=0)

    """
    Extract rdms
    """
    
    kets = [driver.split_mps(ket, ir, tag="KET-%d" % ir) for ir in range(ket.nroots)]
    sa_1pdm = np.mean([driver.get_1pdm(k) for k in kets], axis=0)
    sa_2pdm = np.mean([driver.get_2pdm(k) for k in kets], axis=0).transpose(0, 3, 1, 2)

    # State-averaged energy from SA-RDMs
    e_avg = (np.einsum("ij,ij->", sa_1pdm, h1_new)
            + 0.5 * np.einsum("ijkl,ijkl->", sa_2pdm, g2_new)
            + nuc_repulsion)

    idx_back = np.argsort(idx)
    sa_1pdm = sa_1pdm[idx_back][:, idx_back]
    sa_2pdm = sa_2pdm[idx_back][:, idx_back][:, :, idx_back][:, :, :, idx_back]
    sa_2pdm_phys = sa_2pdm.swapaxes(1, 2)  # Physics Notation

    with open(output_file, "a") as f:
        f.write(f"# distance {reported_distance:.3f}: orbitals {len(gs_orbs)} GS + {len(cis_orbs)} CIS + "
                f"{len(cispd_orbs)} CIS(D) -> {n_orbitals}, generation time PNO {pno_time:.2f} s, "
                f"CIS {cis_time:.2f} s, CIS(D) {cispd_time:.2f} s\n")
        f.write(f"{reported_distance:.3f} 0 0.00 " + " ".join(f"{x:.15f}" for x in energies) + f" {e_avg:.15f}\n")

    print(f"\nDistance: {reported_distance:.3f}")
    print(f"{'Iter':<6}{'Energies':<{14 * number_roots}}{'E_SA':<16}dE_SA")
    print(f"{0:<6}{''.join(f'{e:<14.8f}' for e in energies)}{e_avg:<16.8f}-")

    current = e_avg
    converged = False

    for iter in range(iterations):
        iter_start = time.perf_counter()

        """
        Refine orbitals
        """
        opti = fe.OrbitalRefinement(world, Vnuc, nuc_repulsion)
        orbs = opti.get_orbitals(orbitals=orbs, rdm1=sa_1pdm, rdm2=sa_2pdm_phys, opt_thresh=0.001, occ_thresh=0.001)
        
        # for i in range(len(orbs)):
        #    world.line_plot(f"orb{i}_d{d}.dat", orbs[i], axis="z", datapoints=2001))

        """
        DMRG with refined orbitals
        """
        G = integrals.compute_two_body_integrals(orbs, ordering="chem").elems
        T = integrals.compute_kinetic_integrals(orbs)
        V = integrals.compute_potential_integrals(orbs, Vnuc)
        h1 = T + V

        driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SU2, n_threads=8)
        driver.initialize_system(n_sites=n_orbitals, n_elec=n_elec, spin=0)

        idx = driver.orbital_reordering(h1, G)
        h1_new = h1[idx][:, idx]
        g2_new = G[idx][:, idx][:, :, idx][:, :, :, idx]

        mpo = driver.get_qc_mpo(h1e=h1_new, g2e=g2_new, ecore=nuc_repulsion, iprint=0)
        ket = driver.get_random_mps(tag="KET", bond_dim=100, nroots=number_roots)
        energies = driver.dmrg(mpo, ket, n_sweeps=10, bond_dims=[100], noises=[1e-5] * 4 + [0], thrds=[1e-10] * 8, iprint=0)
        
        kets = [driver.split_mps(ket, ir, tag="KET-%d" % ir) for ir in range(ket.nroots)]
        sa_1pdm = np.mean([driver.get_1pdm(k) for k in kets], axis=0)
        sa_2pdm = np.mean([driver.get_2pdm(k) for k in kets], axis=0).transpose(0, 3, 1, 2)
        
        # State-averaged energy from SA-RDMs
        e_avg = (np.einsum("ij,ij->", sa_1pdm, h1_new)
                + 0.5 * np.einsum("ijkl,ijkl->", sa_2pdm, g2_new)
                + nuc_repulsion)

        idx_back = np.argsort(idx)
        sa_1pdm = sa_1pdm[idx_back][:, idx_back]
        sa_2pdm = sa_2pdm[idx_back][:, idx_back][:, :, idx_back][:, :, :, idx_back]
        sa_2pdm_phys = sa_2pdm.swapaxes(1, 2)  # Physics Notation

        print(f"{iter + 1:<6}{''.join(f'{e:<14.8f}' for e in energies)}{e_avg:<16.8f}{e_avg - current:+.1e}")
         
        iter_time = time.perf_counter() - iter_start
        with open(output_file, "a") as f:
            f.write(f"{reported_distance:.3f} {iter + 1} {iter_time:.2f} " + " ".join(f"{x:.15f}" for x in energies) + f" {e_avg:.15f}\n")
    
        if abs(e_avg - current) < econv:
            converged = True
            print(f"Converged after {iter + 1} iterations (|dE_SA| < {econv:.0e})")
            break  # stop as soon as the state-averaged energy changes less than econv
        current = e_avg
    
    dist_time = time.perf_counter() - dist_start
    status = f"converged after {iter + 1} iterations" if converged else f"not converged after {iterations} iterations"
    print(f"Distance {reported_distance:.3f}: {status}, time = {dist_time:.1f} s")
    
    with open(output_file, "a") as f:
        f.write(f"# distance {reported_distance:.3f}: {status}, econv = {econv:.0e}, time = {dist_time:.2f} s\n")
    
    del integrals
    del opti
    del madpno
    del world
    
with open(output_file, "a") as f:
    f.write(f"# total time = {time.perf_counter() - total_start:.2f} s\n")
    
print(f"\nTotal time: {time.perf_counter() - total_start:.1f} s")
    
fe.cleanup(globals())