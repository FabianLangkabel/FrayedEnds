import subprocess as sp
import time

import numpy as np
from pyscf import fci

import frayedends as fe

molecule_name = "h4"
n_elec = 4 
number_roots = 3
iterations = 20
box_size = 50.0
wavelet_order = 7
madness_thresh = 1.0e-4
basisset = "6-31g"
econv = 1.0e-4 

geometry_mode = "equidistant"  # "equidistant" or "h2_pair"
print(f"Geometry mode: {geometry_mode}")

if geometry_mode == "equidistant":
    distance = np.arange(2.5, 0.45, -0.05).tolist()  # for linear H4 molecule with equidistant spacing d
elif geometry_mode == "h2_pair":
    distance = np.arange(1.5, 0.2, -0.02).tolist()  # for H2 pair getting closer
else:
    raise ValueError("Invalid geometry mode selected.")

output_file = f"{molecule_name}_{geometry_mode}_nwchem_fci_oo.dat"
 
with open(output_file, "w") as f:
    f.write("distance iteration iteration_time_s " + " ".join(f"energy_{i}" for i in range(number_roots)) + " energy_SA\n")

total_start = time.perf_counter()

for d in distance:
    dist_start = time.perf_counter()
    reported_distance = 2 * d if geometry_mode == "h2_pair" else d

    if geometry_mode == "equidistant":  # equidistant linear H4 molecule
        z = [-1.5 * d, -0.5 * d, 0.5 * d, 1.5 * d]
    else:  # two H2 molecules (bond length 2.55) getting closer
        z = [-d - 2.55, -d, d, d + 2.55]
    geom = "\n".join(f"    H 0.0 0.0 {zi}" for zi in z)

    nwchem_input = f"""
    title "molecule"
    memory stack 1500 mb heap 100 mb global 1400 mb
    charge 0
    geometry units angstrom noautosym nocenter
    {geom}
    end
    basis
    * library {basisset}
    end
    scf
    maxiter 200
    end
    task scf
    """

    with open("nwchem", "w") as f:
        f.write(nwchem_input)
    programm = sp.call(
        "/opt/anaconda3/envs/frayedends/bin/nwchem nwchem",
        stdout=open("nwchem.out", "w"),
        stderr=open("nwchem_err.log", "w"),
        shell=True,
    )

    world = fe.MadWorld(ndims=3, L=box_size, k=wavelet_order, thresh=madness_thresh)

    converter = fe.NWChem_Converter(world)
    converter.read_nwchem_file("nwchem")
    orbs = converter.get_mos()
    Vnuc = converter.get_Vnuc()
    nuclear_repulsion_energy = converter.get_nuclear_repulsion_energy()

    # for i in range(n_orbitals):
    #    world.line_plot(f"es_orb{i}.dat", orbs[i], axis="z", datapoints=2001)  # Plots guess orbitals

    integrals = fe.Integrals(world)
    G = integrals.compute_two_body_integrals(orbs, ordering="chem").elems
    T = integrals.compute_kinetic_integrals(orbs)
    V = integrals.compute_potential_integrals(orbs, Vnuc)
    h1 = T + V

    n_orbitals = len(orbs)

    cisolver = fci.direct_spin1.FCI()
    cisolver = fci.addons.fix_spin_(cisolver, ss=0)
    e, fcivec = cisolver.kernel(h1, G, n_orbitals, n_elec, nroots=number_roots)
    energies = e + nuclear_repulsion_energy
    
    rdms = [cisolver.make_rdm12(c, n_orbitals, n_elec) for c in fcivec]
    sa_1pdm = np.mean([r[0] for r in rdms], axis=0)  # Compute the state average 1-body rdm
    sa_2pdm = np.mean([r[1] for r in rdms], axis=0)  # Compute the state average 2-body rdm
    
    # State-averaged energy from SA-RDMs
    e_avg = (np.einsum("ij,ij->", sa_1pdm, h1)
                + 0.5 * np.einsum("ijkl,ijkl->", sa_2pdm, G)
                + nuclear_repulsion_energy)

    sa_2pdm_phys = sa_2pdm.swapaxes(1, 2)  # Physics Notation

    with open(output_file, "a") as f:
        f.write(f"{reported_distance:.3f} 0 0.00 " + " ".join(f"{x:.15f}" for x in energies) + f" {e_avg:.15f}\n")
    
    print(f"\nDistance: {reported_distance:.3f}")
    print(f"{'Iter':<6}{'Energies':<{14 * number_roots}}{'E_SA':<16}dE_SA")
    print(f"{0:<6}{''.join(f'{e:<14.8f}' for e in energies)}{e_avg:<16.8f}-")
    
    current = e_avg
    converged = False

    for iter in range(iterations):
        iter_start = time.perf_counter()

        # Orbital optimization
        opti = fe.OrbitalRefinement(world, Vnuc, nuclear_repulsion_energy)
        orbs = opti.get_orbitals(orbitals=orbs, rdm1=sa_1pdm, rdm2=sa_2pdm_phys, opt_thresh=0.001, occ_thresh=0.001)  # Optimizes the orbitals and returns the new ones

        # for i in range(n_orbitals):
        #    world.line_plot(f"orb{i}.dat", orbs[i], axis="z", datapoints=2001)  # Plots the optimized orbitals

        G = integrals.compute_two_body_integrals(orbs, ordering="chem").elems
        T = integrals.compute_kinetic_integrals(orbs)
        V = integrals.compute_potential_integrals(orbs, Vnuc)
        h1 = T + V
    
        n_orbitals = len(orbs)
    
        e, fcivec = cisolver.kernel(h1, G, n_orbitals, n_elec, nroots=number_roots)
        energies = e + nuclear_repulsion_energy
        
        rdms = [cisolver.make_rdm12(c, n_orbitals, n_elec) for c in fcivec]
        sa_1pdm = np.mean([r[0] for r in rdms], axis=0)  # Compute the state average 1-body rdm
        sa_2pdm = np.mean([r[1] for r in rdms], axis=0)  # Compute the state average 2-body rdm
        
        # State-averaged energy from SA-RDMs
        e_avg = (np.einsum("ij,ij->", sa_1pdm, h1)
                    + 0.5 * np.einsum("ijkl,ijkl->", sa_2pdm, G)
                    + nuclear_repulsion_energy)
    
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
    
    del converter
    del integrals
    del opti
    del world
    
with open(output_file, "a") as f:
    f.write(f"# total time = {time.perf_counter() - total_start:.2f} s\n")
    
print(f"\nTotal time: {time.perf_counter() - total_start:.1f} s")
    
fe.cleanup(globals())