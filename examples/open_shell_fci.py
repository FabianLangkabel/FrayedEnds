from time import time
import subprocess as sp
import shutil
import numpy as np
from pyscf import fci

import frayedends

def compute(world, distance):
    true_start = time()

    # CN geometry (bond distance ~2.67 Angstrom)
    geom = f"C 0.0 0.0 0.0\nN 0.0 0.0 {distance}"  # geometry in Angstrom
    molgeom = frayedends.MolecularGeometry(geom, units="angstrom")

    n_orbitals = 10  # 2 core + 8 active
    n_core_orbitals = 2
    n_act_orbitals = 8
    n_act_electrons = (5, 4)

    print(f"Active space electrons: {n_act_electrons}")


    nwchem = shutil.which("nwchem")
    nwchem_input = ("""
title "CN UHF starting orbitals"

charge 0
geometry units angstrom noautoz noautosym nocenter
""" + geom + """
end

basis spherical
    * library STO-3G
end

scf
    uhf
    doublet
    maxiter 200
end

task scf energy
"""
    )
    with open("nwchem", "w") as f:
        f.write(nwchem_input)
    programm = sp.call(
        f"{nwchem} nwchem",
        stdout=open("nwchem.out", "w"),
        stderr=open("nwchem_err.log", "w"),
        shell=True,
    )

    converter = frayedends.NWChem_Converter(world)
    converter.read_nwchem_file("nwchem")
    nw_orbitals = converter.get_mos()
    integrals = frayedends.Integrals(world)
    orbitals = integrals.orthonormalize(orbitals=nw_orbitals)

    # Get nuclear potential and repulsion energy
    nuc_repulsion = molgeom.get_nuclear_repulsion()
    Vnuc = molgeom.get_vnuc(world)

    # Split into core and active orbitals
    core = [orbitals[:n_core_orbitals], orbitals[:n_core_orbitals]]
    active = [orbitals[n_core_orbitals:], orbitals[n_core_orbitals:]]


    integrals = frayedends.Integrals_open_shell(world)

    current_energy = 0.0
    # SCF-like loop with orbital refinement and core refinement
    for iteration in range(20):
        # Get initial effective Hamiltonian
        c, h1, g2 = integrals.compute_effective_hamiltonian(core[0], core[1], active[0], active[1], Vnuc, nuc_repulsion)
        g2[0] = g2[0].transpose(0, 2, 1, 3)
        g2[1] = g2[1].transpose(0, 2, 1, 3)
        g2[2] = g2[2].transpose(0, 2, 1, 3)

        # FCI calculation on active space
        e, fcivec = fci.direct_uhf.kernel(h1, g2, n_act_orbitals, n_act_electrons)  # Computes the energy and the FCI vector
        rdm1, rdm2 = fci.direct_uhf.make_rdm12s(fcivec, n_act_orbitals, n_act_electrons)
        rdm2 = np.swapaxes(rdm2, 1, 2)
        rdm_2_phys_aa = rdm2[0].transpose(0, 2, 1, 3)
        rdm_2_phys_ab = rdm2[1].transpose(0, 2, 1, 3)
        rdm_2_phys_bb = rdm2[2].transpose(0, 2, 1, 3)

        print(f"Iteration {iteration} - FCI energy: {e + c:+2.10f} and core energy: {c:+2.10f}")

        if abs(e + c - current_energy) < 1e-5:
            print("Converged!")
            converged = True
            break
        current_energy = e + c

        op_s = time()
        # Orbital refinement with core orbital refinement enabled
        opti = frayedends.OrbitalRefinement_open_shell(world, Vnuc, nuc_repulsion)
        core, active, converged = opti.refine_orbitals(
            orbitals=[core[0], core[1], active[0], active[1]],
            rdm1=rdm1,
            rdm2=[rdm_2_phys_aa, rdm_2_phys_ab, rdm_2_phys_bb],
            opt_thresh=0.0001,
            occ_thresh=0.0001,
            maxiter=1,
            refine_core=True,  # Enable core orbital refinement
            redirect_filename=f"compli2_madopt{iteration}.log",
        )
        op_e = time()
        print(f"Orbital refinement time: {op_e - op_s:.2f} seconds")


    orbitals = core[0] + active[0]
    for i in range(len(orbitals)):
        world.cube_plot(f"oscompLi2_ref{i}_plot", orbitals[i], molgeom, zoom=5, datapoints=101)
        orbitals[i].save_to_file(f"oscompLi2_ref{i}")

    true_end = time()
    print(f"Total time: {true_end - true_start:.2f} seconds")

world = frayedends.MadWorld(ndims=3, thresh=1e-5)
energies = []
for distance in [1.75]:
    print(f"\n\nComputing for distance: {distance} Angstrom")
    compute(world, distance)
print(energies)

frayedends.cleanup(globals())
