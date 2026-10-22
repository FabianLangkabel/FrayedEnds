import numpy as np
import subprocess as sp
from time import time
import frayedends as fe
from pyscf import fci
from pyscf.lib import logger


def ch4_madopt(bond_length, world):
    start = time()
    geom = f"""C 0.0000 0.0000 0.0000
H {bond_length} {bond_length} {bond_length}
H {bond_length} -{bond_length} -{bond_length}
H -{bond_length} {bond_length} -{bond_length}
H -{bond_length} -{bond_length} {bond_length}"""

    # this returns the raw basis functions of the according quantum chemistry basis set in MRA form
    nwchem_start = time()
    
    basisset = "sto-3g"
   
    nwchem_input = (
           """
title "molecule"
memory stack 1500 mb heap 100 mb global 1400 mb
charge 0  
geometry units angstrom noautosym nocenter"""
        + "\n"
        + geom
        + """
end
basis  
* library """
        + basisset
        + """
end
scf  
maxiter 200
end   
task scf  """
    )
   
    with open("nwchem", "w") as f:
        f.write(nwchem_input)
    programm = sp.call(
        "/Users/timo/miniforge3/envs/fe_test/bin/nwchem nwchem",
        stdout=open("nwchem.out", "w"),
        stderr=open("nwchem_err.log", "w"),
        shell=True,
    )
   
    converter = fe.NWChem_Converter(world)
    converter.read_nwchem_file("nwchem")
    orbitals = converter.get_mos()
    

    nwchem_end = time()
       
    #pno = fe.MadPNO(world, geom, units="a", n_orbitals=9)
    #orbitals = pno.get_orbitals()
    #atombas=fe.AtomicBasisProjector(world, geom, units="a", basisset="sto-3g")
    #orbitals = atombas.get_orbitals()
     
    integrator = fe.Integrals(world)
    orbitals = integrator.orthonormalize(orbitals)

    for i in range(len(orbitals)):
        world.cube_plot(f"aorbital_{i}.dat", orbitals[i], fe.MolecularGeometry(geom, units="a"), zoom=5.0, datapoints=101)
    

    #for i in range(len(orbitals)):
    #    world.cube_plot(f"pnorbital_{i}.dat", orbitals[i], fe.MolecularGeometry(geom, units="a"), zoom=5.0, datapoints=101)
    molgeom = fe.MolecularGeometry(geom, units="a")
    Vnuc = molgeom.get_vnuc(world)
    nuclear_repulsion = molgeom.get_nuclear_repulsion()
    n_act_electrons = molgeom.n_electrons - 2
    n_core_orbs = 1
    n_orbs = len(orbitals)
    print(f"Solving CH4 with geometry:\n{geom}\nActive orbitals: {n_orbs - n_core_orbs}, Core orbitals: {n_core_orbs}")
    n_act_orbs = n_orbs - n_core_orbs

    core = orbitals[:n_core_orbs]
    active = orbitals[n_core_orbs:]

    c, h1, g2 = integrator.compute_effective_hamiltonian(core, active, Vnuc, nuclear_repulsion, g_ordering="chem")

    current_energy = 0.0
    converged = False
    for iteration in range(10):
        print("\n")
        fci_start = time()
        # FCI calculation
        e, fcivec = fci.direct_spin1.kernel(h1, g2, n_act_orbs, n_act_electrons)  # Computes the energy and the FCI vector
        rdm1, rdm2 = fci.direct_spin1.make_rdm12(
            fcivec, n_act_orbs, n_act_electrons
        )  # Computes the 1- and 2- body reduced density matrices
        rdm2 = np.swapaxes(rdm2, 1, 2)
        
        fci_end = time()
        print(f"FCI calculation time: {fci_end - fci_start:.2f} seconds")
        print("iteration {} energy {:+2.10f} and core energy {:+2.10f}".format(iteration, e + c, c))

        #if abs(e + c - current_energy) < 1e-6:
        #    print("Converged!")
        #    converged = True
        #    break
        current_energy = e + c

        opti_start = time()
        opti = fe.OrbitalRefinement(
            world, Vnuc, nuclear_repulsion
        )
        if iteration == 0:
            print(opti.get_numerical_parameters())
        core, active, converged = opti.refine_orbitals(
            orbitals=[core, active],
            rdm1=rdm1,
            rdm2=rdm2,
            opt_thresh=0.0001,
            occ_thresh=0.0001,
            redirect_filename=f"symm_lm_ch4{iteration}_mos_as_symm.log",
            maxiter = 1,
            refine_core=True
        )

        c, h1, g2 = opti.get_effective_hamiltonian(g_ordering="chem")
        opti_end = time()
        print(f"Orbital refinement time: {opti_end - opti_start:.2f} seconds")


    end = time()
    print("Total time: ", end - start)
    orbitals = core + active
    for i in range(len(orbitals)):
        world.cube_plot(f"ref_orbital_{i}.dat", orbitals[i], fe.MolecularGeometry(geom, units="a"), zoom=5.0, datapoints=101)
        

    return e + c, converged



world = fe.MadWorld(ndims=3, thresh=1e-7)

energies = []
converged = []
true_start = time()
for i in [0.6 + 0.05*j for j in range(1)]:
    res = ch4_madopt(i, world)
    energies.append(res[0])
    converged.append(res[1])

true_end = time()
print("Total time for all bond lengths: ", true_end - true_start)
fe.cleanup(globals())
