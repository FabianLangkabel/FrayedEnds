import numpy as np
import frayedends as fe
from pyscf import fci, gto
from pyscf.geomopt import geometric_solver, as_pyscf_method
from time import time

"""
This code is used to optimize the geometry of molecules using geomeTRIC. 
We calculate the energy of a given molecular geometry using orbital refinement and FCI (energy_and_gradient function). 
This function calculates an initial orbital basis using pair natural orbitals and then starts an iterative algorithm:
1. Compute one- and two-body integrals using the current orbital basis
2. Use FCI to compute the ground state energy and reduced density matrices
3. Refine the orbital basis using Green's operators
4. Repeat until energy convergence is reached
After convergence the energy gradient w.r.t. nuclear coordinates is computed. Since the orbitals and the many-body wave function are variational, 
we can use the Hellmann-Feynman theorem when computing the energy gradient.
The energy_and_gradient function is wrapped to be compatible with geomeTRIC using the geomeTRIC interface by pyscf.
"""

world = fe.MadWorld3D(thresh=1e-6) # setting up the numerical environment for the MRA calculations. thresh is numerical precision of function representation.

def energy_and_gradient(world: fe.MadWorld3D, molgeom: fe.MolecularGeometry, n_orbitals: int, maxiter_whole_alg=30, maxiter_orbopt=1, e_convergence=1e-6):
    true_start = time()
    geom_str= molgeom.get_geometry_string() # extracting the geometry and units as a string

    pno_start = time()
    # we calculate initial guess orbitals for the orbital refinement using pair natural orbitals (PNO) 
    # for more details on this method see: J.S. Kottmann, F.A. Bischoff, E.F. Valeev, J. Chem. Phys. 152, 2020
    madpno = fe.MadPNO(world, geom_str[0], units=geom_str[1], n_orbitals=n_orbitals)
    orbitals = madpno.get_orbitals() # initial guess orbitals as MRA functions, orb.type determines whether the orbital is 'active' or 'frozen_occ' (in this case all active)
    for orb in orbitals:
        orb.type = "active" # in this example we do not freeze any orbitals, so we set all orbitals to active
    pno_end = time()
    print("pno time:", pno_end-pno_start)

    world.set_function_defaults() # pno code might change some defaults of the numerical environment (world), we reset them to original values here
    nuclear_repulsion = molgeom.get_nuclear_repulsion() # nuclear repulsion energy
    Vnuc = molgeom.get_vnuc(world) # nuclear potential as MRA function

    integrals = fe.Integrals3D(world) # class to compute different types of integrals
    orbitals = integrals.orthonormalize(orbitals=orbitals) # orthonormalize pnos

    current_energy = 0.0
    for iteration in range(maxiter_whole_alg):
        # using the guess orbitals we compute the one- and two-body integrals 
        # (h=<i|-\Delta/2+Vnuc|j> and g=(ij|kl) (ordering of g depends on chosen convention))
        G = integrals.compute_two_body_integrals(orbitals, ordering="chem") # pyscf fci code assumes chem ordering of g tensor
        T = integrals.compute_kinetic_integrals(orbitals)
        V = integrals.compute_potential_integrals(orbitals, Vnuc)
        h = T + V
        g = G

        # with these integrals we can use fci to determine GS energy, one- and two-body rdms
        fci_start=time()
        e, fcivec = fci.direct_spin1.kernel(
            h, g.elems, n_orbitals, molgeom.n_electrons
        )  # Computes the energy and the FCI vector
        rdm1, rdm2 = fci.direct_spin1.make_rdm12(
            fcivec, n_orbitals, molgeom.n_electrons
        )  # Computes the 1- and 2- body reduced density matrices
        rdm2 = np.swapaxes(rdm2, 1, 2) # swapping axes to match convention used in orbital refinement code
        fci_end = time()
        print("fci time:", fci_end-fci_start)

        print("iteration {} energy {:+2.7f}".format(iteration, e + nuclear_repulsion))
        if abs(current_energy - (e + nuclear_repulsion)) < 1e-6: 
            break # stops the algorithm if energy is converged
        current_energy = e + nuclear_repulsion

        # using the rdms we now refine the orbitals, for more details see Theory at https://github.com/FabianLangkabel/FrayedEnds
        # or arXiv:2410.19116 
        ref_start = time()
        orb_refiner = fe.Optimization3D(world, Vnuc, nuclear_repulsion)
        orbitals = orb_refiner.get_orbitals(
            orbitals=orbitals,
            rdm1=rdm1,
            rdm2=rdm2,
            maxiter=maxiter_orbopt, # maximum number of iterations the refinement algorithm does 
        ) # orbitals are now the refined orbitals
        for orb in orbitals:
            print(orb.type)
        ref_end = time()
        print("orb ref time:", ref_end-ref_start)
        # with the refined orbitals obtained the algorithm loops back to recompute integrals -> fci -> orbital refinement until convergence
        
    # as soon as our energy is converged we can compute the energy gradient w.r.t. nuclear coordinates
    grad = molgeom.compute_energy_gradient(world, orbitals, rdm1)
    true_end = time()
    print("total time:", true_end-true_start)
    return current_energy, np.array(grad)

def f(pyscf_mol):
    molgeom = fe.MolecularGeometry.from_pyscf_mol(pyscf_mol, units="bohr")
    e, g = energy_and_gradient(world, molgeom, n_orbitals=9)
    return e, g

geom_opt_start = time()
pyscf_mol = gto.M(atom="H 0.0 0.0 -2.5\nBe 0.0 0.0 0.0\nH 0.0 0.0 2.5", unit="bohr") #initial guess molecule
print(pyscf_mol.atom_coords())

fake_method = as_pyscf_method(pyscf_mol, f) # wrapping energy and gradient function to be compatible with geomeTRIC

new_mol = geometric_solver.optimize(fake_method) # geometry optimization with geomeTRIC
geom_opt_end = time()
print('New geometry (bohr)')
print(new_mol.atom_coords())
print("new geometry a")
print(new_mol.tostring())
print("Geometry optimization time:", geom_opt_end - geom_opt_start)
