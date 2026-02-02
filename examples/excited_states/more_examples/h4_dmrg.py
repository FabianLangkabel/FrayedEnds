import subprocess as sp
import frayedends as fe
import time
import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

molecule_name = "h4"
box_size = 50.0
wavelet_order = 7
madness_thresh = 0.0001
basisset = '6-31g'
n_elec = 4
number_roots = 3

iteration_results = []

geometry_mode = "equidistant"  # "equidistant" or "h2_pair"
print(f"Geometry mode: {geometry_mode}")

if geometry_mode == "equidistant": # linear H4 molecule with equidistant spacing d
    distance = np.arange(2.5, 0.45, -0.05).tolist()
elif geometry_mode == "h2_pair": # for H2 pair getting closer
    distance = np.arange(1.5, 0.2, -0.03).tolist()
else:
    raise ValueError("geometry_mode must be 'equidistant' or 'h2_pair'")

with open("results_nwchem_dmrg.dat", "w") as f:
    header = "distance " + " ".join(f"energy_{i}" for i in range(number_roots))
    f.write(header + "\n")

total_start = time.perf_counter()
for d in distance:
    dist_start = time.perf_counter()
    reported_distance = 2 * d if geometry_mode == "h2_pair" else d

    if geometry_mode == "equidistant":
        nwchem_input = '''
        title "molecule"
        memory stack 1500 mb heap 100 mb global 1400 mb
        charge 0  
        geometry units angstrom noautosym nocenter
            H 0.0 0.0 ''' + (-d-d/2).__str__() + '''
            H 0.0 0.0 ''' + (-d/2).__str__() + '''
            H 0.0 0.0 ''' + (d/2).__str__() + '''
            H 0.0 0.0 ''' + (d+d/2).__str__() + '''
        end
        basis  
          * library ''' + basisset + '''
        end
        scf  
         maxiter 200
        end   
        task scf  
        '''
    elif geometry_mode == "h2_pair":
        nwchem_input = '''
        title "molecule"
        memory stack 1500 mb heap 100 mb global 1400 mb
        charge 0  
        geometry units angstrom noautosym nocenter
            H 0.0 0.0 ''' + (-d - 2.55).__str__() + '''
            H 0.0 0.0 ''' + (-d).__str__() + '''
            H 0.0 0.0 ''' + d.__str__() + '''
            H 0.0 0.0 ''' + (d + 2.55).__str__() + '''
        end
        basis  
          * library ''' + basisset + '''
        end
        scf  
         maxiter 200
        end   
        task scf  
        '''

    with open("nwchem", "w") as f:
        f.write(nwchem_input)
    programm = sp.call("/opt/anaconda3/envs/frayedends/bin/nwchem nwchem", stdout=open('nwchem.out', 'w'), stderr=open('nwchem_err.log', 'w'), shell = True)

    world = fe.MadWorld3D(L=box_size, k=wavelet_order, thresh=madness_thresh)

    converter = fe.NWChem_Converter(world)
    converter.read_nwchem_file("nwchem")
    orbs = converter.get_mos()
    Vnuc = converter.get_Vnuc()
    nuclear_repulsion_energy = converter.get_nuclear_repulsion_energy()

    for i in range(len(orbs)):
        orbs[i].type="active"

    #for i in range(len(orbs)):
    #   world.line_plot(f"initial_orb{i}_d{d}.dat", orbs[i], axis="z", datapoints=2001)

    integrals = fe.Integrals3D(world)
    G = integrals.compute_two_body_integrals(orbs, ordering="chem").elems
    T = integrals.compute_kinetic_integrals(orbs)
    V = integrals.compute_potential_integrals(orbs, Vnuc)
    S = integrals.compute_overlap_integrals(orbs)

    n_orbitals = len(orbs)

    driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SU2, n_threads=8)
    driver.initialize_system(n_sites=n_orbitals, n_elec=n_elec, spin=0)
    mpo = driver.get_qc_mpo(h1e=T + V, g2e=G, ecore=nuclear_repulsion_energy, iprint=0)
    ket = driver.get_random_mps(tag="KET", bond_dim=100, nroots=number_roots)
    energies = driver.dmrg(mpo, ket, n_sweeps=10, bond_dims=[100], noises=[1e-5] * 4 + [0], thrds=[1e-10] * 8, iprint=1)

    with open("results_nwchem_dmrg.dat", "a") as f:
        f.write(f"{reported_distance:.3f} " + " ".join(f"{x:.15f}" for x in energies) + "\n")

    dist_end = time.perf_counter()
    dist_time = dist_end - dist_start
    print(f"Distance {reported_distance:.3f} took {dist_time:.2f} s")
    with open("distance_times_nwchem_dmrg.dat", "a") as f:
        f.write(f"{reported_distance:.3f} {dist_time:.2f}\n")

    del converter
    del integrals
    del world
