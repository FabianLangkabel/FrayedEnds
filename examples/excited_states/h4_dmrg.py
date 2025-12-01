import subprocess as sp
import frayedends as fe
import time
import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

# distance = np.arange(1.5, 0.2, -0.03).tolist() for H2 pair getting closer
distance = np.arange(2.5, 0.45, -0.05).tolist()
molecule_name = "h4"
box_size = 50.0
wavelet_order = 7
madness_thresh = 0.0001
basisset = '6-31g'
n_elec = 4
number_roots = 3

iteration_results = []

with open("results_nwchem_dmrg.dat", "w") as f:
    header = "distance " + " ".join(f"energy_{i}" for i in range(number_roots))
    f.write(header + "\n")

total_start = time.perf_counter()
for d in distance:
    dist_start = time.perf_counter()
    # linear H4 molecule with equidistant spacing d
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
    # two H2 molecules getting closer and closer to a H4 molecule
    # H 0.0 0.0 ''' + (-d - 2.55).__str__() + '''
    # H 0.0 0.0 ''' + (-d).__str__() + '''
    # H 0.0 0.0 ''' + d.__str__() + '''
    # H 0.0 0.0 ''' + (d + 2.55).__str__() + '''

    with open("nwchem", "w") as f:
        f.write(nwchem_input)
    programm = sp.call("/opt/anaconda3/envs/frayedends/bin/nwchem nwchem", stdout=open('nwchem.out', 'w'), stderr=open('nwchem_err.log', 'w'), shell = True)

    world = fe.MadWorld3D(L=box_size, k=wavelet_order, thresh=madness_thresh)

    converter = fe.NWChem_Converter(world)
    converter.read_nwchem_file("nwchem")
    orbs = converter.get_mos()
    Vnuc = converter.get_Vnuc()
    nuclear_repulsion_energy = converter.get_nuclear_repulsion_energy()
    del converter

    for i in range(len(orbs)):
        orbs[i].type="active"

    # for i in range(len(orbs)):
    #    world.line_plot(f"orb{i}_d{d}.dat", orbs[i])

    integrals = fe.Integrals3D(world)
    G = integrals.compute_two_body_integrals(orbs, ordering="chem").elems
    T = integrals.compute_kinetic_integrals(orbs)
    V = integrals.compute_potential_integrals(orbs, Vnuc)
    S = integrals.compute_overlap_integrals(orbs)

    ncas = len(orbs)

    driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SU2, n_threads=4)
    driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=0)
    mpo = driver.get_qc_mpo(h1e=T + V, g2e=G, ecore=nuclear_repulsion_energy, iprint=0)
    ket = driver.get_random_mps(tag="KET", bond_dim=100, nroots=number_roots)
    energies = driver.dmrg(mpo, ket, n_sweeps=10, bond_dims=[100], noises=[1e-5] * 4 + [0], thrds=[1e-10] * 8, iprint=1)
    print('State-averaged MPS energies = [%s]' % " ".join("%20.15f" % x for x in energies))

    kets = [driver.split_mps(ket, ir, tag="KET-%d" % ir) for ir in range(ket.nroots)]
    sa_1pdm = np.mean([driver.get_1pdm(k) for k in kets], axis=0)
    sa_2pdm = np.mean([driver.get_2pdm(k) for k in kets], axis=0).transpose(0, 3, 1, 2)
    print('Energy from SA-pdms = %20.15f' % (
                np.einsum('ij,ij->', sa_1pdm, T + V) + 0.5 * np.einsum('ijkl,ijkl->', sa_2pdm,
                                                                       G) + nuclear_repulsion_energy))
    sa_2pdm_phys = sa_2pdm.swapaxes(1, 2)  # Physics Notation

    np.savetxt("initial_energies.txt", energies)

    with open("results_nwchem_dmrg.dat", "a") as f:
        f.write(f"{d:.6f} " + " ".join(f"{x:.15f}" for x in energies) + "\n") # for H2 pair use 2*d

    dist_end = time.perf_counter()
    dist_time = dist_end - dist_start
    print(f"Distance {d:.6f} took {dist_time:.2f} s") # for H2 pair use 2*d
    with open("distance_times_nwchem_dmrg.dat", "a") as f:
        f.write(f"{d:.6f} {dist_time:.6f}\n") # for H2 pair use 2*d

    del integrals
    del world

total_end = time.perf_counter()
total_time = total_end - total_start
print(f"Total runtime: {total_time:.2f} s")
with open("total_time_nwchem_dmrg.dat", "w") as f:
    f.write(f"total_runtime_s {total_time:.6f}\n")