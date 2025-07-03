#This will become the master script which includes the whole workflow of spinorbital refinement. It contains all the necessary python functions and is linked to the OrbitalTranslator
# and SpinorbOpt programs. There are also Python functions who can create the json input files for the C++ programs.


from pyscf import gto, scf
import numpy as np
from pyblock2._pyscf.ao2mo import integrals as itg
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
import os
import json
import subprocess


#this is for deleting the temporary files so that each dmrg calculation can be run fresh
def delete_tmp_file(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check if it's a file (not a directory) and delete it
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")


#define here the input for the uhf calculation which includes the geometry in Bohr, the basisset, charge and spin for H3 linear
def define_H3lin_mole_object(number, basisset): 
    x = float(number) * -1
    y = float(number)
    mol = gto.Mole()
    mol.atom = [["H", 0, 0, 0], ["H", y, 0, 0], ["H", x, 0, 0]]
    mol.unit = 'Bohr'
    mol.basis = basisset
    mol.charge = 0
    mol.spin = 1
    mol.build()
    return mol

#define here the input for the uhf calculation which includes the geometry in Bohr, the basisset, charge and spin for H3 triangular
def define_H3tri_mole_object(number, basisset): 
    x = float(number)
    mol = gto.Mole()
    mol.atom = [["H", 0, 0, 0], ["H", 1.64406161, 0, 0], ["H", 0.822030805, x, 0]]
    mol.unit = 'Bohr'
    mol.basis = basisset
    mol.charge = 0
    mol.spin = 1
    mol.build()
    return mol

#define here the input for the uhf calculation which includes the geometry in Bohr, the basisset, charge and spin for He-H
def define_HeH_mole_object(number, basisset): 
    #number = number.replace("_", ".")
    x = float(number)
    print(x)
    mol = gto.Mole()
    mol.atom = [["He", 0, 0, 0], ["H", x, 0, 0]]
    mol.unit = 'Bohr'
    mol.basis = basisset
    mol.charge = 0
    mol.spin = 1
    mol.build()
    return mol

#run UHF and return the alpha and beta mo-coefficients for exporting and return the required info for dmrg
def calculate_alpha_beta_coeffs(mol):
    uhf=scf.UHF(mol)
    uhf.kernel()  
    alpha_coeff=uhf.mo_coeff[0]
    beta_coeff=uhf.mo_coeff[1]
    ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_uhf_integrals(uhf,
    ncore=0, ncas=None, g2e_symm=1)
    return alpha_coeff, beta_coeff, ncas, n_elec, spin, ecore, h1e, g2e, orb_sym

#export the alpha and beta coefficients of a certain bondlength in npy files, so that they can be read by OrbitalTranslator
def save_mocoeffs(alpha_coeff, beta_coeff, folderpath, x):
    #x = str(x)
    #x = x.replace(".", "")
    alpha_path = f"{folderpath}/{x}_alpha_coeffs.npy"
    beta_path = f"{folderpath}/{x}_beta_coeffs.npy"
    np.save(alpha_path, alpha_coeff)
    np.save(beta_path, beta_coeff)

#run dmrg calculation and return the 1-body and 2-body rdms
def run_dmrg_and_get_rdms(ncas, n_elec, spin, ecore, h1e, g2e, orb_sym):
    bond_dims = [50] * 10
    noises = [1e-4] * 5 + [1e-5] * 5 + [0]
    thrds = [1e-10] * 10
    driver = DMRGDriver(scratch="./tmp4", symm_type=SymmetryTypes.SZ, n_threads=4)
    driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym)
    mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=0)
    ket = driver.get_random_mps(tag="GS", bond_dim=50, nroots=1)
    energy = driver.dmrg(mpo, ket, n_sweeps=14, bond_dims=bond_dims, noises=noises,
    thrds=thrds, iprint=0)
    onerdm = driver.get_1pdm(ket)
    twordm = driver.get_2pdm(ket) 
    alpha_1rdm=onerdm[0]
    beta_1rdm=onerdm[1]
    aa_2rdm = twordm[0]
    ab_2rdm = twordm[1]
    bb_2rdm = twordm[2]
    print('DMRG energy: ', energy)
    return energy, alpha_1rdm, beta_1rdm, aa_2rdm, ab_2rdm, bb_2rdm

#export the 1-body and 2-body rdms of a certain bondlength in npy files, so that they can be read by SpinorbOpt
def save_rdms(alpha_1rdm, beta_1rdm, aa_2rdm, ab_2rdm, bb_2rdm, folderpath, x):
    #x = str(x)
    #x = x.replace(".", "")
    alpha_1rdm_path = f"{folderpath}/{x}_alpha_1rdm.npy"
    beta_1_rdm_path = f"{folderpath}/{x}_beta_1rdm.npy"
    alpha_alpha_2rdm_path = f"{folderpath}/{x}_alpha_alpha_2rdm.npy"
    alpha_beta_2_rdm_path = f"{folderpath}/{x}_alpha_beta_2rdm.npy"
    beta_beta_2rdm_path = f"{folderpath}/{x}_beta_beta_2rdm.npy"
    np.save(alpha_1rdm_path, alpha_1rdm)
    np.save(beta_1_rdm_path, beta_1rdm)
    np.save(alpha_alpha_2rdm_path, aa_2rdm)
    np.save(alpha_beta_2_rdm_path, ab_2rdm)
    np.save(beta_beta_2rdm_path, bb_2rdm)


#use this to make the geometry file for He-H
def make_geometry_file_HeH(directory, position):
    x = position.replace(".", "_")
    #position = float(position)
    filename = f"{directory}/{x}_geometry_HeH.mol"
    with open(filename, "w") as file:
        l1 = "geometry \n"
        l2 = "units Bohr \n"
        l3 = "no_orient 1 \n"
        l4 = "eprec 1e-6 \n"
        l5 = "He 0.00000000 0.00000000 0.00000000 \n"
        l6 = f"H {position} 0.00000000 0.00000000 \n"
        l7 = "end"
        file.writelines([l1, l2, l3, l4, l5, l6, l7])


#this function creates the json files with all the necessary information for the Orbital Translator programm; k and madness_thresh need to be adjusted manually
#currently set for HeH
def make_orbitaltranslator_json(directory, pos):
    #pos = pos.replace(".", "")
    alpha_path = f"{directory}/{pos}_alpha_coeffs.npy"
    beta_path = f"{directory}/{pos}_beta_coeffs.npy"
    dict = {
        "box_size": 50.0,
        "wavelet_order": 8,
        "madness_thresh": 0.00001,
        "optimization_thresh": 0.001,
        "NO_occupation_thresh": 0.001,
        "molecule_file": f"{directory}/{pos}_geometry_HeH.mol",
        "calpha_coeff_file": alpha_path,
        "cbeta_coeff_file": beta_path,
        "output_folder": directory,
    }
    dict_json = json.dumps(dict, indent=4)
    output_jsonfile = f"{directory}/{pos}_HeH_OrbitalTranslator.json"
    print("Save OrbitalTranslator Json file")
    with open(output_jsonfile, "w") as file:
        file.write(dict_json)
    return output_jsonfile
    

#this function creates the Json Files with all the information that the SpinorbitalOptimizer needs, one needs to modify k, madness_thresh and the geometry_file_path manually
#currently set for HeH
def make_spinorbopt_json(directory, pos, num_orbs): 
    #pos = pos.replace(".", "")
    dict_1 = {
        "box_size": 50.0,
        "wavelet_order": 8,
        "madness_thresh": 0.00001,
        "optimization_thresh": 0.001,
        "NO_occupation_thresh": 0.001,
        "molecule_file": f"{directory}/{pos}_geometry_HeH.mol",
        "output_folder": directory,
        "alpha_one_rdm_file": f"{directory}/{pos}_alpha_1rdm.npy",
        "beta_one_rdm_file": f"{directory}/{pos}_beta_1rdm.npy",
        "alpha_alpha_rdm_file": f"{directory}/{pos}_alpha_alpha_2rdm.npy",
        "alpha_beta_rdm_file": f"{directory}/{pos}_alpha_beta_2rdm.npy",
        "beta_beta_rdm_file": f"{directory}/{pos}_beta_beta_2rdm.npy",
    }
    alpha_entries=[]
    beta_entries=[]
    for i in np.arange(0,num_orbs, 1):
        alpha_filename = f"{directory}/alpha_orbital_{i}"
        beta_filename = f"{directory}/beta_orbital_{i}"
        orb_type = "active"
        alpha_idx = int(2*i)
        beta_idx =int(2*i+1)
        alpha_entry = (alpha_filename, orb_type, alpha_idx)
        beta_entry = (beta_filename, orb_type, beta_idx)
        alpha_entries.append(alpha_entry)
        beta_entries.append(beta_entry)
    alpha_dict = {
        "alpha_orbitals": [
            {
              "alpha_orbital_file_name": alpha_file,
              "alpha_orbital_type": alpha_orb_type,
              "alpha_active_space_index": alpha_idx,
            } for alpha_file, alpha_orb_type, alpha_idx in alpha_entries
        ]
    }
    beta_dict = {
        "beta_orbitals": [
            {
              "beta_orbital_file_name": beta_file,
              "beta_orbital_type": beta_orb_type,
              "beta_active_space_index": beta_idx,
            } for beta_file, beta_orb_type, beta_idx in beta_entries
        ]
    }

    combined_data = {**dict_1, **alpha_dict, **beta_dict}
    dict_json = json.dumps(combined_data, indent=2)
    output_jsonfile = f"{directory}/{pos}_HeH_SpinorbOpt.json"
    print("Save SpinorbOpt Json file")
    with open(output_jsonfile, "w") as file:
        file.write(dict_json)
    return output_jsonfile




#use os.mkdir("directory_name") to make a new directory for the new bondlength


#delete_tmp_file('/workspaces/MRA-OrbitalOptimization/automate_process/tmp2')
delete_tmp_file('/workspaces/MRA-OrbitalOptimization/automate_process/tmp4')

#new_pos = str(1.75743603) #the next lines are going to incorporated into the loop
#new_geom = f'H {new_pos} 0.00000000 0.00000000'
#old_pos = str(1.85192233)
#old_geom = f'H {old_pos} 0.00000000 0.00000000'

#H_pos = str(1.75743603)

#modify_geometry_file('/workspaces/MRA-OrbitalOptimization/geometry/H3_lin_min.mol', old_geom, new_geom)
#H3_lin_geom = define_H3lin_mole_object(1.75743603, 'sto-3g')

#H3_tri_geom = define_H3tri_mole_object(1.42379912, '6-31g')
#H3lin_alpha_coeffs, H3lin_beta_coeffs, ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = calculate_alpha_beta_coeffs(H3_lin_geom)

#H3tri_alpha_coeffs, H3tri_beta_coeffs, ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = calculate_alpha_beta_coeffs(H3_tri_geom)
#save_mocoeffs(H3lin_alpha_coeffs, H3lin_beta_coeffs, '/workspaces/MRA-OrbitalOptimization/coefficients/h3lin_sto3g_coeffs', 1.75743603)


#for i in np.arange(1.2, 7.0, 0.1):
#    atom_pos_num = str(i)
#    atom_pos_str = atom_pos_num.replace(".", "_")
#    outputfolder = f"/workspaces/MRA-OrbitalOptimization/automate_process/{atom_pos_str}_HeH_sto3g" #specify output_dir
#    os.mkdir(outputfolder) #make output_dir
#    HeH_geom = define_HeH_mole_object(atom_pos_num, 'sto-3g') #make mole object
#    make_geometry_file_HeH(outputfolder, atom_pos_num) #make geometry file
#    HeH_alpha_coeffs, HeH_beta_coeffs, ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = calculate_alpha_beta_coeffs(HeH_geom) #run UHf and get the coefficients
#    save_mocoeffs(HeH_alpha_coeffs, HeH_beta_coeffs, outputfolder, atom_pos_str) #save the rdms in npy files
#    energy, alpha_1rdm, beta_1rdm, aa_2rdm, ab_2rdm, bb_2rdm = run_dmrg_and_get_rdms(ncas, n_elec, spin, ecore, h1e, g2e, orb_sym) #run dmrg and get the rdms
#    save_rdms(alpha_1rdm, beta_1rdm, aa_2rdm, ab_2rdm, bb_2rdm, outputfolder, atom_pos_str) #save the rdms
#    orbtrans_file = make_orbitaltranslator_json(outputfolder, atom_pos_str) #make orbitaltranslator json file
#    OrbitalTranslator_cmd = ["/workspaces/MRA-OrbitalOptimization/build/madness_programs/orbital_translation/OrbitalTranslation", orbtrans_file] #make cmd for OrbitalTranslator
#    try: 
#        run_orbtrans = subprocess.run(OrbitalTranslator_cmd, capture_output=True, text=True, check=True)
#        print("Orbital Translator output:")
#        print(run_orbtrans.stdout)
#    except subprocess.CalledProcessError as e:
#        print("Error occurred:")
#        print(e.stderr)
#    spinorbopt_file = make_spinorbopt_json(outputfolder, atom_pos_str, 2) #make spinorbopt json file
#    SpinorbOpt_cmd = ["/workspaces/MRA-OrbitalOptimization/build/madness_programs/spinorb_opt/SpinorbOpt", spinorbopt_file]
#    print("Run Spin orbital refinement at bondlength ", i)
#    try: 
#        run_spinorbopt = subprocess.Popen(SpinorbOpt_cmd, stdout=subprocess.PIPE, text=True)
#        for line in run_spinorbopt.stdout:
#            print(line, end='')
#        run_spinorbopt.wait()
#    except subprocess.CalledProcessError as e:
#        print("Error occurred:")
#        print(e.stderr)


H_pos_HeH_num = str(7.0) #specify H position, I can change this in the loop
H_pos_HeH_str = H_pos_HeH_num.replace(".", "_")

outputfolder = f"/workspaces/MRA-OrbitalOptimization/automate_process/{H_pos_HeH_str}_HeH_sto3g" #specify output_dir
os.mkdir(outputfolder) #make output_dir
HeH_geom = define_HeH_mole_object(H_pos_HeH_num, 'sto-3g') #make mole object
make_geometry_file_HeH(outputfolder, H_pos_HeH_num) #make geometry file
HeH_alpha_coeffs, HeH_beta_coeffs, ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = calculate_alpha_beta_coeffs(HeH_geom) #run UHf and get the coefficients
save_mocoeffs(HeH_alpha_coeffs, HeH_beta_coeffs, outputfolder, H_pos_HeH_str) #save the rdms in npy files
energy, alpha_1rdm, beta_1rdm, aa_2rdm, ab_2rdm, bb_2rdm = run_dmrg_and_get_rdms(ncas, n_elec, spin, ecore, h1e, g2e, orb_sym) #run dmrg and get the rdms
save_rdms(alpha_1rdm, beta_1rdm, aa_2rdm, ab_2rdm, bb_2rdm, outputfolder, H_pos_HeH_str) #save the rdms

orbtrans_file = make_orbitaltranslator_json(outputfolder, H_pos_HeH_str) #make orbitaltranslator json file
OrbitalTranslator_cmd = ["/workspaces/MRA-OrbitalOptimization/build/madness_programs/orbital_translation/OrbitalTranslation", orbtrans_file] #make cmd for OrbitalTranslator


try: 
    run_orbtrans = subprocess.run(OrbitalTranslator_cmd, capture_output=True, text=True, check=True)
    print("Orbital Translator output:")
    print(run_orbtrans.stdout)
except subprocess.CalledProcessError as e:
    print("Error occurred:")
    print(e.stderr)

spinorbopt_file = make_spinorbopt_json(outputfolder, H_pos_HeH_str, 2) #make spinorbopt json file
SpinorbOpt_cmd = ["/workspaces/MRA-OrbitalOptimization/build/madness_programs/spinorb_opt/SpinorbOpt", spinorbopt_file]
print("Run Spin orbital refinement:")
try: 
    run_spinorbopt = subprocess.Popen(SpinorbOpt_cmd, stdout=subprocess.PIPE, text=True)
    for line in run_spinorbopt.stdout:
        print(line, end='')
    
    run_spinorbopt.wait()
except subprocess.CalledProcessError as e:
    print("Error occurred:")
    print(e.stderr)





