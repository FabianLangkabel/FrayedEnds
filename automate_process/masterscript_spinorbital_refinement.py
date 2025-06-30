from pyscf import gto, scf
import numpy as np
from pyblock2._pyscf.ao2mo import integrals as itg
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
import os
import json


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
    x = float(number)
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
    x = str(x)
    x = x.replace(".", "")
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
    x = str(x)
    x = x.replace(".", "")
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


#using this primarily to update the geometry file, but can be in priniciple used for any file that needs one line modified
def modify_geometry_file(filename, old_geom, new_geom):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()

        modified = False
        for i, line in enumerate(lines):
            if old_geom in line:
                lines[i] = new_geom + '\n'
                modified = True
                break  

        if not modified:
            print(f"No line containing '{old_geom}' was found.")
            return

        with open(filename, 'w') as file:
            file.writelines(lines)

        print(f"Line containing '{new_geom}' has been updated.")

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")

#this function creates the json files with all the necessary information for the Orbital Translator programm; k and madness_thresh need to be adjusted manually
def make_orbitaltranslator_json(coeff_path, output_json, x):
    x = x.replace(".", "")
    output_path = f"{coeff_path}"
    alpha_path = f"{coeff_path}/{x}_alpha_coeffs.npy"
    beta_path = f"{coeff_path}/{x}_beta_coeffs.npy"
    dict = {
        "box_size": 50.0,
        "wavelet_order": 8,
        "madness_thresh": 0.00001,
        "optimization_thresh": 0.001,
        "NO_occupation_thresh": 0.001,
        "molecule_file": "/workspaces/MRA-OrbitalOptimization/geometry/H3_tri_min.mol",
        "calpha_coeff_file": alpha_path,
        "cbeta_coeff_file": beta_path,
        "output_folder": output_path,
    }
    dict_json = json.dumps(dict, indent=4)
    output_jsonfile = f"{output_json}/{x}_H3lin.json"
    print(dict_json)
    print(output_jsonfile)
    with open(output_jsonfile, "w") as file:
        file.write(dict_json)
    

#this function creates the Json Files with all the information that the SpinorbitalOptimizer needs
def make_spinorbopt_json(rdms_path, orb_path, output_json, output_spinorbopt, x, num_orbs): 
    x = x.replace(".", "")
    dict_1 = {
        "box_size": 50.0,
        "wavelet_order": 7,
        "madness_thresh": 0.0001,
        "optimization_thresh": 0.001,
        "NO_occupation_thresh": 0.001,
        "molecule_file": "/workspaces/MRA-OrbitalOptimization/geometry/H3_lin_min.mol",
        "output_folder": output_spinorbopt,
        "alpha_one_rdm_file": f"{rdms_path}/{x}_alpha_1rdm.npy",
        "beta_one_rdm_file": f"{rdms_path}/{x}_beta_1rdm.npy",
        "alpha_alpha_rdm_file": f"{rdms_path}/{x}_alpha_alpha_2rdm.npy",
        "alpha_beta_rdm_file": f"{rdms_path}/{x}_alpha_beta_2rdm.npy",
        "beta_beta_rdm_file": f"{rdms_path}/{x}_beta_beta_2rdm.npy",
    }
    alpha_entries=[]
    beta_entries=[]
    for i in np.arange(0,num_orbs, 1):
        alpha_filename = f"{orb_path}/alpha_orbital_{i}"
        beta_filename = f"{orb_path}/beta_orbital_{i}"
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
              "alpha_orbital_file_name:": alpha_file,
              "alpha_orbital_type": alpha_orb_type,
              "alpha_active_space_index": alpha_idx,
            } for alpha_file, alpha_orb_type, alpha_idx in alpha_entries
        ]
    }
    beta_dict = {
        "beta_orbitals": [
            {
              "beta_orbital_file_name:": beta_file,
              "beta_orbital_type": beta_orb_type,
              "beta_active_space_index": beta_idx,
            } for beta_file, beta_orb_type, beta_idx in beta_entries
        ]
    }

    combined_data = {**dict_1, **alpha_dict, **beta_dict}
    dict_json = json.dumps(combined_data, indent=4)
    output_jsonfile = f"{output_json}_{x}.json"
    #print(dict_json)
    #print(output_jsonfile)
    with open(output_jsonfile, "w") as file:
        file.write(dict_json)




#use os.mkdir("directory_name") to make a new directory for the new bondlength

#delete_tmp_file('/workspaces/MRA-OrbitalOptimization/automate_process/tmp2')
delete_tmp_file('/workspaces/MRA-OrbitalOptimization/automate_process/tmp4')

new_pos = str(1.75743603) #the next lines are going to incorporated into the loop
new_geom = f'H {new_pos} 0.00000000 0.00000000'
old_pos = str(1.85192233)
old_geom = f'H {old_pos} 0.00000000 0.00000000'

H_pos = str(1.75743603)
#modify_geometry_file('/workspaces/MRA-OrbitalOptimization/geometry/H3_lin_min.mol', old_geom, new_geom)
H3_lin_geom = define_H3lin_mole_object(1.75743603, 'sto-3g')
#HeH_geom = define_HeH_mole_object(2.2676711863, '6-31g')
#H3_tri_geom = define_H3tri_mole_object(1.42379912, '6-31g')
H3lin_alpha_coeffs, H3lin_beta_coeffs, ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = calculate_alpha_beta_coeffs(H3_lin_geom)
#HeH_alpha_coeffs, HeH_beta_coeffs, ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = calculate_alpha_beta_coeffs(HeH_geom)
#H3tri_alpha_coeffs, H3tri_beta_coeffs, ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = calculate_alpha_beta_coeffs(H3_tri_geom)
save_mocoeffs(H3lin_alpha_coeffs, H3lin_beta_coeffs, '/workspaces/MRA-OrbitalOptimization/coefficients/h3lin_sto3g_coeffs', 1.75743603)
energy, alpha_1rdm, beta_1rdm, aa_2rdm, ab_2rdm, bb_2rdm = run_dmrg_and_get_rdms(ncas, n_elec, spin, ecore, h1e, g2e, orb_sym)
save_rdms(alpha_1rdm, beta_1rdm, aa_2rdm, ab_2rdm, bb_2rdm, '/workspaces/MRA-OrbitalOptimization/reduced_density_matrices/h3lin_sto3g_min', 1.75743603)