import json
import numpy as np


def dict_spinorbopt(rdms_path, orb_path, output_json, output_spinorbopt, x, num_orbs): 
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


def make_geometry_file_HeH(directory, position):
    pos = position.replace(".", "")
    filename = f"{directory}/{pos}_geometry_HeH.mol"
    with open(filename, "a") as file:
        l1 = "geometry \n"
        l2 = "units Bohr \n"
        l3 = "no orient 1 \n"
        l4 = "eprec 1e-6 \n"
        l5 = "He 0.00000000 0.00000000 0.00000000 \n"
        l6 = f"H {position} 0.00000000 0.00000000 \n"
        l7 = "end"
        file.writelines([l1, l2, l3, l4, l5, l6, l7])


H_pos_HeH = str(2.2676711863)
#rdms_path = "/workspaces/MRA-OrbitalOptimization/reduced_density_matrices/h3lin_631g_min"
#orb_path = "/workspaces/MRA-OrbitalOptimization/coefficients/h3lin_631g_coeffs/h3lin_min/output_k8"
output_jsonfile = "/workspaces/MRA-OrbitalOptimization/automate_process"
#output_spinorbopt="/workspaces/MRA-OrbitalOptimization/output_folder/h3lin_631g_min_k8"
#x = str(1.75743603)
#dict_spinorbopt(rdms_path, orb_path, output_jsonfile, output_spinorbopt, x, 6)
make_geometry_file_HeH(output_jsonfile, H_pos_HeH)

