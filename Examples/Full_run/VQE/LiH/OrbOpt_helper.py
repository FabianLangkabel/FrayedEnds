import numpy as np
import os
import shutil
import json
from pathlib import Path
import glob
from tequila.quantumchemistry import ParametersQC
import tequila as tq

def get_best_initial_values(mol):
    tries = 20
    U = mol.make_ansatz(name="HCB-UpCCGD")
    best_opt = tq.quantumchemistry.optimize_orbitals(molecule=mol, circuit=U, silent=True, use_hcb=True, initial_guess="random")
    opt = tq.quantumchemistry.optimize_orbitals(molecule=mol, circuit=U, silent=True, use_hcb=True)
    if opt.energy < best_opt.energy:
        best_opt = opt
    
    for _ in range(tries):
        #opt = tq.quantumchemistry.optimize_orbitals(molecule=mol, circuit=U, silent=True, use_hcb=True, initial_guess="random")
        initial_guess = np.eye(mol.n_orbitals) + np.random.normal(scale=1.0, loc=0.0, size=mol.n_orbitals**2).reshape(mol.n_orbitals, mol.n_orbitals)
        opt = tq.quantumchemistry.optimize_orbitals(molecule=mol, circuit=U, silent=True, use_hcb=True, initial_guess=initial_guess)
        if opt.energy < best_opt.energy:
            best_opt = opt
            
    return best_opt
                        
def transform_rdms(TransformationMatrix, rdm1, rdm2):
    new_rdm1 = np.dot(np.dot(TransformationMatrix.transpose(), rdm1),TransformationMatrix)
    n = rdm2.shape[0]
    
    temp1 = np.zeros(shape=(n, n, n, n))
    for i in range(n):
        for j in range(n):
            for k2 in range(n):
                for l in range(n):
                    k_value = 0
                    for k in range(n):
                        k_value += TransformationMatrix[k][k2] * rdm2[i][j][k][l]
                    temp1[i][j][k2][l] = k_value
                    
    temp2 = np.zeros(shape=(n, n, n, n))
    for i2 in range(n):
        for j in range(n):
            for k2 in range(n):
                for l in range(n):
                    i_value = 0
                    for i in range(n):
                        i_value += TransformationMatrix[i][i2] * temp1[i][j][k2][l]
                    temp2[i2][j][k2][l] = i_value
                    
    temp3 = np.zeros(shape=(n, n, n, n))
    for i2 in range(n):
        for j in range(n):
            for k2 in range(n):
                for l2 in range(n):
                    l_value = 0
                    for l in range(n):
                        l_value += TransformationMatrix[l][l2] * temp2[i2][j][k2][l]
                    temp3[i2][j][k2][l2] = l_value
                    
    new_rdm2 = np.zeros(shape=(n, n, n, n))
    for i2 in range(n):
        for j2 in range(n):
            for k2 in range(n):
                for l2 in range(n):
                    j_value = 0
                    for j in range(n):
                        j_value += TransformationMatrix[j][j2] * temp3[i2][j][k2][l2]
                    new_rdm2[i2][j2][k2][l2] = j_value
                    
    return new_rdm1, new_rdm2

def convert_geometry_from_bohr_to_angstrom(geom):
    factor = 1/1.8897259
    geom_new = ""
    for line in geom.splitlines():
        splitted_line = line.split()
        if(len(splitted_line) == 4):
            geom_new += splitted_line[0] + " "
            geom_new += (float(splitted_line[1]) * factor).__str__() + " "
            geom_new += (float(splitted_line[2]) * factor).__str__() + " "
            geom_new += (float(splitted_line[3]) * factor).__str__() + "\n"
        else:
            geom_new += line + "\n"
    geom_new[:geom_new.rfind('\n')]
    return geom_new

def convert_geometry_from_angstrom_to_bohr(geom):
    factor = 1.8897259
    geom_new = ""
    for line in geom.splitlines():
        splitted_line = line.split()
        if(len(splitted_line) == 4):
            geom_new += splitted_line[0] + " "
            geom_new += (float(splitted_line[1]) * factor).__str__() + " "
            geom_new += (float(splitted_line[2]) * factor).__str__() + " "
            geom_new += (float(splitted_line[3]) * factor).__str__() + "\n"
        else:
            geom_new += line + "\n"
    geom_new[:geom_new.rfind('\n')]
    return geom_new

def create_molecule_file(geometry_bohr):
    molecule_file_str = """molecule
units Bohr
no_orient 1
eprec 1.e-6"""
    molecule_file_str += geometry_bohr
    molecule_file_str += "\nend"
    molecule_file_str = os.linesep.join([s for s in molecule_file_str.splitlines() if s])
    f = open("molecule", "w")
    f.write(molecule_file_str)
    f.close()
    
def create_orbital_opt_input(FilePath, it, all_orbitals, frozen_occupied_orbitals, active_orbitals, 
                             box_size, wavelet_order, madness_thresh, optimization_thresh, NO_occupation_thresh,
                             molecule_name):
    it_str = it.__str__()
    current_dir = str(Path().absolute())
    if os.path.exists(FilePath):
        os.remove(FilePath)
        
    mad_input = json.loads('{}')
    mad_input["box_size"] = box_size
    mad_input["wavelet_order"] = wavelet_order
    mad_input["madness_thresh"] = madness_thresh
    mad_input["optimization_thresh"] = optimization_thresh
    mad_input["NO_occupation_thresh"] = NO_occupation_thresh
    mad_input["molecule_file"] = current_dir + "/molecule"
    mad_input["active_space_one_rdm_file"] = current_dir + "/" + it_str + "/" + molecule_name + "_1rdm.npy"
    mad_input["active_space_two_rdm_file"] = current_dir + "/" + it_str + "/" + molecule_name + "_2rdm.npy"
    mad_input["output_folder"] = current_dir + "/" + (it + 1).__str__()
    mad_input["print_final_orbitals"] = True
    mad_input["print_final_integrals"] = True
    mad_input["orbitals"] = []
    for orb in all_orbitals:
        orb_json = json.loads('{}')
        orb_json["orbital_file_name"] = current_dir + "/" + it_str + "/mra_orbital_" + orb.__str__()
        if(orb in frozen_occupied_orbitals):
            orb_json["orbital_type"] = "frozen_occupied"
        elif(orb in active_orbitals):
            orb_json["orbital_type"] = "active"
            orb_json["active_space_index"] = active_orbitals.index(orb)
        mad_input["orbitals"].append(orb_json)
    mad_input_file = open(FilePath, 'w')
    json.dump(mad_input, mad_input_file)
    mad_input_file.close()
    
def PNO_cleanup():
    # Define the patterns for the files to delete
    patterns = [
        "*.00000",                # Files ending with .00000
        "N7madness*",             # Files starting with N7madness
        "mad.calc_info.json",     # Specific file
        "mad.restartaodata",      # Specific file
        "pnoinfo.txt"             # Specific file
    ]

    # Iterate over each pattern and delete matching files
    for pattern in patterns:
        for file in glob.glob(pattern):
            try:
                os.remove(file)
                print(f"Deleted: {file}")
            except OSError as e:
                print(f"Error deleting {file}: {e}")


def PNO_input(params: ParametersQC, molecule_file, n_pno=None, n_virt=0, maxrank=None, **kwargs) -> str:
    n_electrons = params.n_electrons
    if params.frozen_core:
            # only count active electrons (will not compute pnos for frozen pairs)
        n_core_electrons = params.get_number_of_core_electrons()
        n_electrons -= n_core_electrons

    n_pairs = n_electrons // 2
    if n_pno is None:
        n_pno = n_electrons - n_pairs

    if maxrank is None:
        maxrank = max(1, int(np.ceil(n_pno / n_pairs)))

    data = {}
    if params.multiplicity != 1:
        raise Exception(
            "Currently only closed shell supported for MRA-PNO-MP2, you demanded multiplicity={} for the surrogate".format(params.multiplicity))
    
    data["dft"] = {"charge": params.charge, "xc": "hf", "k": 7, "econv": 1.e-4, "dconv": 5.e-4,
                       "localize": "boys"}
    data["nemo"] = {"ncf": "( none , 1.0 )"}
    data["pno"] = {"maxrank": maxrank, "f12": "false", "thresh": 1.e-4, "diagonal": True}
    if not params.frozen_core:            
        data["pno"]["freeze"] = 0
    data["pnoint"] = {"n_pno": n_pno, "n_virt": n_virt, "orthog": "symmetric"}
    data["plot"] = {}
    data["f12"] = {}
    for key in data.keys():
        if key in kwargs:
            data[key] = {**data[key], **kwargs[key]}
    
    if data["pno"]["maxrank"] <= 0:
        raise Exception(
            "maxrank={} in tequila madness backend! No PNOs will be computed. Set the value when initializing the Molecule as tq.Molecule(..., pno={\"maxrank\":1, ...})".format(
                data["pnoint"]["maxrank"]))

    input_str="pno --geometry=\"source_type=inputfile; no_orient=1; source_name="+molecule_file+"\""
    input_str += " --dft=\""
    for k, v in data["dft"].items():
        input_str += "{}={}; ".format(k, v)
    input_str = input_str[:-2] + "\""
    input_str += " --pno=\""
    for k, v in data["pno"].items():
        input_str += "{}={}; ".format(k, v)
    input_str = input_str[:-2] + "\""
    input_str += " --nemo=\""
    for k, v in data["nemo"].items():
        input_str += "{}={}; ".format(k, v)
    input_str += " --pnoint=\""
    for k, v in data["pnoint"].items():
        input_str += "{}={}; ".format(k, v)
    input_str = input_str[:-2] + "\""
    if data["plot"] != {}:
        input_str += " --plot=\""
        for k, v in data["plot"].items():
            input_str += "{}={}; ".format(k, v)
        input_str = input_str[:-2] + "\""
    if data["f12"] != {}:
        input_str += " --f12=\""
        for k, v in data["f12"].items():
            input_str += "{}={}; ".format(k, v)
        input_str = input_str[:-2] + "\""
    
    return input_str
