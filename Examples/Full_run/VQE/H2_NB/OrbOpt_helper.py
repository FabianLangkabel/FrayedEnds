import numpy as np
import os
import shutil
import json
from pathlib import Path
import glob

                        
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
    molecule_file_str = """geometry
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


def PNO_input(**kwargs):
    if n_pno is not None and n_orbitals is not None:
        raise TequilaMadnessException(
            "n_pno={} and n_orbitals={} given ... please pick one".format(n_pno, n_orbitals))

    n_electrons = self.parameters.n_electrons
    if self.parameters.frozen_core:
            # only count active electrons (will not compute pnos for frozen pairs)
        n_core_electrons = self.parameters.get_number_of_core_electrons()
        n_electrons -= n_core_electrons

    n_pairs = n_electrons // 2
    if n_orbitals is None:
        n_orbitals = n_electrons  # minimal correlated (each active pair will have one virtual)

    if n_pno is None:
        n_pno = n_orbitals - n_pairs

    if maxrank is None:
        # need at least maxrank=1, otherwise no PNOs are computed
        # this was a bug in <=v1.8.5 
        maxrank = max(1, int(numpy.ceil(n_pno // n_pairs)))

    if maxrank <= 0:
        warnings.warn(
            "maxrank={} in tequila madness backend! No PNOs will be computed. Set the value when initializing the Molecule as tq.Molecule(..., pno={\"maxrank\":1, ...})".format(
                maxrank), TequilaWarning)

    data = {}
    if :
        raise TequilaMadnessException(
            "Currently only closed shell supported for MRA-PNO-MP2, you demanded multiplicity={} for the surrogate".format(
                    self.parameters.multiplicity))
    data["dft"] = {"charge": self.parameters.charge, "xc": "hf", "k": 7, "econv": 1.e-4, "dconv": 5.e-4,
                       "localize": "boys",
                       "ncf": "( none , 1.0 )"}
    data["pno"] = {"maxrank": maxrank, "f12": "false", "thresh": 1.e-4, "diagonal": True}
    if not self.parameters.frozen_core:            
        data["pno"]["freeze"] = 0
    data["pnoint"] = {"n_pno": n_pno, "n_virt": n_virt, "orthog": "symmetric"}
    data["plot"] = {}
    data["f12"] = {}
    for key in data.keys():
        if key in kwargs:
            data[key] = {**data[key], **kwargs[key]}

    if filename is not None:
        with open(filename, "w") as f:
            for k1, v1 in data.items():
                print(k1, file=f)
                for k2, v2 in v1.items():
                    print("{} {}".format(k2, v2), file=f)
                print("end\n", file=f)

            print("geometry", file=f)
            print("units angstrom", file=f)
            print("eprec 1.e-6", file=f)
            for line in self.parameters.get_geometry_string().split("\n"):
                line = line.strip()
                if line != "":
                    print(line, file=f)
            print("end", file=f)
