#include "translator.hpp"
using json = nlohmann::json;

int main(int argc, char** argv)
{
    //----------------------- Read Input File -----------------------
    if(argc != 2){std::cout << "The program must be started with a json input file" << std::endl; return 0;}
    std::ifstream file(argv[1]);
    if(!file.good()){ std::cout << "Input file does not exist or cannot be read" << std::endl; return 0;}

    double box_size;
    long wavelet_order;
    double madness_thresh;
    std::string molecule_file;
    std::string calpha_coeff_file; //Input C_alpha
    std::string cbeta_coeff_file; //Input C_beta
    std::string output_folder = "";


	
    json input = json::parse(file);
    if(input.contains("box_size")) box_size = input["box_size"];
    if(input.contains("wavelet_order")) wavelet_order = input["wavelet_order"];
    if(input.contains("madness_thresh")) madness_thresh = input["madness_thresh"];
    if(input.contains("molecule_file")) molecule_file = input["molecule_file"];
    if(input.contains("calpha_coeff_file")) calpha_coeff_file = input["calpha_coeff_file"];
    if(input.contains("cbeta_coeff_file")) cbeta_coeff_file = input["cbeta_coeff_file"];
    if(input.contains("output_folder")) output_folder = input["output_folder"];
    //if(input.contains("reorder_orbitals")) reorder_orbitals = input["reorder_orbitals"];
    //if(input.contains("new_orbital_order")) new_orbital_order = input["new_orbital_order"].get<std::vector<int>>();
    //if(input.contains("write_tequila_file")) write_tequila_file = input["write_tequila_file"];
    //if(input.contains("tequila_prefix")) tequila_prefix = input["tequila_prefix"];
    //if(input.contains("tequila_indices_string")) tequila_indices_string = input["tequila_indices_string"];
    //if(input.contains("check_orthonormality")) check_orthonormality = input["check_orthonormality"];
    //if(input.contains("replace_core_with_mra_hf")) replace_core_with_mra_hf = input["replace_core_with_mra_hf"];
    //if(input.contains("number_core_orbitals")) number_core_orbitals = input["number_core_orbitals"];
    //if(input.contains("calculate_integrals")) calculate_integrals = input["calculate_integrals"];
	

    //----------------------- Execute -----------------------
    Translator* trans = new Translator(argc, argv, box_size, wavelet_order, madness_thresh);
    trans->read_mo_coeffs(calpha_coeff_file, cbeta_coeff_file); // Hier werden C_alpha und C_beta eingelesen
    trans->create_aos(molecule_file); //Hier werden AOs erstellt
    trans->create_mo_orbitals();
    trans->print_mo_orbitals(output_folder); //Hier werden die MOs auf Platte geschrieben

    return 0;
}