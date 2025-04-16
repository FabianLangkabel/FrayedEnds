#include "converter.hpp"
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
    //std::string molecule_file;
    //std::string mo_coeff_file;
    //std::string basisset;
    std::string nwchem_file;
    std::string output_folder;
    int number_occupied_orbitals;
    std::vector<int> active_orbitals;


    json input = json::parse(file);
    if(!input.contains("box_size"))
    {
        std::cout << "box_size parameter could not be read from input file" << std::endl; 
        return 0;
    }
    else
    {
        box_size = input["box_size"];
    }

    if(!input.contains("wavelet_order"))
    {
        std::cout << "wavelet_order parameter could not be read from input file" << std::endl; 
        return 0;
    }
    else
    {
        wavelet_order = input["wavelet_order"];
    }

    if(!input.contains("madness_thresh"))
    {
        std::cout << "madness_thresh parameter could not be read from input file and is set to 0.0001 instead" << std::endl; 
        madness_thresh = 0.0001;
    }
    else
    {
        madness_thresh = input["madness_thresh"];
    }
    /*
    if(!input.contains("molecule_file"))
    {
        std::cout << "molecule_file parameter could not be read from input file" << std::endl; 
        return 0;
    }
    else
    {
        molecule_file = input["molecule_file"];
    }
    if(!input.contains("mo_coeff_file"))
    {
        std::cout << "mo_coeff_file parameter could not be read from input file" << std::endl; 
        return 0;
    }
    else
    {
        mo_coeff_file = input["mo_coeff_file"];
    }
    if(!input.contains("basisset"))
    {
        std::cout << "basisset parameter could not be read from input file" << std::endl; 
        return 0;
    }
    else
    {
        basisset = input["basisset"];
    }
    */

    if(!input.contains("nwchem_file"))
    {
        std::cout << "nwchem_file parameter could not be read from input file" << std::endl; 
        return 0;
    }
    else
    {
        nwchem_file = input["nwchem_file"];
    }
    if(!input.contains("output_folder"))
    {
        std::cout << "output_folder parameter could not be read from input file" << std::endl; 
        return 0;
    }
    else
    {
        output_folder = input["output_folder"];
    }
    if(!input.contains("number_occupied_orbitals"))
    {
        std::cout << "number_occupied_orbitals parameter could not be read from input file" << std::endl; 
        return 0;
    }
    else
    {
        number_occupied_orbitals = input["number_occupied_orbitals"];
    }
    if(!input.contains("active_orbitals"))
    {
        std::cout << "active_orbitals parameter could not be read from input file" << std::endl; 
        return 0;
    }
    else
    {
        active_orbitals = input["active_orbitals"].get<std::vector<int>>();
    }

    Converter* conv = new Converter(argc, argv, box_size, wavelet_order, madness_thresh);
    std::cout << "Create MOs" << std::endl;
    conv->create_mos(nwchem_file);
    std::cout << "Define AS" << std::endl;
    conv->define_as(number_occupied_orbitals, active_orbitals);
    std::cout << "Calculate all Integrals" << std::endl;
    conv->CalculateAllIntegrals();
    std::cout << "Calculate Core Energy" << std::endl;
    conv->CalculateCoreEnergy();
    std::cout << "Save Hamiltonian and Orbitals" << std::endl;
    conv->SaveEffectiveHamiltonian(output_folder);
    conv->save_orbitals(output_folder);

    return 0;
}