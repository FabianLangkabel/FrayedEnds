#include "spinorb_optimizer.hpp"
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
    double optimization_thresh;
    double NO_occupation_thresh;
    std::string molecule_file;
    std::string output_folder;
    bool print_final_integrals;
    std::vector<std::string> one_body_rdm_files;
    std::vector<std::string> two_body_rdm_files;

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

    if(!input.contains("optimization_thresh"))
    {
        std::cout << "optimization_thresh parameter could not be read from input file and is set to 0.001 instead" << std::endl; 
        optimization_thresh = 0.001;
    }
    else
    {
        optimization_thresh = input["optimization_thresh"];
    }

    if(!input.contains("NO_occupation_thresh"))
    {
        std::cout << "NO_occupation_thresh parameter could not be read from input file" << std::endl;
        NO_occupation_thresh = 0.001;
    }
    else
    {
        NO_occupation_thresh = input["NO_occupation_thresh"];
    }

    if(!input.contains("molecule_file"))
    {
        std::cout << "molecule_file parameter could not be read from input file" << std::endl; 
        return 0;
    }
    else
    {
        molecule_file = input["molecule_file"];
    }
	
    /*if(!input.contains("print_final_integrals"))
    {
        std::cout << "print_final_integrals parameter could not be read from input file and is set to False instead" << std::endl; 
        print_final_integrals = false;
    }
    else
    {
        print_final_integrals = input["print_final_integrals"];
    }*/

    if(!input.contains("output_folder"))
    {
        std::cout << "output_folder parameter could not be read from input file" << std::endl; 
        std::cout << "print_final_orbitals and print_final_integrals are set to False" << std::endl; 
        print_final_integrals = false;
    }
    else
    {
        output_folder = input["output_folder"];
    }
    if(!input.contains("alpha_orbitals"))
    {
        std::cout << "alpha orbitals parameter could not be read from input file" << std::endl; 
        return 0;
    }

    if(!input.contains("beta_orbitals"))
    {
        std::cout << "beta orbitals parameter could not be read from input file" << std::endl; 
        return 0;
    }

    if(!input.contains("alpha_one_rdm_file"))
    {
        std::cout << "alpha one-body rdm files could not be read from input file" << std::endl; 
        return 0;
    }
    else
    {
        std::string alpha_rdm= input["alpha_one_rdm_file"];
        one_body_rdm_files.push_back(alpha_rdm);
    }
    
    if(!input.contains("beta_one_rdm_file"))
    {
        std::cout << "beta one-body rdm files could not be read from input file" << std::endl; 
        return 0;
    }
    else
    {
        std::string beta_rdm= input["beta_one_rdm_file"];
        one_body_rdm_files.push_back(beta_rdm);
    }

    if(!input.contains("alpha_alpha_rdm_file"))
    {
        std::cout << "alpha alpha two body rdm files could not be read from input file" << std::endl; 
        return 0;
    }
    else
    {
        std::string alpha_alpha_rdm= input["alpha_alpha_rdm_file"];
        two_body_rdm_files.push_back(alpha_alpha_rdm);
    }
    
    if(!input.contains("alpha_beta_rdm_file"))
    {
        std::cout << "alpha beta two body rdm files could not be read from input file" << std::endl; 
        return 0;
    }
    else
    {
        std::string alpha_beta_rdm= input["alpha_beta_rdm_file"];
        two_body_rdm_files.push_back(alpha_beta_rdm);
    }
    

    if(!input.contains("beta_beta_rdm_file"))
    {
        std::cout << "beta beta two body rdm files could not be read from input file" << std::endl; 
        return 0;
    }
    else
    {
        std::string beta_beta_rdm= input["beta_beta_rdm_file"];
        two_body_rdm_files.push_back(beta_beta_rdm);
    }

    
    std::vector<Orbital> all_alpha_orbitals;
    int number_alpha_active_orbitals = 0;
    for(int i = 0; i < input["alpha_orbitals"].size(); i++)
    {
        json AlphaOrbitalDetails = input["alpha_orbitals"][i];
        if(!AlphaOrbitalDetails.contains("alpha_orbital_file_name")){std::cout << "alpha_orbital_file_name for the "<< i << "-th orbital could not be read" << std::endl; return 0;}
        if(!AlphaOrbitalDetails.contains("alpha_orbital_type")){std::cout << "alpha_orbital_type for orbital "<< AlphaOrbitalDetails["alpha_orbital_file_name"] << " could not be read" << std::endl; return 0;}
        if(AlphaOrbitalDetails["alpha_orbital_type"] != "frozen_occupied" && AlphaOrbitalDetails["alpha_orbital_type"] != "active" && AlphaOrbitalDetails["alpha_orbital_type"] != "inactive_virtual")
        {std::cout << "The orbital type " << AlphaOrbitalDetails["alpha_orbital_type"] << " of Orbital " << AlphaOrbitalDetails["alpha_orbital_file_name"] << " is not known. Currently the types frozen_occupied, inactive_virtual and active can be set." << std::endl; return 0;}
        if(AlphaOrbitalDetails["alpha_orbital_type"] == "active" && !AlphaOrbitalDetails.contains("alpha_active_space_index")){std::cout << "alpha_active_space_index for active orbital "<< AlphaOrbitalDetails["alpha_orbital_file_name"] << " could not be read" << std::endl; return 0;}
        all_alpha_orbitals.push_back(Orbital(AlphaOrbitalDetails["alpha_orbital_file_name"], AlphaOrbitalDetails["alpha_orbital_type"]));
        if(AlphaOrbitalDetails["alpha_orbital_type"] == "active"){all_alpha_orbitals[i].active_space_index = AlphaOrbitalDetails["alpha_active_space_index"]; number_alpha_active_orbitals++;}
    }

    std::vector<Orbital> all_beta_orbitals;
    int number_beta_active_orbitals = 0;
    for(int j = 0; j < input["beta_orbitals"].size(); j++)
    {
        json BetaOrbitalDetails = input["beta_orbitals"][j];
        if(!BetaOrbitalDetails.contains("beta_orbital_file_name")){std::cout << "beta_orbital_file_name for the "<< j << "-th orbital could not be read" << std::endl; return 0;}
        if(!BetaOrbitalDetails.contains("beta_orbital_type")){std::cout << "beta_orbital_type for orbital "<< BetaOrbitalDetails["beta_orbital_file_name"] << " could not be read" << std::endl; return 0;}
        if(BetaOrbitalDetails["beta_orbital_type"] != "frozen_occupied" && BetaOrbitalDetails["beta_orbital_type"] != "active" && BetaOrbitalDetails["beta_orbital_type"] != "inactive_virtual")
        {std::cout << "The orbital type " << BetaOrbitalDetails["beta_orbital_type"] << " of Orbital " << BetaOrbitalDetails["beta_orbital_file_name"] << " is not known. Currently the types frozen_occupied, inactive_virtual and active can be set." << std::endl; return 0;}
        if(BetaOrbitalDetails["beta_orbital_type"] == "active" && !BetaOrbitalDetails.contains("beta_active_space_index")){std::cout << "beta_active_space_index for active orbital "<< BetaOrbitalDetails["beta_orbital_file_name"] << " could not be read" << std::endl; return 0;}
        all_beta_orbitals.push_back(Orbital(BetaOrbitalDetails["beta_orbital_file_name"], BetaOrbitalDetails["beta_orbital_type"]));
        if(BetaOrbitalDetails["beta_orbital_type"] == "active"){all_beta_orbitals[j].active_space_index = BetaOrbitalDetails["beta_active_space_index"]; number_beta_active_orbitals++;}
    }


    //----------------------- Perform Integrals Calculation -----------------------
    SpinorbOpt* opti = new SpinorbOpt(argc, argv, box_size, wavelet_order, madness_thresh);
    std::cout << "Read orbitals and calculate the integrals" << std::endl;
    opti->CreateNuclearPotentialAndRepulsion(molecule_file);
    opti->ReadInitialOrbitals(all_alpha_orbitals, all_beta_orbitals, number_alpha_active_orbitals, number_beta_active_orbitals);
    opti->ReadRDMFiles(one_body_rdm_files, two_body_rdm_files);
    //opti->TransformToNObasis();
    opti->CalculateAllIntegrals();
    opti->CalculateEnergy();
    

    //----------------------- Write Output -----------------------
    //opti->RotateOrbitalsAndIntegralsBack();
    //if(print_final_integrals){opti->SaveIntegralsToNumpy(output_folder);}
    //opti->SaveIntegralsToNumpy(output_folder);

    return 0;
}