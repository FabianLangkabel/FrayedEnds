#include "optimization.hpp"
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
    std::string active_space_one_rdm_file;
    std::string active_space_two_rdm_file;
    std::string output_folder;
    bool print_final_orbitals;
    bool print_final_integrals;
	bool print_final_nos;


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

    if(!input.contains("active_space_one_rdm_file"))
    {
        std::cout << "active_space_one_rdm_file parameter could not be read from input file" << std::endl; 
        return 0;
    }
    else
    {
        active_space_one_rdm_file = input["active_space_one_rdm_file"];
    }

    if(!input.contains("active_space_two_rdm_file"))
    {
        std::cout << "active_space_two_rdm_file parameter could not be read from input file" << std::endl; 
        return 0;
    }
    else
    {
        active_space_two_rdm_file = input["active_space_two_rdm_file"];
    }

	if(!input.contains("print_final_nos"))
    {
        std::cout << "print_final_nos parameter could not be read from input file and is set to False instead" << std::endl; 
        print_final_nos = false;
    }
    else
    {
        print_final_nos = input["print_final_orbitals"];
    }
	
    if(!input.contains("print_final_orbitals"))
    {
        std::cout << "print_final_orbitals parameter could not be read from input file and is set to False instead" << std::endl; 
        print_final_orbitals = false;
    }
    else
    {
        print_final_orbitals = input["print_final_orbitals"];
    }

    if(!input.contains("print_final_integrals"))
    {
        std::cout << "print_final_integrals parameter could not be read from input file and is set to False instead" << std::endl; 
        print_final_integrals = false;
    }
    else
    {
        print_final_integrals = input["print_final_integrals"];
    }

    if(!input.contains("output_folder"))
    {
        std::cout << "output_folder parameter could not be read from input file" << std::endl; 
        std::cout << "print_final_orbitals and print_final_integrals are set to False" << std::endl; 
        print_final_orbitals = false;
        print_final_integrals = false;
		print_final_nos = false;
    }
    else
    {
        output_folder = input["output_folder"];
    }


    if(!input.contains("orbitals"))
    {
        std::cout << "orbitals parameter could not be read from input file" << std::endl; 
        return 0;
    }

    std::vector<std::string> frozen_occ_orbs_files;
    std::vector<std::string> active_orbs_files;
    std::vector<std::string> frozen_virt_orb_files;

    int number_active_orbitals = 0;
    for(int i = 0; i < input["orbitals"].size(); i++)
    {
        json OrbitalDetails = input["orbitals"][i];
        if(!OrbitalDetails.contains("orbital_file_name")){std::cout << "orbital_file_name for the "<< i << "-th orbital could not be read" << std::endl; return 0;}
        if(!OrbitalDetails.contains("orbital_type")){std::cout << "orbital_type for orbital "<< OrbitalDetails["orbital_file_name"] << " could not be read" << std::endl; return 0;}
        if(OrbitalDetails["orbital_type"] == "frozen_occupied"){frozen_occ_orbs_files.push_back(OrbitalDetails["orbital_file_name"]);}
        else if(OrbitalDetails["orbital_type"] == "active"){active_orbs_files.push_back(OrbitalDetails["orbital_file_name"]);}
        else if(OrbitalDetails["orbital_type"] == "inactive_virtual"){frozen_virt_orb_files.push_back(OrbitalDetails["orbital_file_name"]);}
        else{std::cout << "The orbital type " << OrbitalDetails["orbital_type"] << " of Orbital " << OrbitalDetails["orbital_file_name"] << " is not known. Currently the types frozen_occupied, inactive_virtual and active can be set." << std::endl; return 0;}
    }


    //----------------------- Perform Orbital Optimization -----------------------
    Optimization* opti = new Optimization(argc, argv, box_size, wavelet_order, madness_thresh);
    opti->nocc = 2; // spatial orbital = 2; spin orbitals = 1
    opti->truncation_tol = 1e-6;
    opti->coulomb_lo = 0.001;
    opti->coulomb_eps = 1e-6;
    opti->BSH_lo = 0.01;
    opti->BSH_eps = 1e-6;

    std::cout << "Read rdms, create initial guess and calculate initial energy" << std::endl;
    opti->CreateNuclearPotentialAndRepulsion(molecule_file);
    opti->ReadInitialOrbitals(frozen_occ_orbs_files, active_orbs_files, frozen_virt_orb_files);
    opti->ReadRDMFilesAndRotateOrbitals(active_space_one_rdm_file, active_space_two_rdm_file);
    opti->CalculateAllIntegrals();
    opti->CalculateCoreEnergy();
    opti->CalculateEnergies();

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "Start orbital optimization" << std::endl;
    opti->OptimizeOrbitals(optimization_thresh, NO_occupation_thresh);
    
    opti->RotateOrbitalsBackAndUpdateIntegrals();
    opti->SaveOrbitals(output_folder);
    opti->SaveEffectiveHamiltonian(output_folder);

    return 0;
}