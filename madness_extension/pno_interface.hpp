#pragma once

#include <iomanip>
#include<madness/chem/SCF.h>
#include<madness/chem/nemo.h>
#include<madness/chem/PNO.h>
#include <madness/misc/info.h>


using namespace madness;

// DEFINE PARAMETER TAGS FOR THE INPUT FILE
const std::string TAG_PNO = "pno";
const std::string TAG_F12 = "f12";
const std::string TAG_CP = "computeprotocol";


inline char** stringToCharPointerArray(const std::string& input) {
    // Split the input string by spaces
    std::istringstream stream(input);
    std::vector<std::string> tokens;
    std::string token;
    while (stream >> token) {
        tokens.push_back(token);
    }

    // Allocate memory for the array of char* pointers
    char** charArray = new char*[tokens.size()];

    // Copy each token into the array
    for (size_t i = 0; i < tokens.size(); ++i) {
        charArray[i] = new char[tokens[i].size() + 1]; // +1 for null terminator
        std::strcpy(charArray[i], tokens[i].c_str());
    }

    return charArray;
}

// Clean up memory
inline void freeCharPointerArray(char** charArray, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        delete[] charArray[i]; // Free each string
    }
    delete[] charArray; // Free the array of pointers
}


class PNOInterface{
	public:
		PNOInterface(int argc, std::string argv, double L, long k, double thresh)
		{
			int arg = 0;
			char **a = new char*[0]();
		
			world = &initialize(arg, a);
			startup(*world,arg,a);
			delete[] a;
			
			std::cout.precision(6);
		
			FunctionDefaults<3>::set_k(k);
			FunctionDefaults<3>::set_thresh(thresh);
			FunctionDefaults<3>::set_refine(true);
			FunctionDefaults<3>::set_initial_level(5);
			FunctionDefaults<3>::set_truncate_mode(1);
			FunctionDefaults<3>::set_cubic_cell(-L, L);

			char** charArray = stringToCharPointerArray(argv);

			parser = commandlineparser(argc,charArray);
			freeCharPointerArray(charArray,argc);
		}
		~PNOInterface()
		{	std::cout << "Finalize madness env" << std::endl;
			world->gop.fence();
			finalize();
		}
		
		void run() 
		{
			if(world->rank()==0){
				std::cout << "\n\n";
				std::cout << "-------------------------------------------------------------------------------------\n";
				std::cout << "SOLVING MRA-PNO-F12 as described in \n";
				std::cout << "J.S. Kottmann, F.A. Bischoff, E.F. Valeev\n";
				std::cout << "Direct determination of optimal pair-natural orbitals in a real-space representation:\n";
				std::cout << "the second-order Møller-Plesset energy\n";
				std::cout << "Journal of Chemical Physics ... 2019\n";
				std::cout << "-------------------------------------------------------------------------------------\n";
				std::cout << "\n\n";
			}
		
			if (parser.key_exists("help")) {
				PNO::help();
			
			} else if (parser.key_exists("print_parameters")) {
				PNO::print_parameters();
		
			} else {


				// Compute the SCF Reference
				const double time_scf_start = wall_time();
				Nemo nemo(*world,parser);
				nemo.get_calc()->param.print();
				const double scf_energy = nemo.value();
				if (world->rank() == 0) print("nemo energy: ", scf_energy);
				if (world->rank() == 0) printf(" at time %.1f\n", wall_time());
				const double time_scf_end = wall_time();
		
				// Compute MRA-PNO-MP2-F12
				const double time_pno_start = wall_time();
				PNOParameters parameters(*world,parser,nemo.get_calc()->molecule,TAG_PNO);
				F12Parameters paramf12(*world, parser, parameters, TAG_F12);
				PNO pno(*world, nemo, parameters, paramf12);
				std::vector<PNOPairs> all_pairs;
				pno.solve(all_pairs);
				const double time_pno_end = wall_time();
		
		
				if(world->rank()==0){
					std::cout << std::setfill(' ');
					std::cout << "\n\n\n";
					std::cout << "--------------------------------------------------\n";
					std::cout << "MRA-PNO-MP2-F12 ended \n";
					std::cout << "--------------------------------------------------\n";
					std::cout << std::setw(25) << "time scf" << " = " << time_scf_end - time_scf_start << "\n";
					std::cout << std::setw(25) << "energy scf" << " = " << scf_energy << "\n";
					std::cout << "--------------------------------------------------\n";
				}
				double mp2_energy = 0.0;
				std::cout<< std::setw(25) << "time pno" << " = " << time_pno_end - time_pno_start << "\n";
				for(const auto& pairs: all_pairs){
					if(pairs.type == MP2_PAIRTYPE){
						mp2_energy = pairs.energies.total_energy();
					}
					std::pair<size_t, size_t> ranks= pno.get_average_rank(pairs.pno_ij);
					if(world->rank()==0){
						std::string name;
						std::stringstream ss;
						ss << pairs.type;
						ss >> name;
						std::cout<< std::setw(25) << "energy "+name << " = " << pairs.energies.total_energy() << "\n";
						std::cout<< std::setw(25) << "average pno rank " + name << " = " << ranks.first << "\n";
						std::cout<< std::setw(25) << "max pno rank " + name << " = " << ranks.second << "\n";
					}
				}
				if(world->rank()==0 and mp2_energy != 0.0){
					std::cout << "--------------------------------------------------\n";
						std::cout<< std::setw(25) << "energy(total)" << " = " << scf_energy + mp2_energy << "\n";
						std::cout << "--------------------------------------------------\n";
						std::cout << "\n\n\n";
				}
			}
		
		}


	private:
		World* world;
		commandlineparser parser;
		
};


