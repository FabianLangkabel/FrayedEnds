#pragma once

#include "functionsaver.hpp"
#include <iomanip>
#include <madness/mra/vmra.h>
#include<madness/chem/SCF.h>
#include<madness/chem/nemo.h>
#include<madness/chem/PNO.h>
#include <string>
#include <vector>
#include <sstream>
#include <cstring>
#include <fstream>
#include <regex>
#include <utility> // For std::pair
#include <tuple>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "MadnessProcess.hpp"

using namespace madness;
namespace nb = nanobind;

class MinBasProjector: public MadnessProcess{
	public:
		MinBasProjector(std::string argv, double L, long k, double thresh, int initial_level, int truncate_mode, bool refine, int n_threads) : MadnessProcess(L, k, thresh, initial_level, truncate_mode, refine, n_threads)
		{
			auto [argc, charArray] = stringToCharPointerArray(argv);
			parser = commandlineparser(argc,charArray);
			freeCharPointerArray(charArray,argc);
		}
		~MinBasProjector()
		{
			atomicbasis.clear();
		}

        void run()
		{
			SCF calc(*world, parser);
            calc.reset_aobasis("sto-3g");
            atomicbasis = calc.project_ao_basis(*world, calc.aobasis);
            basisname = "sto-3g";
            nuclear_repulsion = calc.molecule.nuclear_repulsion_energy();
            calc.make_nuclear_potential(*world);
            Vnuc = calc.potentialmanager -> vnuclear();
		}

		std::vector<SavedFct> get_atomic_basis()const{
		    std::vector<SavedFct> result;
		    for(auto x : atomicbasis){
		        SavedFct y(x);
		        result.push_back(y);
		    }
		    return result;
		}

		std::string get_basis_name()const{
		    return basisname;
		}

		SavedFct get_nuclear_potential(){
			return SavedFct(Vnuc);
		}

		double get_nuclear_repulsion() const
		{
			return nuclear_repulsion;
		}

		private:
        commandlineparser parser;
	    std::vector<madness::Function<double,3>> atomicbasis;
	    std::string basisname;
	    double nuclear_repulsion;
	    madness::Function<double,3> Vnuc;


};