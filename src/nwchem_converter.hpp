#pragma once

#include <iostream>
#include <madness/mra/mra.h>
#include <madness/mra/vmra.h>
#include <madness/mra/operator.h>
#include <madness/chem/oep.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <nlohmann/json.hpp>
#include <madness/chem/molecular_functors.h>
#include <madness/chem/NWChem.h>
#include <madness/chem/correlationfactor.h>
#include <madness/chem/potentialmanager.h>
#include "functionsaver.hpp"
#include "madness_process.hpp"

using namespace madness;
using namespace madchem;

class NWChem_Converter {
  public:
    NWChem_Converter(MadnessProcess<3>& mp);
    ~NWChem_Converter();

    void read_nwchem_file(std::string nwchem_file);

    std::vector<SavedFct<3>> get_normalized_aos();
    std::vector<SavedFct<3>> get_mos();
    SavedFct<3> get_vnuc() { return SavedFct<3>(Vnuc); }
    double get_nuclear_repulsion_energy() { return nuclear_repulsion_energy; }
    std::vector<double> get_occupancies() { return occupancies; }
    std::vector<double> get_orbital_energies() { return orbital_energies; }
    // atoms as (symbol, x, y, z) in bohr, in the coordinate frame of the NWChem calculation
    std::vector<std::tuple<std::string, double, double, double>> get_atoms() { return atoms; }

  private:
    MadnessProcess<3>& madness_process;
    std::vector<std::tuple<std::string, double, double, double>> atoms;
    std::vector<double> occupancies;
    std::vector<double> orbital_energies;
    std::vector<real_function_3d> aos;
    std::vector<real_function_3d> mos;
    real_function_3d Vnuc;
    double nuclear_repulsion_energy;
};