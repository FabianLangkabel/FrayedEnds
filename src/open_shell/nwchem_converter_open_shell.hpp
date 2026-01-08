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
#include <madness/external/nlohmann_json/json.hpp>
#include <madness/chem/molecular_functors.h>
#include <madness/chem/NWChem.h>
#include <madness/chem/correlationfactor.h>
#include <madness/chem/potentialmanager.h>
#include "../npy.hpp"
#include "../functionsaver.hpp"
#include "../madness_process.hpp"

using namespace madness;
using namespace madchem;

class NWChem_Converter_open_shell {
  public:
    NWChem_Converter_open_shell(MadnessProcess<3>& mp);
    ~NWChem_Converter_open_shell();

    void read_nwchem_file(std::string nwchem_file);

    std::vector<SavedFct<3>> get_normalized_aos();
    std::vector<SavedFct<3>> get_alpha_mos();
    std::vector<SavedFct<3>> get_beta_mos();
    SavedFct<3> get_vnuc() { return SavedFct<3>(Vnuc); }
    double get_nuclear_repulsion_energy() { return nuclear_repulsion_energy; }

  private:
    MadnessProcess<3>& madness_process;
    std::vector<std::vector<double>> atoms;
    std::vector<real_function_3d> aos;
    std::vector<real_function_3d> alpha_mos;
    std::vector<real_function_3d> beta_mos;
    real_function_3d Vnuc;
    double nuclear_repulsion_energy;
};