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
#include "npy.hpp"
#include "madness_process.hpp"

using namespace madness;
using namespace madchem;

class NWChem_Converter {
  public:
    NWChem_Converter(MadnessProcess<3>& mp);
    ~NWChem_Converter();

    void read_nwchem_file(std::string nwchem_file);

    std::vector<Function<double, 3>> get_normalized_aos();
    std::vector<Function<double, 3>> get_mos();
    Function<double, 3> get_vnuc() { return Vnuc; }
    double get_nuclear_repulsion_energy() { return nuclear_repulsion_energy; }

  private:
    MadnessProcess<3>& madness_process;
    std::vector<std::vector<double>> atoms;
    std::vector<real_function_3d> aos;
    std::vector<real_function_3d> mos;
    real_function_3d Vnuc;
    double nuclear_repulsion_energy;
};