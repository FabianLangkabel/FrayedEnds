#pragma once

#include <iostream>
#include <madness/mra/mra.h>
#include <madness/mra/vmra.h>
#include <madness/mra/operator.h>
#include <madness/chem/oep.h>

using namespace madness;

class Orbital{
public:
    Orbital(std::string orbital_file_name, std::string orbital_type){this->orbital_file_name = orbital_file_name, this->orbital_type = orbital_type; }
    std::string orbital_file_name;
    std::string orbital_type;
    int active_space_index;
    real_function_3d function;
};