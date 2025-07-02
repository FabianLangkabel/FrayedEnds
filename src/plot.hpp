#pragma once

#include "MadnessProcess.hpp"

using namespace madness;
using namespace madchem;

class Plot : public MadnessProcess{
public:
    Plot(double L, long k, double thresh) : MadnessProcess(L,k,thresh) {std::cout.precision(6);}
    ~Plot(){}
};