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

using namespace madness;

class Integrals{
public:
    Integrals(int argc, char** argv, double L, long k, double thresh);
    ~Integrals();

    void ReadOrbitals();
    void CalculateIntegrals();

private:
    World* world;
    std::vector<real_function_3d> orbitals;
};