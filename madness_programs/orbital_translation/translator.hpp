#pragma once

#include <iostream>
#include <madness/mra/mra.h>
#include <madness/mra/vmra.h>
#include <madness/mra/operator.h>
#include <madness/chem/oep.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <madness/chem/molecular_functors.h>
#include "npy.hpp"
#include <vector>
#include <string>

using namespace madness;
using namespace madchem;

class Translator{
public:
    Translator(int argc, char** argv, double L, long k, double thresh);
    ~Translator();

    void create_aos(std::string GeometryFile, std::string basis_set);
    void read_mo_coeffs(std::string calpha_coeff_file, std::string cbeta_coeff_file);
    void create_mo_orbitals();
    void print_mo_orbitals(std::string output_folder);

private:
    World* world;
    Eigen::MatrixXd Alpha_Coeff_Matrix;
    Eigen::MatrixXd Beta_Coeff_Matrix;
    Molecule molecule;
    std::vector<real_function_3d> ao;
    std::vector<real_function_3d> mo_alpha_orbs;
    std::vector<real_function_3d> mo_beta_orbs; //MOs
};