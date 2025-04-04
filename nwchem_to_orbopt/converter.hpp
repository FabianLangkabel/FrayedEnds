#pragma once

#include <iostream>
#include <madness/mra/mra.h>
#include <madness/mra/vmra.h>
#include <madness/mra/operator.h>
#include <madness/chem/oep.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <madness/external/nlohmann_json/json.hpp>
#include <madness/chem/molecular_functors.h>
#include <madness/chem/NWChem.h>
#include "npy.hpp"

using namespace madness;
using namespace madchem;

class Converter{
public:
    Converter(int argc, char** argv, double L, long k, double thresh);
    ~Converter();

    //void create_mos(std::string GeometryFile, std::string Basisset, std::string mo_coeff_file);
    void create_mos(std::string nwchem_file);
    void define_as(int number_occupied_orbitals, std::vector<int> active_orbitals);
    void CalculateAllIntegrals();
    void CalculateCoreEnergy();
    void SaveEffectiveHamiltonian(std::string OutputPath);
    void save_orbitals(std::string output_folder);

private:
    World* world;
    std::vector<std::vector<double>> atoms;
    std::vector<real_function_3d> mos;

    double nuclear_repulsion_energy;
    Nuclear<double,3>* Vnuc;
    double core_total_energy;

    int nocc = 2; // spatial orbital = 2; spin orbitals = 1
    double truncation_tol = 1e-6;
    double coulomb_lo = 0.001;
    double coulomb_eps = 1e-6;
    double BSH_lo = 0.01;
    double BSH_eps = 1e-6;

    std::vector<real_function_3d> frozen_occ_orbs;
    std::vector<real_function_3d> active_orbs;
    int as_dim;
    int core_dim;

    

    //Integrals
    Eigen::MatrixXd as_integrals_one_body; // (k,l)
    Eigen::Tensor<double, 4> as_integrals_two_body; // (k,l,m,n)
    Eigen::Tensor<double, 3> core_as_integrals_two_body_akal; // (a,k,l)
    Eigen::Tensor<double, 3> core_as_integrals_two_body_akla; // (a,k,l)
};