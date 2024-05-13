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
#include "npy.hpp"
#include "orbital.hpp"

using namespace madness;

class Optimization{
public:
    Optimization(int argc, char** argv, double L, long k, double thresh);
    ~Optimization();

    void CreateNuclearPotentialAndRepulsion(std::string GeometryFile);
    void ReadInitialOrbitals(std::vector<Orbital> all_orbitals, int number_active_orbitals);
    

    void ReadRDMFilesAndRotateOrbitals(std::string one_rdm_file, std::string two_rdm_file);
    void RotateOrbitalsAndIntegralsBack();
    void TransformMatrix(Eigen::MatrixXd* ObjectMatrix, Eigen::MatrixXd TransformationMatrix);
    void TransformTensor(Eigen::Tensor<double, 4>* ObjectTensor, Eigen::MatrixXd TransformationMatrix);
    void CalculateAllIntegrals();
    void UpdateIntegrals();
    void CalculateEnergies();
    void OptimizeOrbitals(double optimization_thresh, double NO_occupation_thresh);
    real_function_3d GetOrbitalUpdate(int i);
    void CalculateLagrangeMultiplier();
    double CalculateLagrangeMultiplierElement(int dim, int a, int i);
    void SaveOrbitals(std::string OutputPath);
    void SaveIntegralsToNumpy(std::string OutputPath);


private:
//World& world;
    World* world;
    std::vector<std::vector<double>> atoms;
    double nuclear_repulsion_energy;
    Nuclear<double,3>* Vnuc;

    std::vector<Orbital> all_orbitals;
    std::vector<int> active_orbital_indicies;
    std::vector<int> frozen_occupied_orbital_indicies;

    Eigen::MatrixXd ActiveSpaceRotationMatrix;
    Eigen::MatrixXd full_one_rdm;
    Eigen::Tensor<double, 4> full_two_rdm;

    std::vector<Eigen::MatrixXd> integrals_kinetic;
    Eigen::MatrixXd integrals_potential;
    Eigen::MatrixXd integrals_one_body;
    Eigen::Tensor<double, 4> integrals_two_body;

    std::vector<real_function_3d> orbs_kl; // |kl>
    std::vector<real_function_3d> coul_orbs_mn; // 1/r|mn>

    Eigen::MatrixXd LagrangeMultiplier;
    double highest_error;
};