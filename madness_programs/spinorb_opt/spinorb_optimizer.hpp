#pragma once

#include <iostream>
#include <madness/mra/mra.h>
#include <madness/mra/vmra.h>
#include <madness/mra/operator.h>
#include <madness/chem/oep.h>
#include <madness/chem/projector.h>
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
using json = nlohmann::json;

class SpinorbOpt{
public:
    SpinorbOpt(int argc, char** argv, double L, long k, double thresh);
    ~SpinorbOpt();

    void CreateNuclearPotentialAndRepulsion(std::string GeometryFile);
    void ReadInitialOrbitals(std::vector<Orbital> all_alpha_orbitals, std::vector<Orbital> all_beta_orbitals, int number_alpha_active_orbitals, int number_beta_active_orbitals);
    void ReadRDMFiles(std::vector<std::string> one_body_rdm_files, std::vector<std::string> two_body_rdm_files);
    void TransformMatrix(Eigen::MatrixXd* ObjectMatrix, Eigen::MatrixXd TransformationMatrix);
    void TransformTensor(Eigen::Tensor<double, 4>* ObjectTensor, Eigen::MatrixXd TransformationMatrix);
    void TransformToNObasis();
    void CalculateAllIntegrals();
    std::vector<double> CalculateEnergy();
    void OptimizeSpinorbitals(double optimization_thresh, double NO_occupation_thresh, std::string OutputPath);
    std::vector<real_function_3d> GetAllActiveSpinorbitalUpdates(std::vector<int> spin_orbs_indices_for_update);
    void CalculateLagrangeMultiplier();
    double CalculateLagrangeMultiplierElement(int dim, int a, int i);
    //std::vector<real_function_3d> ProjectSpinorbitals(std::vector<real_function_3d> orbs);
    double CalculateTotalSpin();
    void TransformIntegralsAndOrbitalsBack();
    //void SaveNOs(std::string OutputPath);
    //void SaveSpinorbitals(std::string OutputPath); 
    //void SaveIntegralsToNumpy(std::string OutputPath);

private:
//World& world;
    World* world;
    std::vector<std::vector<double>> atoms;
    double nuclear_repulsion_energy;
    Nuclear<double,3>* Vnuc;

    json SpinorbOpt_data;

    int num_active_alpha;
    int num_active_beta;
    std::vector<int> active_spin_orb_indices;
    std::vector<int> frozen_occ_spin_orb_indices;
    std::vector<real_function_3d> active_alpha_beta_orbs;

    Eigen::MatrixXd Alpha_Rdm_Matrix;
    Eigen::MatrixXd Beta_Rdm_Matrix;
    Eigen::MatrixXd Alpha_Beta_Rdm_Matrix;
    Eigen::Tensor<double, 4> Alpha_Beta_Rdm_Tensor;
    
    Eigen::MatrixXd Alpha_Rotation_Matrix;
    Eigen::MatrixXd Beta_Rotation_Matrix;
    Eigen::MatrixXd ActiveSpaceRotationMatrix;

    std::vector<Eigen::MatrixXd> integrals_kinetic;
    Eigen::MatrixXd integrals_potential;
    Eigen::MatrixXd integrals_one_body; 
    Eigen::Tensor<double, 4> integrals_two_body;

    std::vector<real_function_3d> ab_orbs_kl; // |kl)
    std::vector<real_function_3d> ab_coul_orbs_mn; // 1/r|mn)

    Eigen::MatrixXd LagrangeMultiplier;
    double highest_error;
};