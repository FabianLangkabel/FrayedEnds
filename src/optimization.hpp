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
#include "functionsaver.hpp"
#include "MadnessProcess.hpp"

using namespace madness;

class Optimization {
public:
    Optimization(MadnessProcess& mp);
    ~Optimization();

    //input
    void GiveInitialOrbitals(std::vector<SavedFct> all_orbs);
    void GiveRDMsAndRotateOrbitals(std::vector<double> one_rdm_elements, std::vector<double> two_rdm_elements); //TODO

    //output
    double GetC();
    std::vector<double> GetHTensor();
    std::vector<double> GetGTensor();
    std::vector<SavedFct> GetOrbitals();

    void CreateNuclearPotentialAndRepulsion(std::string GeometryFile);
    void GiveCustomPotential(SavedFct custom_pot);
    void ReadInitialOrbitals(std::vector<std::string> frozen_occ_orbs_files, std::vector<std::string> active_orbs_files, std::vector<std::string> frozen_virt_orb_files);
    void ReadRDMFilesAndRotateOrbitals(std::string one_rdm_file, std::string two_rdm_file);
    void TransformMatrix(Eigen::MatrixXd* ObjectMatrix, Eigen::MatrixXd TransformationMatrix);
    void TransformTensor(Eigen::Tensor<double, 4>* ObjectTensor, Eigen::MatrixXd TransformationMatrix);
    void CalculateAllIntegrals();
    void CalculateCoreEnergy();
    void CalculateEnergies();
    void CalculateLagrangeMultiplier();
    double CalculateLagrangeMultiplierElement_AS_AS(int z, int i);
    double CalculateLagrangeMultiplierElement_AS_Core(int z, int i);
    void OptimizeOrbitals(double optimization_thresh, double NO_occupation_thresh);
    std::vector<real_function_3d> GetAllActiveOrbitalUpdates(std::vector<int> orbital_indicies_for_update);
    void RotateOrbitalsBackAndUpdateIntegrals();
    void SaveOrbitals(std::string OutputPath);
    void SaveEffectiveHamiltonian(std::string OutputPath);

    int nocc = 2; // spatial orbital = 2; spin orbitals = 1
    double truncation_tol = 1e-6;
    double coulomb_lo = 0.001;
    double coulomb_eps = 1e-6;
    double BSH_lo = 0.01;
    double BSH_eps = 1e-6;
    
private:
    MadnessProcess& madness_process;

    //Madness + Molecule
    std::vector<std::vector<double>> atoms;
    double nuclear_repulsion_energy=0.0;
    Nuclear<double,3>* Vnuc=nullptr;
    bool use_custom_potential=false;
    real_function_3d* custom_potential;

    //Orbitals
    std::vector<std::string> frozen_occ_orbs_files;
    std::vector<std::string> active_orbs_files;
    std::vector<std::string> frozen_virt_orb_files;
    std::vector<real_function_3d> frozen_occ_orbs;
    std::vector<real_function_3d> active_orbs;
    std::vector<real_function_3d> frozen_virt_orb;
    int core_dim;
    int as_dim;
    int froz_virt_dim;

    //RDMs
    Eigen::MatrixXd ActiveSpaceRotationMatrix;
    Eigen::MatrixXd as_one_rdm;
    Eigen::Tensor<double, 4> as_two_rdm;

    //Integrals
    Eigen::MatrixXd as_integrals_one_body; // (k,l)
    Eigen::Tensor<double, 4> as_integrals_two_body; // (k,l,m,n)

    Eigen::MatrixXd core_as_integrals_one_body_ak; // (a,k)
    Eigen::Tensor<double, 4> core_as_integrals_two_body_akln; // (a,k,l,n)
    Eigen::Tensor<double, 3> core_as_integrals_two_body_akal; // (a,k,l)
    Eigen::Tensor<double, 3> core_as_integrals_two_body_akla; // (a,k,l)
    Eigen::Tensor<double, 3> core_as_integrals_two_body_abak; // (a,b,k), Optimales Integral
    Eigen::Tensor<double, 3> core_as_integrals_two_body_baak; // (a,b,k), Optimales Integral

    //Energies
    double core_total_energy;

    //Refinement
    double highest_error;
    Eigen::MatrixXd LagrangeMultiplier_AS_AS;       
    Eigen::MatrixXd LagrangeMultiplier_AS_Core;

    //Stored AS orbital combinations
    std::vector<real_function_3d> orbs_kl; // |kl>
    std::vector<real_function_3d> coul_orbs_mn; // 1/r|mn>
};