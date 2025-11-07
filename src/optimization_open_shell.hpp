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
#include <utility>
#include <madness/external/nlohmann_json/json.hpp>
#include "npy.hpp"
#include "functionsaver.hpp"
#include "madness_process.hpp"
#include "coulomboperator_nd.hpp"

using namespace madness;

template <std::size_t NDIM>
class Optimization_open_shell {
  public:
    Optimization_open_shell(MadnessProcess<NDIM>& mp);
    ~Optimization_open_shell();

    // input
    void give_initial_orbitals(std::vector<SavedFct<NDIM>> alpha_orbitals, std::vector<SavedFct<NDIM>> beta_orbitals); //Adapted
    void give_rdm_and_rotate_orbitals(std::vector<std::vector<double>> one_rdm_elements, std::vector<std::vector<double>> two_rdm_elements); //Adapted

    // output
    double get_c();
    std::vector<double> get_h_tensor();
    std::vector<double> get_g_tensor();
    std::vector<SavedFct<NDIM>> get_orbitals();

    void give_potential_and_repulsion(SavedFct<NDIM> potential, double nuclear_repulsion);
    void TransformMatrix(madness::Tensor<double>* ObjectMatrix, madness::Tensor<double>& TransformationMatrix);
    void TransformTensor(madness::Tensor<double>& ObjectTensor, madness::Tensor<double>& TransformationMatrix);
    void Transform_ab_mixed_Tensor(madness::Tensor<double>& ObjectTensor, madness::Tensor<double>& TransformationMatrix_alpha, madness::Tensor<double>& TransformationMatrix_beta);
    void calculate_all_integrals();
    void calculate_core_energy();
    void calculate_energies();
    void calculate_lagrange_multiplier();
    double calculate_lagrange_multiplier_element_as_as(int z, int i, int spin);
    double calculate_lagrange_multiplier_element_as_core(int z, int i);
    bool optimize_orbitals(double optimization_thresh, double NO_occupation_thresh, int maxiter);
    std::array<std::vector<Function<double, NDIM>>, 2> get_all_active_orbital_updates(std::array<std::vector<int>, 2> orbital_indicies_for_update);
    void rotate_orbitals_back();
    void save_orbitals(std::string OutputPath);
    void save_effective_hamiltonian(std::string OutputPath);

    // helper
    void sort_eigenpairs_descending(madness::Tensor<double>& eigenvectors, madness::Tensor<double>& eigenvalues);
    madness::Tensor<double> matmul_mxm(const madness::Tensor<double>& A, const madness::Tensor<double>& B,
                                       std::size_t n);

    int nocc = 2; // spatial orbital = 2; spin orbitals = 1
    double truncation_tol = 1e-6;
    double coulomb_lo = 0.001;
    double coulomb_eps = 1e-6;
    double BSH_lo = 0.001;
    double BSH_eps = 1e-6;

  private:
    MadnessProcess<NDIM>& madness_process;

    // Madness + Molecule
    std::vector<std::vector<double>> atoms;
    double nuclear_repulsion_energy = 0.0;
    Function<double, NDIM> Vnuc;

    // Orbitals
    std::array<std::vector<Function<double, NDIM>>, 2> frozen_occ_orbs;
    std::array<std::vector<Function<double, NDIM>>, 2> active_orbs;
    std::array<std::vector<Function<double, NDIM>>, 2> frozen_virt_orbs;
    std::array<int, 2> core_dims;
    std::array<int, 2> as_dims;
    std::array<int, 2> froz_virt_dims;

    // RDMs
    std::array<madness::Tensor<double>, 2> as_one_rdm;
    std::array<madness::Tensor<double>, 3> as_two_rdm; //aaaa, abab, bbbb
    std::array<madness::Tensor<double>, 2> ActiveSpaceRotationMatrices;

    // Integrals
    std::array<madness::Tensor<double>, 2> as_integrals_one_body; // (k,l)
    std::array<madness::Tensor<double>, 3> as_integrals_two_body; // (k,l,m,n)

    madness::Tensor<double> core_as_integrals_one_body_ak;   // (a,k)
    madness::Tensor<double> core_as_integrals_two_body_akln; // (a,k,l,n)
    madness::Tensor<double> core_as_integrals_two_body_akal; // (a,k,l)
    madness::Tensor<double> core_as_integrals_two_body_akla; // (a,k,l)
    madness::Tensor<double> core_as_integrals_two_body_abak; // (a,b,k), Optimales Integral
    madness::Tensor<double> core_as_integrals_two_body_baak; // (a,b,k), Optimales Integral

    // Energies
    double core_total_energy;

    // Refinement
    double highest_error;
    std::array<madness::Tensor<double>, 2> LagrangeMultiplier_AS_AS;
    madness::Tensor<double> LagrangeMultiplier_AS_Core;

    // Stored AS orbital combinations
    //std::vector<Function<double, NDIM>> orbs_kl;      // |kl> //Wird nicht benutzt
    std::array<std::vector<Function<double, NDIM>>, 2> coul_orbs_mn; // 1/r|mn> //alpha-alpha and beta-beta 
};