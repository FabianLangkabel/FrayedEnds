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
#include "functionsaver.hpp"
#include "madness_process.hpp"
#include "coulomboperator_nd.hpp"
#include "integrals.hpp"
#include "refinement_utility.hpp"

using namespace madness;
namespace nb = nanobind;
using Numpy2D = nb::ndarray<nb::numpy, double, nb::ndim<2>>;
using Numpy4D = nb::ndarray<nb::numpy, double, nb::ndim<4>>;

template <std::size_t NDIM> class Optimization {
  public:
    Optimization(MadnessProcess<NDIM>& mp);
    ~Optimization();

    refinement_utils::NumericalParameters num_params;
    void override_numerical_parameters(double truncation_tol, double coulomb_lo, double coulomb_eps, double BSH_lo, double BSH_eps) {
        num_params = {truncation_tol, coulomb_lo, coulomb_eps, BSH_lo, BSH_eps};
        Integrator.override_numerical_parameters(num_params);
    }

    std::vector<std::tuple<std::string, double>> get_numerical_parameters() {
        return {std::make_tuple("truncation_tol", num_params.truncation_tol),
                std::make_tuple("coulomb_lo", num_params.coulomb_lo),
                std::make_tuple("coulomb_eps", num_params.coulomb_eps),
                std::make_tuple("BSH_lo", num_params.BSH_lo),
                std::make_tuple("BSH_eps", num_params.BSH_eps),
                std::make_tuple("Integrator truncation_tol", Integrator.num_params.truncation_tol),
                std::make_tuple("Integrator coulomb_lo", Integrator.num_params.coulomb_lo),
                std::make_tuple("Integrator coulomb_eps", Integrator.num_params.coulomb_eps)};
    }

    // input
    void give_initial_orbitals(std::vector<SavedFct<NDIM>> fr_core_orbs, std::vector<SavedFct<NDIM>> act_orbs);
    void give_rdm_and_rotate_orbitals(Numpy2D& one_rdms, Numpy4D& two_rdms);

    // output
    nb::tuple get_effective_hamiltonian(); // returns (core_energy + nuclear repulsion, h_eff_one_body, h_eff_two_body)
    double get_c(); // core_energy + nuclear repulsion
    std::tuple<std::vector<SavedFct<NDIM>>, std::vector<SavedFct<NDIM>>> get_orbitals();

    void give_potential_and_repulsion(SavedFct<NDIM> potential, double nuclear_repulsion);
    void calculate_all_integrals(bool update_aa = true);
    void calculate_energies();
    void calculate_lagrange_multiplier();
    double calculate_lagrange_multiplier_element_as_as(int z, int i);
    double calculate_lagrange_multiplier_element_as_core(int z, int i);
    double calculate_lagrange_multiplier_element_core_core(int z, int c); // Core refinement
    double calculate_lagrange_multiplier_element_core_as(int z, int c); // Core refinement
    bool optimize_orbitals(double optimization_thresh, double NO_occupation_thresh, int maxiter, bool refine_c);
    std::vector<Function<double, NDIM>> get_all_active_orbital_updates(std::vector<int> orbital_indicies_for_update);
    std::vector<Function<double, NDIM>> get_all_core_orbital_updates(); // Core refinement
    void rotate_orbitals_back();

    bool refine_core;

    // Orthonormalization control
    void set_orthonormalization_method(const std::string& method, double degeneracy_tol = 1e-3);
    std::vector<Function<double, NDIM>> orthonormalize_mixed_by_degeneracy(
        std::vector<Function<double, NDIM>>& orbitals); // use integrals stuff

  private:
    MadnessProcess<NDIM>& madness_process;

    Integrals<NDIM> Integrator;
    // Madness + Molecule
    std::vector<std::vector<double>> atoms;
    double nuclear_repulsion_energy = 0.0;
    Function<double, NDIM> Vnuc;

    // Orbitals
    std::vector<Function<double, NDIM>> frozen_occ_orbs;
    std::vector<Function<double, NDIM>> active_orbs;
    int core_dim;
    int as_dim;

    // RDMs
    madness::Tensor<double> ActiveSpaceRotationMatrix;
    madness::Tensor<double> as_one_rdm;
    madness::Tensor<double> as_two_rdm;

    // Integrals
    madness::Tensor<double> as_integrals_one_body; // (k,l)
    madness::Tensor<double> as_integrals_two_body; // (k,l,m,n)

    madness::Tensor<double> core_as_integrals_one_body_ak;   // (a,k)
    madness::Tensor<double> core_as_integrals_two_body_akln; // (a,k,l,n)
    madness::Tensor<double> core_as_integrals_two_body_akal; // (a,k,l)
    madness::Tensor<double> core_as_integrals_two_body_akla; // (a,k,l)
    madness::Tensor<double> core_as_integrals_two_body_abak; // (a,b,k), Optimales Integral
    madness::Tensor<double> core_as_integrals_two_body_baak; // (a,b,k), Optimales Integral
    
    // Integrals only necessary for core refinement
    madness::Tensor<double> core_core_integrals_one_body_ab; // (a,b)
    madness::Tensor<double> core_as_integrals_two_body_baca; // (a,b,c)
    madness::Tensor<double> core_as_integrals_two_body_baac; // (a,b,c)
    madness::Tensor<double> core_as_integrals_two_body_akcl; // (a,k,c,l)
    madness::Tensor<double> core_as_integrals_two_body_aklc; // (a,k,l,c)

    // Energies
    double core_total_energy;

    // AS Refinement
    double highest_as_error;
    madness::Tensor<double> LagrangeMultiplier_AS_AS;
    madness::Tensor<double> LagrangeMultiplier_AS_Core;
    //Core Refinement
    double highest_core_error;
    madness::Tensor<double> LagrangeMultiplier_Core_Core;
    madness::Tensor<double> LagrangeMultiplier_Core_AS;

    // Stored AS orbital combinations
    std::vector<Function<double, NDIM>> orbs_kl;      // |kl>
    std::vector<Function<double, NDIM>> coul_orbs_mn; // 1/r|mn>
    
    // Stored core orbital combinations
    std::vector<Function<double, NDIM>> orbs_aa; // |aa>
    std::vector<Function<double, NDIM>> coul_orbs_aa; // 1/r|aa>

    // Orthonormalization settings
    std::string orthonormalization_method = "symmetric";
    double degeneracy_tolerance = 1e-3;
};