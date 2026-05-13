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
        if (Integrator) {
            Integrator->override_numerical_parameters(num_params);
        }
    }

    // input
    void give_initial_orbitals(std::vector<SavedFct<NDIM>> fr_core_orbs, std::vector<SavedFct<NDIM>> act_orbs);
    void give_rdm_and_rotate_orbitals(std::vector<double> one_rdm_elements, std::vector<double> two_rdm_elements);

    // output
    double get_c();
    std::vector<double> get_h_tensor();
    std::vector<double> get_g_tensor();
    nb::tuple get_effective_hamiltonian();
    std::tuple<std::vector<SavedFct<NDIM>>, std::vector<SavedFct<NDIM>>> get_orbitals();

    void give_potential_and_repulsion(SavedFct<NDIM> potential, double nuclear_repulsion);
    void calculate_all_integrals();
    void calculate_core_energy();
    void calculate_energies();
    void calculate_lagrange_multiplier();
    double calculate_lagrange_multiplier_element_as_as(int z, int i);
    double calculate_lagrange_multiplier_element_as_core(int z, int i);
    // core lagrange
    bool optimize_orbitals(double optimization_thresh, double NO_occupation_thresh, int maxiter);
    std::vector<Function<double, NDIM>> get_all_active_orbital_updates(std::vector<int> orbital_indicies_for_update);
    void rotate_orbitals_back();
    // core stuff missing of course
    
    bool has_core_orbitals;
    bool refine_core;

    // Orthonormalization control
    void set_orthonormalization_method(const std::string& method, double degeneracy_tol = 1e-3);
    std::vector<Function<double, NDIM>> orthonormalize_mixed_by_degeneracy(
        std::vector<Function<double, NDIM>>& orbitals); // use integrals stuff

    // helper
    void sort_eigenpairs_descending(madness::Tensor<double>& eigenvectors, madness::Tensor<double>& eigenvalues);

    int nocc = 2; // spatial orbital = 2; spin orbitals = 1
    double truncation_tol = 1e-6;
    double coulomb_lo = 0.001;
    double coulomb_eps = 1e-6;
    double BSH_lo = 0.001;
    double BSH_eps = 1e-6;

  private:
    MadnessProcess<NDIM>& madness_process;

    Integrals<NDIM>* Integrator;
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

    // Energies
    double core_total_energy;

    // Refinement
    double highest_error;
    madness::Tensor<double> LagrangeMultiplier_AS_AS;
    madness::Tensor<double> LagrangeMultiplier_AS_Core;
    // core ref lm

    // Stored AS orbital combinations
    std::vector<Function<double, NDIM>> orbs_kl;      // |kl>
    std::vector<Function<double, NDIM>> coul_orbs_mn; // 1/r|mn>
    std::vector<Function<double, NDIM>> orbs_aa;

    // Orthonormalization settings
    std::string orthonormalization_method = "symmetric";
    double degeneracy_tolerance = 1e-3;
};