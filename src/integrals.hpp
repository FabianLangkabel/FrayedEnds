#pragma once
#include "madness_process.hpp"
#include "functionsaver.hpp"
#include "coulomboperator_nd.hpp"
#include "refinement_utility.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

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
namespace nb = nanobind;
using Numpy2D = nb::ndarray<nb::numpy, double, nb::ndim<2>>;
using Numpy4D = nb::ndarray<nb::numpy, double, nb::ndim<4>>;

template <std::size_t NDIM> class Integrals {
  public:
    Integrals(MadnessProcess<NDIM>& mp);
    ~Integrals() {};

    // Numerical parameters
    refinement_utils::NumericalParameters num_params;
    void override_numerical_parameters(refinement_utils::NumericalParameters params) {
        num_params = params;
    }
    void override_numerical_parameters(double truncation_tol, double coulomb_lo, double coulomb_eps) {
        num_params = {truncation_tol, coulomb_lo, coulomb_eps, 0.001, 1e-6}; //BSH parameters irrelevant for integrals class
    }

    std::vector<std::tuple<std::string, double>> get_numerical_parameters() {
        return {std::make_tuple("truncation_tol", num_params.truncation_tol),
                std::make_tuple("coulomb_lo", num_params.coulomb_lo),
                std::make_tuple("coulomb_eps", num_params.coulomb_eps)};
    }

    // Utility functions
    std::vector<Function<double, NDIM>> read_orbitals(std::vector<SavedFct<NDIM>> orbs);
    void update_as_integral_combinations(const std::vector<Function<double, NDIM>> &orbitals, std::vector<Function<double, NDIM>> &orbs_kl, std::vector<Function<double, NDIM>> &coul_orbs_mn);
    void update_core_integral_combinations(const std::vector<Function<double, NDIM>> &core_orbitals, std::vector<Function<double, NDIM>> &orbs_aa);
    void update_core_integral_combinations(const std::vector<Function<double, NDIM>> &core_orbitals, std::vector<Function<double, NDIM>> &orbs_aa, std::vector<Function<double, NDIM>> &coul_orbs_aa);
    
    // Different one and two body operators
    Numpy2D nb_one_body_op_integrals(std::string op_name, const std::vector<SavedFct<NDIM>>& orbitals);

    // Nanobind bindings for Integrators
    Numpy2D nb_compute_overlap_integrals(const std::vector<SavedFct<NDIM>>& all_orbs, const std::vector<SavedFct<NDIM>>& other);
    Numpy2D nb_compute_potential_integrals(const std::vector<SavedFct<NDIM>>& all_orbs, const SavedFct<NDIM>& potential);
    Numpy2D nb_compute_kinetic_integrals(const std::vector<SavedFct<NDIM>>& all_orbs);
    Numpy4D nb_compute_two_body_integrals(const std::vector<SavedFct<NDIM>>& all_orbs);
    Numpy2D nb_compute_frozen_core_interaction(const std::vector<SavedFct<NDIM>>& fr_c_orbs, const std::vector<SavedFct<NDIM>>& a_orbs);
    nb::tuple nb_compute_effective_hamiltonian(const std::vector<SavedFct<NDIM>>& core_orbitals, const std::vector<SavedFct<NDIM>>& active_orbitals, const SavedFct<NDIM>& potential, double energy_offset);
    
    // Integrators
    Tensor<double> compute_potential_integrals(const std::vector<Function<double, NDIM>>& orbitals,const Function<double, NDIM>& V);
    Tensor<double> compute_kinetic_integrals(const std::vector<Function<double, NDIM>>& orbitals);
    Tensor<double> compute_two_body_integrals(const std::vector<Function<double, NDIM>> &orbitals, const std::vector<Function<double, NDIM>> &orbs_kl, const std::vector<Function<double, NDIM>> &coul_orbs_mn);
    double compute_core_energy(const std::vector<Function<double, NDIM>>& core_orbitals, std::vector<Function<double, NDIM>>& orbs_aa, std::vector<Function<double, NDIM>>& coul_orbs_aa, const Function<double, NDIM>& V, double energy_offset);
    Tensor<double> compute_core_as_integrals_one_body(const std::vector<Function<double, NDIM>>& core_orbitals, const std::vector<Function<double, NDIM>>& active_orbitals, const Function<double, NDIM>& V);
    
    std::array<Tensor<double>, 2> compute_core_as_2e_integrals_energy(
        const std::vector<Function<double, NDIM>> &core_orbitals, 
        const std::vector<Function<double, NDIM>> &active_orbitals, 
        const std::vector<Function<double, NDIM>> &orbs_kl, 
        const std::vector<Function<double, NDIM>> &coul_orbs_mn, 
        const std::vector<Function<double, NDIM>> &orbs_aa
    );

    std::array<Tensor<double>, 5> compute_core_as_2e_integrals_as_refinement(
        const std::vector<Function<double, NDIM>> &core_orbitals, 
        const std::vector<Function<double, NDIM>> &active_orbitals, 
        const std::vector<Function<double, NDIM>> &orbs_kl, 
        const std::vector<Function<double, NDIM>> &coul_orbs_mn, 
        const std::vector<Function<double, NDIM>> &orbs_aa
    );

    std::array<Tensor<double>, 4> compute_core_as_2e_integrals_core_refinement(
        const std::vector<Function<double, NDIM>> &core_orbitals, 
        const std::vector<Function<double, NDIM>> &active_orbitals, 
        const std::vector<Function<double, NDIM>> &orbs_kl, 
        const std::vector<Function<double, NDIM>> &coul_orbs_mn, 
        const std::vector<Function<double, NDIM>> &orbs_aa,
        const std::vector<Function<double, NDIM>> &coul_orbs_aa
    );

    // Orthonormalization and related utilities
    std::vector<SavedFct<NDIM>> orthonormalize(std::vector<SavedFct<NDIM>> all_orbs, const std::string method,
                                               const double rr_thresh = 0.0,
                                               nb::ndarray<nb::numpy, double, nb::ndim<1>> occupations = {},
                                               double degeneracy_tol = 1e-6);
    std::vector<SavedFct<NDIM>> normalize(std::vector<SavedFct<NDIM>> all_orbs);
    std::vector<SavedFct<NDIM>> project_out(std::vector<SavedFct<NDIM>> kernel, std::vector<SavedFct<NDIM>> target);
    std::vector<SavedFct<NDIM>> project_on(std::vector<SavedFct<NDIM>> kernel, std::vector<SavedFct<NDIM>> target);
    std::vector<SavedFct<NDIM>> transform(std::vector<SavedFct<NDIM>> orbitals, Numpy2D matrix);
    SavedFct<NDIM> compute_electron_density(std::vector<SavedFct<NDIM>> core_orbitals, std::vector<SavedFct<NDIM>> active_orbitals, Numpy2D rdm1);

    // Helper method for mixed orthonormalization using occupations
    std::vector<Function<double, NDIM>> orthonormalize_mixed_by_degeneracy(
        std::vector<Function<double, NDIM>>& orbitals,
        const std::vector<double>& occupations,
        double degeneracy_tol);

  private:
    MadnessProcess<NDIM>& madness_process;
};