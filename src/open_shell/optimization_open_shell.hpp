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
#include "../npy.hpp"
#include "../functionsaver.hpp"
#include "../madness_process.hpp"
#include "../coulomboperator_nd.hpp"
#include "integrals_open_shell.hpp"
#include "utility_open_shell.hpp"
#include "open_shell_integral_storage.hpp"

using namespace madness;
namespace nb = nanobind;
using Numpy2D = nb::ndarray<nb::numpy, double, nb::ndim<2>>;
using Numpy4D = nb::ndarray<nb::numpy, double, nb::ndim<4>>;

template <std::size_t NDIM>
class Optimization_open_shell {
  public:
    Optimization_open_shell(MadnessProcess<NDIM>& mp);
    ~Optimization_open_shell();

    // input
    void give_initial_orbitals(std::vector<SavedFct<NDIM>> core_alpha_orbitals, std::vector<SavedFct<NDIM>> core_beta_orbitals, std::vector<SavedFct<NDIM>> active_alpha_orbitals, std::vector<SavedFct<NDIM>> active_beta_orbitals);
    void give_rdm_and_rotate_orbitals(std::vector<Numpy2D>& one_rdms, std::vector<Numpy4D>& two_rdms);


    // output
    nb::tuple get_effective_hamiltonian();
    std::vector<std::vector<SavedFct<NDIM>>> get_orbitals();

    void give_potential_and_repulsion(SavedFct<NDIM> potential, double nuclear_repulsion);
    void calculate_all_integrals();
    void calculate_energies();
    void calculate_lagrange_multiplier();
    double calculate_lagrange_multiplier_element_as_as(int z, int i, int spin);
    double calculate_lagrange_multiplier_element_as_core(int z, int i, int spin);
    bool optimize_orbitals(double optimization_thresh, double NO_occupation_thresh, int maxiter, std::string orthonormalization_method);
    std::array<std::vector<Function<double, NDIM>>, 2> get_all_active_orbital_updates(std::array<std::vector<int>, 2> orbital_indicies_for_update);
    void rotate_orbitals_back();


    double truncation_tol = 1e-6;
    double coulomb_lo = 0.001;
    double coulomb_eps = 1e-6;
    double BSH_lo = 0.001;
    double BSH_eps = 1e-6;

  private:
    MadnessProcess<NDIM>& madness_process;

    //Integrator
    Integrals_open_shell<NDIM>* Integrator;
    open_shell_integral_storage<NDIM> Integral_storage;

    // Madness + Molecule
    std::vector<std::vector<double>> atoms;
    double nuclear_repulsion_energy = 0.0;
    Function<double, NDIM> Vnuc;

    // Orbitals
    std::array<std::vector<Function<double, NDIM>>, 2> frozen_occ_orbs;
    std::array<std::vector<Function<double, NDIM>>, 2> active_orbs;
    std::array<int, 2> core_dims;
    std::array<int, 2> as_dims;

    // RDMs
    std::array<madness::Tensor<double>, 2> as_one_rdm;
    std::array<madness::Tensor<double>, 3> as_two_rdm; //aaaa, bbbb, aabb
    std::array<madness::Tensor<double>, 2> ActiveSpaceRotationMatrices;

    // Stored relevant orbital combinations
    std::array<std::vector<Function<double, NDIM>>, 2> orbs_kl;      // |kl> //alpha-alpha and beta-beta 
    std::array<std::vector<Function<double, NDIM>>, 2> coul_orbs_mn; // 1/r|mn> //alpha-alpha and beta-beta 
    std::array<std::vector<Function<double, NDIM>>, 2> orbs_aa;

    // Integrals
    std::array<madness::Tensor<double>, 2> as_integrals_one_body; // (k,l)
    std::array<madness::Tensor<double>, 3> as_integrals_two_body; // (k,l,m,n)

    std::array<madness::Tensor<double>, 2> core_as_integrals_one_body_ak;   // (a,k)
    std::vector<madness::Tensor<double>> core_as_integrals_two_body_akln; // (a,k,l,n)
    std::vector<madness::Tensor<double>> core_as_integrals_two_body_akal; // (a,k,l)
    std::vector<madness::Tensor<double>> core_as_integrals_two_body_akla; // (a,k,l)
    std::vector<madness::Tensor<double>> core_as_integrals_two_body_abak; // (a,b,k)
    std::vector<madness::Tensor<double>> core_as_integrals_two_body_baak; // (a,b,k)
    

    // Energies
    double core_total_energy;

    // Refinement
    double highest_error;
    std::array<madness::Tensor<double>, 2> LagrangeMultiplier_AS_AS;
    std::array<madness::Tensor<double>, 2> LagrangeMultiplier_AS_Core;

};