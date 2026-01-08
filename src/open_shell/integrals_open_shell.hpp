#pragma once
#include "../madness_process.hpp"
#include "../functionsaver.hpp"
#include "../coulomboperator_nd.hpp"
#include "../madmolecule.hpp"
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
#include <memory>
#include <nanobind/nanobind.h>

using namespace madness;
namespace nb = nanobind;
using Numpy2D = nb::ndarray<nb::numpy, double, nb::ndim<2>>;
using Numpy4D = nb::ndarray<nb::numpy, double, nb::ndim<4>>;

template <std::size_t NDIM>
class Integrals_open_shell {
  public:
    Integrals_open_shell(MadnessProcess<NDIM>& mp);
    ~Integrals_open_shell() {};

    // Utility Functions
    std::array<std::vector<Function<double, NDIM>>, 2> read_orbitals(std::vector<SavedFct<NDIM>> alpha_orbs, std::vector<SavedFct<NDIM>> beta_orbs);
    void update_as_integral_combinations(std::array<std::vector<Function<double, NDIM>>, 2> &orbitals, std::array<std::vector<Function<double, NDIM>>, 2> &orbs_kl, std::array<std::vector<Function<double, NDIM>>, 2> &coul_orbs_mn);
    void update_core_integral_combinations(std::array<std::vector<Function<double, NDIM>>, 2> &core_orbitals, std::array<std::vector<Function<double, NDIM>>, 2> &orbs_aa);


    // Nanobind bindings
    std::vector<Numpy2D> nb_compute_potential_integrals(std::vector<SavedFct<NDIM>> alpha_orbs, std::vector<SavedFct<NDIM>> beta_orbs, SavedFct<NDIM> potential);
    std::vector<Numpy2D> nb_compute_kinetic_integrals(std::vector<SavedFct<NDIM>> alpha_orbs, std::vector<SavedFct<NDIM>> beta_orbs);
    std::vector<Numpy4D> nb_compute_two_body_integrals(std::vector<SavedFct<NDIM>> alpha_orbs, std::vector<SavedFct<NDIM>> beta_orbs);
    nb::tuple nb_compute_effective_hamiltonian(std::vector<SavedFct<NDIM>> core_alpha_orbitals, std::vector<SavedFct<NDIM>> core_beta_orbitals, std::vector<SavedFct<NDIM>> active_alpha_orbitals, std::vector<SavedFct<NDIM>> active_beta_orbitals, SavedFct<NDIM> potential, double energy_offset);

    // Integrators
    std::array<madness::Tensor<double>, 2> compute_potential_integrals(std::array<std::vector<Function<double, NDIM>>, 2> orbitals, Function<double, NDIM> V);
    std::array<madness::Tensor<double>, 2> compute_kinetic_integrals(std::array<std::vector<Function<double, NDIM>>, 2> orbitals);
    std::array<madness::Tensor<double>, 3> compute_two_body_integrals(std::array<std::vector<Function<double, NDIM>>, 2> &orbitals, std::array<std::vector<Function<double, NDIM>>, 2> &orbs_kl, std::array<std::vector<Function<double, NDIM>>, 2> &coul_orbs_mn);
    double compute_core_energy(std::array<std::vector<Function<double, NDIM>>, 2> core_orbitals, Function<double, NDIM> V, double energy_offset);
    std::array<madness::Tensor<double>, 2> compute_core_as_integrals_one_body(
      std::array<std::vector<Function<double, NDIM>>, 2> core_orbitals, std::array<std::vector<Function<double, NDIM>>, 2> active_orbitals, Function<double, NDIM> V
    );
    std::vector<std::vector<madness::Tensor<double>>> compute_core_as_integrals_two_body(
      std::array<std::vector<Function<double, NDIM>>, 2> &core_orbitals, 
      std::array<std::vector<Function<double, NDIM>>, 2> &active_orbitals, 
      std::array<std::vector<Function<double, NDIM>>, 2> &orbs_kl, 
      std::array<std::vector<Function<double, NDIM>>, 2> &coul_orbs_mn, 
      std::array<std::vector<Function<double, NDIM>>, 2> &orbs_aa, 
      bool calc_akal, bool calc_akla, bool calc_akln, bool calc_abak, bool calc_baak
    );

  private:
    MadnessProcess<NDIM>& madness_process;
};