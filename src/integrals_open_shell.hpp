#pragma once
#include "madness_process.hpp"
#include "functionsaver.hpp"
#include "coulomboperator_nd.hpp"
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

template <std::size_t NDIM>
class Integrals_open_shell {
  public:
    Integrals_open_shell(MadnessProcess<NDIM>& mp);
    ~Integrals_open_shell() {};

    madness::Tensor<double> potential_integrals_alpha_alpha;
    madness::Tensor<double> potential_integrals_beta_beta;
    madness::Tensor<double> kinetic_integrals_alpha_alpha;
    madness::Tensor<double> kinetic_integrals_beta_beta;
    madness::Tensor<double> two_body_integrals_alpha_alpha;
    madness::Tensor<double> two_body_integrals_alpha_beta;
    madness::Tensor<double> two_body_integrals_beta_beta;

    std::vector<Numpy2D> compute_potential_integrals(std::vector<SavedFct<NDIM>> alpha_orbs, std::vector<SavedFct<NDIM>> beta_orbs, SavedFct<NDIM> potential);
    std::vector<Numpy2D> compute_kinetic_integrals(std::vector<SavedFct<NDIM>> alpha_orbs, std::vector<SavedFct<NDIM>> beta_orbs);
    std::vector<Numpy4D> compute_two_body_integrals(std::vector<SavedFct<NDIM>> alpha_orbs, std::vector<SavedFct<NDIM>> beta_orbs);

  private:
    MadnessProcess<NDIM>& madness_process;
};