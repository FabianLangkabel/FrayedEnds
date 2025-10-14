#pragma once
#include "madness_process.hpp"
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

template <std::size_t NDIM> class Integrals {
  public:
    Integrals(MadnessProcess<NDIM>& mp);
    ~Integrals() {};

    madness::Tensor<double> potential_integrals;
    madness::Tensor<double> overlap_integrals;
    madness::Tensor<double> kinetic_integrals;
    madness::Tensor<double> two_body_integrals;
    madness::Tensor<double> frozen_core_interaction;

    nb::ndarray<nb::numpy, double, nb::ndim<2>> compute_overlap_integrals(std::vector<Function<double, NDIM>> all_orbs,
                                                                          std::vector<Function<double, NDIM>> other);
    nb::ndarray<nb::numpy, double, nb::ndim<2>>
    compute_potential_integrals(std::vector<Function<double, NDIM>> all_orbs, Function<double, NDIM> potential);
    nb::ndarray<nb::numpy, double, nb::ndim<2>> compute_kinetic_integrals(std::vector<Function<double, NDIM>> all_orbs);
    nb::ndarray<nb::numpy, double, nb::ndim<4>> compute_two_body_integrals(std::vector<Function<double, NDIM>> all_orbs,
                                                                           double truncation_tol = 1e-6,
                                                                           double coulomb_lo = 0.001,
                                                                           double coulomb_eps = 1e-6, int nocc = 2);
    nb::ndarray<nb::numpy, double, nb::ndim<2>>
    compute_frozen_core_interaction(std::vector<Function<double, NDIM>> fr_c_orbs,
                                    std::vector<Function<double, NDIM>> a_orbs, double truncation_tol = 1e-6,
                                    double coulomb_lo = 0.001, double coulomb_eps = 1e-6, int nocc = 2);

    std::vector<Function<double, NDIM>> orthonormalize(std::vector<Function<double, NDIM>> all_orbs,
                                                       const std::string method, const double rr_thresh);
    std::vector<Function<double, NDIM>> normalize(std::vector<Function<double, NDIM>> all_orbs);

    std::vector<Function<double, NDIM>> project_out(std::vector<Function<double, NDIM>> kernel,
                                                    std::vector<Function<double, NDIM>> target);
    std::vector<Function<double, NDIM>> project_on(std::vector<Function<double, NDIM>> kernel,
                                                   std::vector<Function<double, NDIM>> target);

    std::vector<Function<double, NDIM>> transform(std::vector<Function<double, NDIM>> orbitals,
                                                  nb::ndarray<nb::numpy, double, nb::ndim<2>> matrix);

    void hello() { std::cout << "hello from the integrals class\n"; }

  private:
    MadnessProcess<NDIM>& madness_process;
};