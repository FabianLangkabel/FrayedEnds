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

template <std::size_t NDIM>
class Integrals {
  public:
    Integrals(MadnessProcess<NDIM>& mp);
    ~Integrals() {};

    madness::Tensor<double> potential_integrals;
    madness::Tensor<double> overlap_integrals;
    madness::Tensor<double> kinetic_integrals;
    madness::Tensor<double> two_body_integrals;

    nb::ndarray<nb::numpy, double, nb::ndim<2>> compute_overlap_integrals(std::vector<SavedFct<NDIM>> all_orbs,
                                                                          std::vector<SavedFct<NDIM>> other);
    nb::ndarray<nb::numpy, double, nb::ndim<2>> compute_potential_integrals(std::vector<SavedFct<NDIM>> all_orbs,
                                                                            SavedFct<NDIM> potential);
    nb::ndarray<nb::numpy, double, nb::ndim<2>> compute_kinetic_integrals(std::vector<SavedFct<NDIM>> all_orbs);
    nb::ndarray<nb::numpy, double, nb::ndim<4>> compute_two_body_integrals(std::vector<SavedFct<NDIM>> all_orbs);

    std::vector<SavedFct<NDIM>> orthonormalize(std::vector<SavedFct<NDIM>> all_orbs, const std::string method,
                                         const double rr_thresh);
    std::vector<SavedFct<NDIM>> normalize(std::vector<SavedFct<NDIM>> all_orbs);

    std::vector<SavedFct<NDIM>> project_out(std::vector<SavedFct<NDIM>> kernel, std::vector<SavedFct<NDIM>> target);
    std::vector<SavedFct<NDIM>> project_on(std::vector<SavedFct<NDIM>> kernel, std::vector<SavedFct<NDIM>> target);

    std::vector<SavedFct<NDIM>> transform(std::vector<SavedFct<NDIM>> orbitals,
                                    nb::ndarray<nb::numpy, double, nb::ndim<2>> matrix) {
        std::vector<Function<double, NDIM>> x;
        for (SavedFct<NDIM> orb : orbitals)
            x.push_back(madness_process.loadfct(orb));

        // @todo there are more efficient ways (flatten and rewire the pointer of the first entry)
        madness::Tensor<double> U(matrix.shape(0), matrix.shape(1));
        for (auto k = 0; k < matrix.shape(0); ++k) {
            for (auto l = 0; l < matrix.shape(1); ++l) {
                U(k, l) = matrix(k, l);
            }
        }

        auto y = madness::transform(*(madness_process.world), x, U);

        std::vector<SavedFct<NDIM>> result;
        for (size_t k = 0; k < orbitals.size(); k++)
            result.push_back(SavedFct<NDIM>(y[k], orbitals[k].type, orbitals[k].info + " transformed "));
        return result;
    }

    void hello() { std::cout << "hello from the integrals class\n"; }

  private:
    MadnessProcess<NDIM>& madness_process;
};