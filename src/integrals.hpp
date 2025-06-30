#pragma once
#include "MadnessProcess.hpp"
#include "functionsaver.hpp"
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

class Integrals: public MadnessProcess{
public:
    Integrals(double L, long k, double thresh, int initial_level, int truncate_mode, bool refine, int n_threads);
    ~Integrals(){};

    madness::Tensor<double> potential_integrals;
    madness::Tensor<double> overlap_integrals;
    madness::Tensor<double> kinetic_integrals;
    madness::Tensor<double> two_body_integrals;

    nb::ndarray<nb::numpy, double, nb::ndim<2> > compute_overlap_integrals(std::vector<SavedFct> all_orbs);
    nb::ndarray<nb::numpy, double, nb::ndim<2> > compute_potential_integrals(std::vector<SavedFct> all_orbs, SavedFct potential);
    nb::ndarray<nb::numpy, double, nb::ndim<2> > compute_kinetic_integrals(std::vector<SavedFct> all_orbs);
    nb::ndarray<nb::numpy, double, nb::ndim<4> > compute_two_body_integrals(std::vector<SavedFct> all_orbs);

    void hello(){std::cout << "hello from the integrals class\n";}
};