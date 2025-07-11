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

    nb::ndarray<nb::numpy, double, nb::ndim<2> > compute_overlap_integrals(std::vector<SavedFct> all_orbs, std::vector<SavedFct> other);
    nb::ndarray<nb::numpy, double, nb::ndim<2> > compute_potential_integrals(std::vector<SavedFct> all_orbs, SavedFct potential);
    nb::ndarray<nb::numpy, double, nb::ndim<2> > compute_kinetic_integrals(std::vector<SavedFct> all_orbs);
    nb::ndarray<nb::numpy, double, nb::ndim<4> > compute_two_body_integrals(std::vector<SavedFct> all_orbs);

    std::vector<SavedFct> orthonormalize(std::vector<SavedFct> all_orbs, const std::string method, const double rr_thresh);

    std::vector<SavedFct> project_out(std::vector<SavedFct> kernel, std::vector<SavedFct> target);
    std::vector<SavedFct> project_on(std::vector<SavedFct> kernel, std::vector<SavedFct> target);

    std::vector<SavedFct> transform(std::vector<SavedFct> orbitals, nb::ndarray<nb::numpy, double, nb::ndim<2> > matrix){
            std::vector<real_function_3d> x;
            for(SavedFct orb : orbitals) x.push_back(loadfct(orb));

            // @todo there are more efficient ways (flatten and rewire the pointer of the first entry)
            madness::Tensor<double> U(matrix.shape(0), matrix.shape(1));
            for(auto k=0;k<matrix.shape(0);++k){
                for(auto l=0;l<matrix.shape(1);++l){
                    U(k,l) = matrix(k,l);
                }
            }

            auto y = madness::transform(*world, x, U);

            std::vector<SavedFct> result;
            for(size_t k=0; k<orbitals.size(); k++) result.push_back(SavedFct(y[k], orbitals[k].type, orbitals[k].info + " transformed"));
            return result;
    }



    void hello(){std::cout << "hello from the integrals class\n";}
};
