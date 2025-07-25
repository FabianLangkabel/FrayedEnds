#pragma once

#include "functionsaver.hpp"
#include <madness/mra/vmra.h>
#include <string>
#include <vector>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "madness_process.hpp"

class Transformator {
  public:
    double thresh;

    Transformator(MadnessProcess& mp) : madness_process(mp), thresh(mp.thresh) {}
    ~Transformator() {}

    std::vector<SavedFct> orthorgonalize(const std::vector<SavedFct>& input_functions, const std::string& technique,
                                         const bool print_out = false) {
        std::vector<SavedFct> transformed_functions;

        std::vector<real_function_3d> basis;
        for (SavedFct fct : input_functions)
            basis.push_back(loadfct(fct));

        auto S = madness::matrix_inner(*(madness_process.world), basis, basis, true);

        auto out_basis = basis;
        if (orthogonalization == "cholesky") {
            out_basis = madness::orthonormalize_cd(basis, S);
        } else if (orthogonalization == "symmetric") {
            out_basis = madness::orthonormalize_symmetric(basis, S);
        } else if (orthogonalization == "canonical") {
            out_basis = madness::orthonormalize_canonical(basis, S, thresh);
        } else if (orthogonalization == "rr_cholesky") {
            if (*(madness_process.world).rank() == 0)
                std::cout << std::endl << "Be cautious:" << std::endl;
            if (*(madness_process.world).rank() == 0)
                std::cout << "Rank reduced cholesky changes ordering in basis via pivoting. Make sure, that this does "
                             "not interfere with your application (e.g. active-space)."
                          << std::endl;
            out_basis = madness::orthonormalize_rrcd(basis, S, thresh);
        } else {
            MADNESS_EXCEPTION("unknown orthonormalization method", 1);
        }

        if (print_out) {
            const auto new_S = madness::matrix_inner(*(madness_process.world), out_basis, out_basis);
            if (*(madness_process.world).rank() == 0) {
                std::cout << "Overlap Matrix before " << orthogonalization << "\n";
                std::cout << S;
                std::cout << "Overlap Matrix after " << orthogonalization << "\n";
                std::cout << new_S;
            }
        }
        for (Function<double, 3> fct : out_basis) {
            transformed_functions.push_back(SavedFct(fct));
        }

        return transformed_functions;
    }

  private:
    MadnessProcess& madness_process;
}
