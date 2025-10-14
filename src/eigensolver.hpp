#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <madness/mra/mra.h>
#include <madness/mra/function_interface.h>
#include <madness/mra/nonlinsol.h>
#include <madness/mra/vmra.h>
#include <madness/tensor/tensor.h>
#include <madness/tensor/tensor_lapack.h>
#include <madness/world/vector.h>
#include <madness/world/world.h>
#include <madness/mra/derivative.h>
#include <ostream>
#include <vector>
#include "guesses.hpp"
#include "madness_process.hpp"

using namespace madness;

template <std::size_t NDIM> class Eigensolver {
  public:
    Eigensolver(MadnessProcess<NDIM>& mp);

    ~Eigensolver();

    // Function to solve the eigenvalue problem for the given potential
    void solve(Function<double, NDIM> input_V, int num_levels, int max_iter);

    // Function to solve the eigenvalue problem for the given potential with given guesses
    std::vector<Function<double, NDIM>>
    solve_with_input_guesses(Function<double, NDIM> input_V, const std::vector<Function<double, NDIM>>& input_guesses,
                             int num_levels, int max_iter);

    // Function to calculate the energy
    double energy(const Function<double, NDIM>& phi, const Function<double, NDIM>& V);
    std::vector<Function<double, NDIM>> get_orbitals() const;

  private:
    MadnessProcess<NDIM>& madness_process;

    Function<double, NDIM> V;
    std::vector<Function<double, NDIM>> orbitals;

    // Function to calculate the Hamiltonian matrix, Overlap matrix and Diagonal matrix
    std::pair<Tensor<double>, std::vector<Function<double, NDIM>>>
    diagonalize(const std::vector<Function<double, NDIM>>& functions, const Function<double, NDIM>& V);

    // Function to optimize the eigenfunction for each energy level
    Function<double, NDIM> optimize(Function<double, NDIM>& V, const Function<double, NDIM> guess_function, int N,
                                    const std::vector<Function<double, NDIM>>& prev_phi, int max_iter);
};
