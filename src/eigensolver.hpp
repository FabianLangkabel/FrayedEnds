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
#include <ostream>
#include <vector>
#include "guesses.h"
#include "plot.h"
#include "MadnessProcess.hpp"
#include "functionsaver.hpp"

using namespace madness;

class Eigensolver3D: public MadnessProcess {
    public:     
        Eigensolver(double L, long k, double thresh);

        ~Eigensolver();

        // Function to solve the eigenvalue problem for the given potential
        std::vector<Function<double, 3>> solve(Function<double, 3>& V, int num_levels, int max_iter);

        // Function to solve the eigenvalue problem for the given potential with given guesses
        std::vector<Function<double, 3>> solve(Function<double, 3>& V, const std::vector<Function<double, 3>>& guesses, int num_levels, int max_iter);

        // Function to calculate the energy
        double energy(World& world, const Function<double, 3>& phi, const Function<double, 3>& V);

    private:
        Function<double, 3> V;
        std::vector<Function<double, 3>> orbitals;

        // Function to calculate the Hamiltonian matrix, Overlap matrix and Diagonal matrix
        std::pair<Tensor<double>, std::vector<Function<double, 3>>> diagonalize(World &world, const std::vector<Function<double, 3>>& functions, const Function<double, 3>& V);

        // Function to optimize the eigenfunction for each energy level
        Function<double, 3> optimize(World& world, Function<double, 3>& V, const Function<double, 3> guess_function, int N, const std::vector<Function<double, 3>>& prev_phi, int max_iter);
};



