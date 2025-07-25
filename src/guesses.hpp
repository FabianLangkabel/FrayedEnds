#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <madness/mra/mra.h>
#include <madness/mra/function_interface.h>
#include <madness/mra/vmra.h>
#include <madness/tensor/tensor.h>
#include <madness/tensor/tensor_lapack.h>
#include <madness/world/vector.h>
#include <madness/world/world.h>
#include <ostream>
#include <vector>

#include "madness_process.hpp"

using namespace madness;

template <typename T, std::size_t NDIM> class GuessGenerator {
  public:
    class GuessFunctor : public FunctionFunctorInterface<T, NDIM> {
      public:
        GuessFunctor();

        explicit GuessFunctor(const Vector<int, NDIM>& order, Function<T, NDIM>& V) : order(order), V(V) {}

        const Vector<int, NDIM> order;
        Function<T, NDIM> V;

        /// explicit construction
        double operator()(const Vector<T, NDIM>& r) const override {
            double monomial = 1.0;
            for (int dim = 0; dim < NDIM; dim++) {
                monomial *= std::pow(r[dim], order[dim]);
            }
            return monomial * V(r);
        }
    };

    GuessGenerator(World& world) : world(&world) {}

    // Function to create guesses
    std::vector<Function<T, NDIM>> create_guesses(int num, Function<T, NDIM>& V) {
        std::vector<Function<T, NDIM>> guesses;
        int count = 0;
        int order = 0;
        Vector<int, NDIM> orders(0);
        // iterates over the orders of the monomials
        while (count < num) {
            orders.fill(0);    // array to store the orders of the monomials
            orders[0] = order; // set the order of the first monomial
            std::cout << "Order of the first monomial: " << orders[0] << std::endl;
            while (true) {
                GuessFunctor guessfunction(orders, V);
                Function<T, NDIM> guess_function =
                    FunctionFactory<T, NDIM>(*world).functor(guessfunction); // create guess function
                guesses.push_back(guess_function);                           // add guess function to list

                count++;
                std::cout << "Counter: " << count << " and current num: " << num << std::endl;

                // if the number of guesses is reached, return the guesses
                if (count >= num) {
                    std::cout << "Return guesses" << std::endl;
                    return guesses;
                }

                // get the next orders
                // std::cout << "Get the other orders: " << std::endl;

                int first_nonzero = 0; // index of the first non-zero in the array

                std::cout << "First non-zero: " << first_nonzero << std::endl; // print the first non-zero
                std::cout << "order with first non-zero: " << orders[first_nonzero]
                          << std::endl; // print the order with the first non-zero

                bool all_zero = true;

                // check if all orders are zero
                for (int j = 0; j < orders.size(); j++) {
                    if (orders[j] != 0) {
                        all_zero = false;
                        break;
                    }
                }

                // if all orders are zero, break (it's the first guess if the order is 0)
                if (all_zero) {
                    std::cout << "All zero" << std::endl;
                    for (int j = 0; j < orders.size(); j++) {
                        std::cout << orders[j] << " ";
                    }
                    std::cout << std::endl;
                    break;
                }

                // find the first non-zero order
                while (first_nonzero < NDIM && orders[first_nonzero] == 0) {
                    first_nonzero++;
                }
                std::cout << "First non-zero: " << first_nonzero << std::endl;

                // if the first non-zero is the last element, break
                if (first_nonzero >= NDIM - 1) {
                    std::cout << "Break" << std::endl;
                    break;
                }

                orders[first_nonzero] -= 1;     // decrease the order of the first non-zero
                orders[first_nonzero + 1] += 1; // increase the order of the next monomial

                // if the first non-zero is not the first element, swap the first non-zero with the first element
                if (first_nonzero != 0) {
                    orders[0] = orders[first_nonzero];
                    orders[first_nonzero] = 0;
                }

                // print the orders
                for (int j = 0; j < orders.size(); j++) {
                    std::cout << orders[j] << " ";
                }
                std::cout << std::endl;
            }
            ++order;
        }
        return guesses; // return list of guess functions
    }

  private:
    World* world; // pointer to the world
};
