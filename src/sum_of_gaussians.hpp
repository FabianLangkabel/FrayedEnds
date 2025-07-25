#include <madness/mra/mra.h>
#include <madness/mra/operator.h>
#include "functionsaver.hpp"
#include "madness_process.hpp"

using namespace madness;

// class to create a charge density which can be represented by a sum of gaussian funcions (rho=\sum
// exp(-a_i*(r-r_i)^2))
class SumOfGaussians : public FunctionFunctorInterface<double, 3> {
  public:
    std::vector<double> sharpness_list; // list of prefactors a_i
    double Q;                           // overall charge (rho is later renormalized such that Q=\int dV rho)
    std::vector<Vector<double, 3>> charge_locations; // locations of the peaks (r_i)

    SumOfGaussians(std::vector<double> sl, double Q, std::vector<std::vector<double>> cl) : sharpness_list(sl), Q(Q) {
        for (int i = 0; i < cl.size(); i++) {
            charge_locations.push_back(Vector<double, 3>(cl[i]));
        }
    }

    double operator()(const Vector<double, 3>& r) const override {
        double result = 0.0;
        for (int i = 0; i < charge_locations.size(); i++) {
            result += exp(-sharpness_list[i] * std::pow((r - charge_locations[i]).normf(), 2));
        }
        return result;
    }
};

Function<double, 3> make_potential(World& world, std::vector<double> sl, double Q,
                                   std::vector<std::vector<double>> cl) {
    SumOfGaussians Rho(sl, Q, cl); // charge density
    Function<double, 3> f =
        FunctionFactory<double, 3>(world).special_level(10).special_points(Rho.charge_locations).functor(Rho);
    double norm = f.trace();
    f = Rho.Q / norm * f; // renormalize such that Q=\int dV rho

    SeparatedConvolution<double, 3> op = BSHOperator<3>(world, 0.0, 0.001, 1e-6); // convolution operator
    auto V = op(f);                                                               // potential
    V = V.truncate();
    return -4.0 * constants::pi * V; // conversion into atomic units
}

class CoulombPotentialFromChargeDensity {
  public:
    std::vector<double> sharpness_list;
    double Q;
    std::vector<std::vector<double>> charge_locations;

    CoulombPotentialFromChargeDensity(MadnessProcess& mp, std::vector<double> sl, double Q,
                                      std::vector<std::vector<double>> cl)
        : madness_process(mp), sharpness_list(sl), Q(Q), charge_locations(cl) {
        std::cout.precision(6);
    }

    ~CoulombPotentialFromChargeDensity() {}
    SavedFct create_charge_density() {
        SumOfGaussians Rho(sharpness_list, Q, charge_locations); // charge density
        Function<double, 3> f = FunctionFactory<double, 3>(*(madness_process.world))
                                    .special_level(10)
                                    .special_points(Rho.charge_locations)
                                    .functor(Rho);
        double norm = f.trace();
        f = Rho.Q / norm * f; // renormalize such that Q=\int dV rho
        return SavedFct(f);
    }
    SavedFct create_potential() {
        Function<double, 3> Vnuc = make_potential(*(madness_process.world), sharpness_list, Q, charge_locations);
        return SavedFct(Vnuc);
    }

  private:
    MadnessProcess& madness_process;
};
