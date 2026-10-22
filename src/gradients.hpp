#pragma once

#include "integrals.hpp"
#include "moleculargeometry.hpp"

namespace nb = nanobind;
using Numpy2D = nb::ndarray<nb::numpy, double, nb::ndim<2>>;
using Numpy4D = nb::ndarray<nb::numpy, double, nb::ndim<4>>;

template <std::size_t NDIM>
struct GradientResponse {
    std::vector<SavedFct<NDIM>> orbital_derivatives;
    std::vector<SavedFct<NDIM>> dRdphi_E;
    double parameter_derivative = 0.0;
};

template <std::size_t NDIM>
class Gradients {
  public:
    explicit Gradients(MadnessProcess<NDIM>& mp);
    ~Gradients();

    refinement_utils::NumericalParameters num_params;

    void override_numerical_parameters(double truncation_tol, double coulomb_lo, double coulomb_eps, double BSH_lo, double BSH_eps) {
        num_params = {truncation_tol, coulomb_lo, coulomb_eps, BSH_lo, BSH_eps};
        Integrator.override_numerical_parameters(num_params);
    }
    
    void calc_dR_manybody_wf();
    void calc_dR_rdms();
    void calc_dR_lagrange_mult();
    std::vector<Function<double, NDIM>> calc_dR_act_orb_updates();

    GradientResponse<NDIM> solve_h2(const std::vector<SavedFct<NDIM>>& optimized_orbitals, const Numpy2D& rdm1,
                              const Numpy4D& rdm2, const Numpy2D& lagrange_multipliers, MolecularGeometry& geometry,
                              const Numpy2D& d_rdm1_dz, const Numpy4D& d_rdm2_dz, const Numpy2D& d2_rdm1_dz2,
                              const Numpy4D& d2_rdm2_dz2, int max_iterations = 20);

  private:  
    MadnessProcess<NDIM>& madness_process;
    Integrals<NDIM> Integrator;
    double highest_as_error = 0.0;
    
    // Orbitals
    std::vector<Function<double, NDIM>> active_orbs;
    std::vector<Function<double, NDIM>> dR_active_orbs;
    std::vector<Function<double, NDIM>> orbs_kl;
    std::vector<Function<double, NDIM>> coul_orbs_mn;
    std::vector<Function<double, NDIM>> V_active_orbs;
    std::vector<Function<double, NDIM>> dR_V_active_orbs;
    std::vector<Function<double, NDIM>> coul_response_pairs;
    int as_dim;

    // Many-body wavefunction
    double dR_z;

    // RDMs
    madness::Tensor<double> as_one_rdm;
    madness::Tensor<double> as_two_rdm;
    madness::Tensor<double> dz_as_one_rdm;
    madness::Tensor<double> dz_as_two_rdm;
    madness::Tensor<double> dR_as_one_rdm;
    madness::Tensor<double> dR_as_two_rdm;
    madness::Tensor<double> dz2_as_one_rdm;
    madness::Tensor<double> dz2_as_two_rdm;

    // Integrals
    madness::Tensor<double> as_integrals_one_body; // (k,l)
    madness::Tensor<double> as_integrals_two_body; // (k,l,m,n)
    madness::Tensor<double> dR_as_integrals_one_body; // (k,l) = <dR_k|h|l> + <k|h|dR_l> + <k|dR_V|l>
    madness::Tensor<double> dR_as_integrals_one_body1; // <dR_k|h|l>
    madness::Tensor<double> dR_as_integrals_one_body2; // <k|h|dR_l> + <k|dR_V|l>
    madness::Tensor<double> dR_as_integrals_two_body; // (k,l,m,n)

    // Lagrange multipiers
    madness::Tensor<double> LagrangeMultiplier_AS_AS;
    madness::Tensor<double> dR_LagrangeMultiplier_AS_AS;
    
    // Potential
    madness::Function<double, NDIM> V;
    madness::Function<double, NDIM> dR_V;

};
