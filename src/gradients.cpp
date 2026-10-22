#include "gradients.hpp"

#include "coulomboperator_nd.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>

using namespace madness;

template <std::size_t NDIM>
Gradients<NDIM>::Gradients(MadnessProcess<NDIM>& mp) : madness_process(mp), Integrator(mp) 
{
    std::cout.precision(6);
    Integrator.override_numerical_parameters(num_params);
}

template <std::size_t NDIM>
Gradients<NDIM>::~Gradients() 
{
    V.clear();
    dR_V.clear();
    active_orbs.clear();
    dR_active_orbs.clear();
    orbs_kl.clear();
    coul_orbs_mn.clear();
    V_active_orbs.clear();
    dR_V_active_orbs.clear();
    coul_response_pairs.clear();
}

template <std::size_t NDIM>
void Gradients<NDIM>::calc_dR_manybody_wf() 
{
    double dz2_E = refinement_utils::contract<2>(
        {as_dim, as_dim}, 
        [&](const auto& i) {return dz2_as_one_rdm(i[0], i[1]);},
        [&](const auto& i) {return as_integrals_one_body(i[0], i[1]);}
    );
    dz2_E += 0.5 * refinement_utils::contract<4>(
        {as_dim, as_dim, as_dim, as_dim}, 
        [&](const auto& i) {return dz2_as_two_rdm(i[0], i[1], i[2], i[3]);},
        [&](const auto& i) {return as_integrals_two_body(i[0], i[1], i[2], i[3]);}
    );
    double dzdR_E = refinement_utils::contract<2>(
        {as_dim, as_dim}, 
        [&](const auto& i) {return dz_as_one_rdm(i[0], i[1]);},
        [&](const auto& i) {return dR_as_integrals_one_body(i[0], i[1]);}
    );
    dzdR_E += 0.5 * refinement_utils::contract<4>(
        {as_dim, as_dim, as_dim, as_dim},
        [&](const auto& i) {return dz_as_two_rdm(i[0], i[1], i[2], i[3]);},
        [&](const auto& i) {return dR_as_integrals_two_body(i[0], i[1], i[2], i[3]);}
    );

    dR_z = -dzdR_E / dz2_E;
}

template <std::size_t NDIM>
void Gradients<NDIM>::calc_dR_rdms()
{
    dR_as_one_rdm = dR_z * dz_as_one_rdm;
    dR_as_two_rdm = dR_z * dz_as_two_rdm;
}



template <std::size_t NDIM>
void Gradients<NDIM>::calc_dR_lagrange_mult()
{
    dR_LagrangeMultiplier_AS_AS = inner(as_integrals_one_body, dR_as_one_rdm);
    dR_LagrangeMultiplier_AS_AS += inner(dR_as_integrals_one_body2, as_one_rdm);
    dR_LagrangeMultiplier_AS_AS -= inner(matrix_inner(*(madness_process.world), active_orbs, dR_active_orbs), LagrangeMultiplier_AS_AS);

    std::vector<Function<double, NDIM> > dR_k_l; // todo for optimization: construct this with response_pairs
    for (int i = 0; i < as_dim; i++) {
        std::vector<Function<double, NDIM>> pair_ij = dR_active_orbs[i] * active_orbs;
        dR_k_l.insert(std::end(dR_k_l), std::begin(pair_ij), std::end(pair_ij));
    }
    dR_k_l = truncate(dR_k_l, num_params.truncation_tol);
    
    Tensor<double> dR_as_ints_2e_lagrange = dR_as_integrals_two_body - Integrator.compute_two_body_integrals(active_orbs, dR_k_l, coul_orbs_mn);
    dR_as_ints_2e_lagrange = dR_as_ints_2e_lagrange.reshape(as_dim, as_dim*as_dim*as_dim); 
    Tensor<double> as_two_rdm_rs = as_two_rdm.reshape(as_dim*as_dim*as_dim, as_dim); // can we do this without copying and does it even matter?
    
    dR_LagrangeMultiplier_AS_AS += inner(dR_as_ints_2e_lagrange, as_two_rdm_rs);

    Tensor<double> as_integrals_two_body_rs = as_integrals_two_body.reshape(as_dim, as_dim*as_dim*as_dim);
    Tensor<double> dR_as_two_rdm_rs = dR_as_two_rdm.reshape(as_dim*as_dim*as_dim, as_dim);
    dR_LagrangeMultiplier_AS_AS += inner(as_integrals_two_body_rs, dR_as_two_rdm_rs);
}

template <std::size_t NDIM>
std::vector<Function<double, NDIM>> Gradients<NDIM>::calc_dR_act_orb_updates() 
{
    // V*dR_phi_k 
    std::vector<Function<double, NDIM>> V_dR_active_orbs = V * dR_active_orbs;
    V_dR_active_orbs = truncate(V_dR_active_orbs, num_params.truncation_tol);
    
    std::vector<Function<double, NDIM>> dR_orbital_updates;
    for (int i = 0; i < as_dim; i++) {
        double en = LagrangeMultiplier_AS_AS(i, i)/as_one_rdm(i, i);
        // AS Part
        Function<double, NDIM> rhs = as_one_rdm(i, i) * V_dR_active_orbs[i];
        for (int k = 0; k <as_dim; k++) {
            rhs += dR_as_one_rdm(k, i) * (V_active_orbs[k] + en*active_orbs[k]) + as_one_rdm(k, i) * dR_V_active_orbs[k];
            if (k != i) {
                rhs += as_one_rdm(k, i) * (V_dR_active_orbs[k]+en*dR_active_orbs[k]);
            }
        }
        // Lagrange multiplier part
        for (int k = 0; k <as_dim; k++) {
            rhs -= dR_LagrangeMultiplier_AS_AS(k, i) * active_orbs[k];
            if (k != i) {
                rhs -= LagrangeMultiplier_AS_AS(k, i) * dR_active_orbs[k];
            }
        }
        rhs.truncate(num_params.truncation_tol);
        // Coulomb part
        for (int k = 0; k < as_dim; k++) { // this is inefficient, since lnk needs to be constructed for every orbital i instead of just once!!!
            std::vector<Function<double, NDIM>> ln_dR_k = coul_orbs_mn * dR_active_orbs[k];
            ln_dR_k = truncate(ln_dR_k, num_params.truncation_tol);
            std::vector<Function<double, NDIM>> lnk = coul_orbs_mn * active_orbs[k]; // todo: cache this?
            lnk = truncate(lnk, num_params.truncation_tol);
            std::vector<Function<double, NDIM>> dR_ln_k = coul_response_pairs * active_orbs[k];
            dR_ln_k = truncate(dR_ln_k, num_params.truncation_tol);
            for (int l = 0; l < as_dim; l++) {
                for (int n = 0; n < as_dim; n++) {
                    lnk[l * as_dim + n] *= dR_as_two_rdm(k, l, i, n);
                    ln_dR_k[l * as_dim + n] *= as_two_rdm(k, l, i, n);
                    dR_ln_k[l * as_dim + n] *= as_two_rdm(k, l, i, n);
                }
            }
            rhs += sum(*(madness_process.world), lnk);
            rhs += sum(*(madness_process.world), ln_dR_k);
            rhs += sum(*(madness_process.world), dR_ln_k);
        }
        
        rhs.scale(1/as_one_rdm(i, i));
        SeparatedConvolution<double, NDIM> bsh_op =
            BSHOperator<NDIM>(*(madness_process.world), sqrt(-2 * en), num_params.BSH_lo, num_params.BSH_eps);
        Function<double, NDIM> r = dR_active_orbs[i] + 2.0 * bsh_op(rhs);
        // add non bsh part
        for (int k = 0; k < as_dim; k++) {
            r += (dR_as_one_rdm(k, i)/as_one_rdm(i, i))*active_orbs[k];
            if (k != i) {
                r += (as_one_rdm(k, i)/as_one_rdm(i, i))*dR_active_orbs[k];
            }
        }
        double err = r.norm2();
        if (err > highest_as_error) {
            highest_as_error = err;
        }
        std::cout << "Orbital " << i << " update norm: " << err << std::endl;
        dR_orbital_updates.push_back(r);
    }
    
    return dR_orbital_updates;
}

template <std::size_t NDIM>
GradientResponse<NDIM> Gradients<NDIM>::solve_h2(const std::vector<SavedFct<NDIM>>& optimized_orbitals, const Numpy2D& rdm1,
                                const Numpy4D& rdm2, const Numpy2D& lagrange_multipliers, MolecularGeometry& geometry,
                                const Numpy2D& d_rdm1_dz, const Numpy4D& d_rdm2_dz, const Numpy2D& d2_rdm1_dz2,
                                const Numpy4D& d2_rdm2_dz2, int max_iterations) 
                                {
                                    active_orbs = Integrator.read_orbitals(optimized_orbitals);
                                    as_dim = active_orbs.size();

                                    as_one_rdm = refinement_utils::to_madness(rdm1);
                                    as_two_rdm = refinement_utils::to_madness(rdm2);
                                    dz_as_one_rdm = refinement_utils::to_madness(d_rdm1_dz);
                                    dz_as_two_rdm = refinement_utils::to_madness(d_rdm2_dz);
                                    dz2_as_one_rdm = refinement_utils::to_madness(d2_rdm1_dz2);
                                    dz2_as_two_rdm = refinement_utils::to_madness(d2_rdm2_dz2);

                                    LagrangeMultiplier_AS_AS = refinement_utils::to_madness(lagrange_multipliers);

                                    PotentialManager pm = PotentialManager(geometry.mol, "");
                                    pm.make_nuclear_potential(*(madness_process.world));
                                    V = pm.vnuclear();

                                    int atom = 1;
                                    int axis = 2; // z-axis
                                    madchem::MolecularDerivativeFunctor func(geometry.mol, atom, axis);
                                    dR_V = real_factory_3d(*(madness_process.world)).functor(func).truncate_on_project().truncate_mode(0); // hartree/bohr!

                                    for (int i = 0; i < as_dim; ++i) {
                                        dR_active_orbs.push_back(FunctionFactory<double, NDIM>(*(madness_process.world))); // fill initial guess with empty functions
                                    }

                                    // prepare coulomb operator and refined orbital combinations
                                    auto coul_op_parallel = std::shared_ptr<SeparatedConvolution<double, NDIM>>(CoulombOperatorNDPtr<NDIM>(*(madness_process.world), num_params.coulomb_lo, num_params.coulomb_eps));
                                    Integrator.update_as_integral_combinations(active_orbs, orbs_kl, coul_orbs_mn);
                                    // expand compact (k>=l) storage to full square storage, indexed as l*as_dim+n below
                                    std::vector<Function<double, NDIM>> coul_orbs_mn_full;
                                    for (int l = 0; l < as_dim; l++) {
                                        for (int n = 0; n < as_dim; n++) {
                                            coul_orbs_mn_full.push_back(coul_orbs_mn[std::max(l, n) * (std::max(l, n) + 1) / 2 + std::min(l, n)]);
                                        }
                                    }
                                    coul_orbs_mn = coul_orbs_mn_full;

                                    // h (one body) and g (two body) integrals
                                    as_integrals_one_body = Integrator.compute_potential_integrals(active_orbs, V);
                                    as_integrals_one_body += Integrator.compute_kinetic_integrals(active_orbs);
                                    as_integrals_two_body = Integrator.compute_two_body_integrals(active_orbs, orbs_kl, coul_orbs_mn);

                                    // h(phi_k) 
                                    V_active_orbs = V * active_orbs;
                                    V_active_orbs = truncate(V_active_orbs, num_params.truncation_tol);

                                    // dR_V*phi_k
                                    dR_V_active_orbs = dR_V * active_orbs;
                                    dR_V_active_orbs = truncate(dR_V_active_orbs, num_params.truncation_tol);

                                    for (int iteration = 0; iteration < max_iterations; ++iteration) {
                                        std::cout << "Gradient iteration " << iteration << std::endl;
                                        dR_as_integrals_one_body1 = Integrator.compute_core_as_integrals_one_body(dR_active_orbs, active_orbs, V); // repurposed
                                        dR_as_integrals_one_body2 = Integrator.compute_core_as_integrals_one_body(active_orbs, dR_active_orbs, V); // repurposed
                                        dR_as_integrals_one_body2 += Integrator.compute_potential_integrals(active_orbs, dR_V);
                                        dR_as_integrals_one_body = dR_as_integrals_one_body1 + dR_as_integrals_one_body2;
                                        
                                        std::vector<Function<double, NDIM> > response_pairs;
                                        for (int i = 0; i < as_dim; i++) {
                                            std::vector<Function<double, NDIM>> pair_ij = dR_active_orbs[i] * active_orbs + active_orbs[i] * dR_active_orbs;
                                            response_pairs.insert(std::end(response_pairs), std::begin(pair_ij), std::end(pair_ij));
                                        }
                                        response_pairs = truncate(response_pairs, num_params.truncation_tol);
                                        
                                        coul_response_pairs = apply(*(madness_process.world), *coul_op_parallel, response_pairs);
                                        coul_response_pairs = truncate(coul_response_pairs, num_params.truncation_tol);
                                        
                                        dR_as_integrals_two_body = Integrator.compute_two_body_integrals(active_orbs, orbs_kl, coul_response_pairs);
                                        dR_as_integrals_two_body += Integrator.compute_two_body_integrals(active_orbs, response_pairs, coul_orbs_mn);
                                        
                                        calc_dR_manybody_wf();
                                        calc_dR_rdms();
                                        calc_dR_lagrange_mult();
                                        
                                        dR_active_orbs = dR_active_orbs - calc_dR_act_orb_updates();
                                        madness::Tensor<double> U = matrix_inner(*(madness_process.world), active_orbs, dR_active_orbs);

                                        for (int i = 0; i < as_dim; ++i) {
                                            for (int p = 0; p < as_dim; ++p) {
                                                dR_active_orbs[i] -= 0.5 * (U(i, p) + U(p, i)) * active_orbs[p];
                                            }
                                        }
                                        
                                    }

                                    std::vector<SavedFct<NDIM>> dRdphi_E;
                                    for (int i = 0; i < dR_V_active_orbs.size(); i++) {
                                        SavedFct<NDIM> orb(dR_V_active_orbs[i]);
                                        dRdphi_E.push_back(orb);
                                    }

                                    std::vector<SavedFct<NDIM>> response_orbs;
                                    for (int i = 0; i < dR_active_orbs.size(); i++) {
                                        SavedFct<NDIM> orb(dR_active_orbs[i]);
                                        response_orbs.push_back(orb);
                                    }

                                    return GradientResponse<NDIM>(response_orbs, dRdphi_E, dR_z);
                                }




template class Gradients<3>;
