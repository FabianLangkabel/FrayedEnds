#include "integrals_open_shell.hpp"

using namespace madness;

template <std::size_t NDIM>
Integrals_open_shell<NDIM>::Integrals_open_shell(MadnessProcess<NDIM>& mp) : madness_process(mp) {}

template <std::size_t NDIM>
std::vector<Numpy2D> Integrals_open_shell<NDIM>::compute_potential_integrals(std::vector<SavedFct<NDIM>> alpha_orbs, std::vector<SavedFct<NDIM>> beta_orbs, SavedFct<NDIM> potential) {
    std::vector<Function<double, NDIM>> alpha_orbitals, beta_orbitals;
    Function<double, NDIM> V = madness_process.loadfct(potential);
    for (SavedFct<NDIM> orb : alpha_orbs)
        alpha_orbitals.push_back(madness_process.loadfct(orb));
    for (SavedFct<NDIM> orb : beta_orbs)
        beta_orbitals.push_back(madness_process.loadfct(orb));

    potential_integrals_alpha_alpha = madness::matrix_inner(*(madness_process.world), alpha_orbitals, V * alpha_orbitals);
    potential_integrals_beta_beta = madness::matrix_inner(*(madness_process.world), beta_orbitals, V * beta_orbitals);

    Numpy2D alpha_alpha(potential_integrals_alpha_alpha.ptr(), {alpha_orbitals.size(), alpha_orbitals.size()});
    Numpy2D beta_beta(potential_integrals_beta_beta.ptr(), {beta_orbitals.size(), beta_orbitals.size()});

    alpha_orbitals.clear();
    beta_orbitals.clear();

    return std::vector<Numpy2D>{alpha_alpha, beta_beta};
}

template <std::size_t NDIM>
std::vector<Numpy2D> Integrals_open_shell<NDIM>::compute_kinetic_integrals(std::vector<SavedFct<NDIM>> alpha_orbs, std::vector<SavedFct<NDIM>> beta_orbs) {
    std::vector<Function<double, NDIM>> alpha_orbitals, beta_orbitals;
    for (SavedFct<NDIM> orb : alpha_orbs)
        alpha_orbitals.push_back(madness_process.loadfct(orb));
    for (SavedFct<NDIM> orb : beta_orbs)
        beta_orbitals.push_back(madness_process.loadfct(orb));

    kinetic_integrals_alpha_alpha = madness::Tensor<double>(alpha_orbitals.size(), alpha_orbitals.size());
    for (int k = 0; k < alpha_orbitals.size(); k++) {
        for (int l = 0; l < alpha_orbitals.size(); l++) {
            // Kinetic
            for (int axis = 0; axis < NDIM; axis++) {
                Derivative<double, NDIM> D = free_space_derivative<double, NDIM>(*(madness_process.world), axis);
                Function<double, NDIM> d_orb_k = D(alpha_orbitals[k]);
                Function<double, NDIM> d_orb_l = D(alpha_orbitals[l]);
                kinetic_integrals_alpha_alpha(k, l) += 0.5 * inner(d_orb_k, d_orb_l);
            }
        }
    }

    kinetic_integrals_beta_beta = madness::Tensor<double>(beta_orbitals.size(), beta_orbitals.size());
    for (int k = 0; k < beta_orbitals.size(); k++) {
        for (int l = 0; l < beta_orbitals.size(); l++) {
            // Kinetic
            for (int axis = 0; axis < NDIM; axis++) {
                Derivative<double, NDIM> D = free_space_derivative<double, NDIM>(*(madness_process.world), axis);
                Function<double, NDIM> d_orb_k = D(beta_orbitals[k]);
                Function<double, NDIM> d_orb_l = D(beta_orbitals[l]);
                kinetic_integrals_beta_beta(k, l) += 0.5 * inner(d_orb_k, d_orb_l);
            }
        }
    }

    Numpy2D alpha_alpha(kinetic_integrals_alpha_alpha.ptr(), {alpha_orbitals.size(), alpha_orbitals.size()});
    Numpy2D beta_beta(kinetic_integrals_beta_beta.ptr(), {beta_orbitals.size(), beta_orbitals.size()}); //DUMMY

    return std::vector<Numpy2D>{alpha_alpha, beta_beta};
}

// todo: allow 4 sets of orbitals A,B,C,D as input for <AB|g|CD> all indices
template <std::size_t NDIM>
std::vector<Numpy4D> Integrals_open_shell<NDIM>::compute_two_body_integrals(std::vector<SavedFct<NDIM>> alpha_orbs, std::vector<SavedFct<NDIM>> beta_orbs) {

    std::vector<Function<double, NDIM>> alpha_orbitals, beta_orbitals;
    for (SavedFct<NDIM> orb : alpha_orbs)
        alpha_orbitals.push_back(madness_process.loadfct(orb));
    for (SavedFct<NDIM> orb : beta_orbs)
        beta_orbitals.push_back(madness_process.loadfct(orb));

    double truncation_tol = 1e-6;
    double coulomb_lo = 0.001;
    double coulomb_eps = 1e-6;
    auto coul_op_parallel = std::shared_ptr<SeparatedConvolution<double, NDIM>>(CoulombOperatorNDPtr<NDIM>(*(madness_process.world), coulomb_lo, coulomb_eps));

    //alpha alpha interaction
    {
        int as_dim = alpha_orbitals.size();
        auto t1 = std::chrono::high_resolution_clock::now();
        std::vector<Function<double, NDIM>> orbs_kl_alpha;
        for (int k = 0; k < as_dim; k++) {
            std::vector<Function<double, NDIM>> kl = alpha_orbitals[k] * alpha_orbitals;
            orbs_kl_alpha.insert(std::end(orbs_kl_alpha), std::begin(kl), std::end(kl));
        }
        orbs_kl_alpha = truncate(orbs_kl_alpha, truncation_tol);

        auto t2 = std::chrono::high_resolution_clock::now();
        std::vector<Function<double, NDIM>> coul_orbs_mn_alpha = apply(*(madness_process.world), *coul_op_parallel, orbs_kl_alpha);
        coul_orbs_mn_alpha = truncate(coul_orbs_mn_alpha, truncation_tol);

        auto t3 = std::chrono::high_resolution_clock::now();
        two_body_integrals_alpha_alpha = madness::Tensor<double>(as_dim, as_dim, as_dim, as_dim);
        madness::Tensor<double> Inner_prods = matrix_inner(*(madness_process.world), orbs_kl_alpha, coul_orbs_mn_alpha, false);
        std::vector<double> flat;
        for (int k = 0; k < as_dim; k++) {
            for (int l = 0; l < as_dim; l++) {
                for (int m = 0; m < as_dim; m++) {
                    for (int n = 0; n < as_dim; n++) {
                        auto tmp = Inner_prods(k * as_dim + l, m * as_dim + n);
                        two_body_integrals_alpha_alpha(k, m, l, n) = tmp;
                        flat.push_back(tmp);
                    }
                }
            }
        }
        auto t4 = std::chrono::high_resolution_clock::now();
        // todo: save timings
    }

    //beta beta interaction
    {
        int as_dim = beta_orbitals.size();
        auto t1 = std::chrono::high_resolution_clock::now();
        std::vector<Function<double, NDIM>> orbs_kl_beta;
        for (int k = 0; k < as_dim; k++) {
            std::vector<Function<double, NDIM>> kl = beta_orbitals[k] * beta_orbitals;
            orbs_kl_beta.insert(std::end(orbs_kl_beta), std::begin(kl), std::end(kl));
        }
        orbs_kl_beta = truncate(orbs_kl_beta, truncation_tol);

        auto t2 = std::chrono::high_resolution_clock::now();
        std::vector<Function<double, NDIM>> coul_orbs_mn_beta = apply(*(madness_process.world), *coul_op_parallel, orbs_kl_beta);
        coul_orbs_mn_beta = truncate(coul_orbs_mn_beta, truncation_tol);

        auto t3 = std::chrono::high_resolution_clock::now();
        two_body_integrals_beta_beta = madness::Tensor<double>(as_dim, as_dim, as_dim, as_dim);
        madness::Tensor<double> Inner_prods = matrix_inner(*(madness_process.world), orbs_kl_beta, coul_orbs_mn_beta, false);
        std::vector<double> flat;
        for (int k = 0; k < as_dim; k++) {
            for (int l = 0; l < as_dim; l++) {
                for (int m = 0; m < as_dim; m++) {
                    for (int n = 0; n < as_dim; n++) {
                        auto tmp = Inner_prods(k * as_dim + l, m * as_dim + n);
                        two_body_integrals_beta_beta(k, m, l, n) = tmp;
                        flat.push_back(tmp);
                    }
                }
            }
        }
        auto t4 = std::chrono::high_resolution_clock::now();
        // todo: save timings
    }

    //alpha beta interaction
    /*
    {
        int a_dim = alpha_orbitals.size();
        int b_dim = beta_orbitals.size();

        auto t1 = std::chrono::high_resolution_clock::now();
        std::vector<Function<double, NDIM>> orbs_kl_mixed;
        for (int l = 0; l < b_dim; l++) {
            std::vector<Function<double, NDIM>> kl = alpha_orbitals * beta_orbitals[l];
            orbs_kl_mixed.insert(std::end(orbs_kl_mixed), std::begin(kl), std::end(kl));
        }
        orbs_kl_mixed = truncate(orbs_kl_mixed, truncation_tol);

        auto t2 = std::chrono::high_resolution_clock::now();
        std::vector<Function<double, NDIM>> coul_orbs_mn_mixed = apply(*(madness_process.world), *coul_op_parallel, orbs_kl_mixed);
        coul_orbs_mn_mixed = truncate(coul_orbs_mn_mixed, truncation_tol);

        auto t3 = std::chrono::high_resolution_clock::now();
        two_body_integrals_alpha_beta = madness::Tensor<double>(a_dim, b_dim, a_dim, b_dim);
        madness::Tensor<double> Inner_prods = matrix_inner(*(madness_process.world), orbs_kl_mixed, coul_orbs_mn_mixed, false);
        std::vector<double> flat;
        for (int k = 0; k < a_dim; k++) {
            for (int l = 0; l < b_dim; l++) {
                for (int m = 0; m < a_dim; m++) {
                    for (int n = 0; n < b_dim; n++) {
                        auto tmp = Inner_prods(k + l * a_dim, m + n * a_dim);
                        two_body_integrals_alpha_beta(k, m, l, n) = tmp;
                        flat.push_back(tmp);
                    }
                }
            }
        }
        auto t4 = std::chrono::high_resolution_clock::now();
        // todo: save timings
    }
    */
    {
        // 2. Versuch:
        int a_dim = alpha_orbitals.size();
        int b_dim = beta_orbitals.size();

        auto t1 = std::chrono::high_resolution_clock::now();

        std::vector<Function<double, NDIM>> orbs_kl_alpha;
        for (int k = 0; k < a_dim; k++) {
            std::vector<Function<double, NDIM>> kl = alpha_orbitals[k] * alpha_orbitals;
            orbs_kl_alpha.insert(std::end(orbs_kl_alpha), std::begin(kl), std::end(kl));
        }
        orbs_kl_alpha = truncate(orbs_kl_alpha, truncation_tol);

        std::vector<Function<double, NDIM>> orbs_mn_beta;
        for (int m = 0; m < b_dim; m++) {
            std::vector<Function<double, NDIM>> mn = beta_orbitals[m] * beta_orbitals;
            orbs_mn_beta.insert(std::end(orbs_mn_beta), std::begin(mn), std::end(mn));
        }
        orbs_mn_beta = truncate(orbs_mn_beta, truncation_tol);



        auto t2 = std::chrono::high_resolution_clock::now();
        std::vector<Function<double, NDIM>> coul_orbs_mn_beta = apply(*(madness_process.world), *coul_op_parallel, orbs_mn_beta);
        coul_orbs_mn_beta = truncate(coul_orbs_mn_beta, truncation_tol);

        auto t3 = std::chrono::high_resolution_clock::now();
        two_body_integrals_alpha_beta = madness::Tensor<double>(a_dim, b_dim, a_dim, b_dim);
        madness::Tensor<double> Inner_prods = matrix_inner(*(madness_process.world), orbs_kl_alpha, coul_orbs_mn_beta, false);
        for (int k = 0; k < a_dim; k++) {
            for (int l = 0; l < a_dim; l++) {
                for (int m = 0; m < b_dim; m++) {
                    for (int n = 0; n < b_dim; n++) {
                        auto tmp = Inner_prods(k * a_dim + l, m * b_dim + n);
                        two_body_integrals_alpha_beta(k, m, l, n) = tmp;
                    }
                }
            }
        }
        auto t4 = std::chrono::high_resolution_clock::now();
        // todo: save timings
    }

    Numpy4D alpha_alpha(two_body_integrals_alpha_alpha.ptr(), {alpha_orbitals.size(), alpha_orbitals.size(), alpha_orbitals.size(), alpha_orbitals.size()});
    Numpy4D beta_beta(two_body_integrals_beta_beta.ptr(), {beta_orbitals.size(), beta_orbitals.size(), beta_orbitals.size(), beta_orbitals.size()});
    Numpy4D alpha_beta(two_body_integrals_alpha_beta.ptr(), {alpha_orbitals.size(), beta_orbitals.size(), alpha_orbitals.size(), beta_orbitals.size()});

    return std::vector<Numpy4D>{alpha_alpha, alpha_beta, beta_beta};
}

template class Integrals_open_shell<2>;
template class Integrals_open_shell<3>;