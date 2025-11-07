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
    Numpy2D beta_beta(kinetic_integrals_beta_beta.ptr(), {beta_orbitals.size(), beta_orbitals.size()});

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
    {
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

template <std::size_t NDIM>
nb::tuple Integrals_open_shell<NDIM>::compute_effective_hamiltonian(std::vector<SavedFct<NDIM>> core_alpha_orbs, std::vector<SavedFct<NDIM>> core_beta_orbs, std::vector<SavedFct<NDIM>> active_alpha_orbs, std::vector<SavedFct<NDIM>> active_beta_orbs, SavedFct<NDIM> potential, double energy_offset)
{
    std::array<std::vector<Function<double, NDIM>>, 2> core_orbitals;
    std::array<std::vector<Function<double, NDIM>>, 2> active_orbitals;

    double truncation_tol = 1e-6;
    double coulomb_lo = 0.001;
    double coulomb_eps = 1e-6;
    auto coul_op_parallel = std::shared_ptr<SeparatedConvolution<double, NDIM>>(CoulombOperatorNDPtr<NDIM>(*(madness_process.world), coulomb_lo, coulomb_eps));

    for (SavedFct<NDIM> orb : core_alpha_orbs)
        core_orbitals[0].push_back(madness_process.loadfct(orb));
    for (SavedFct<NDIM> orb : core_beta_orbs)
        core_orbitals[1].push_back(madness_process.loadfct(orb));
    for (SavedFct<NDIM> orb : active_alpha_orbs)
        active_orbitals[0].push_back(madness_process.loadfct(orb));
    for (SavedFct<NDIM> orb : active_beta_orbs)
        active_orbitals[1].push_back(madness_process.loadfct(orb));

    Function<double, NDIM> V = madness_process.loadfct(potential);

    //*********************
    // AS - AS
    //******************** */
    //One electron integrals
    for(int spin = 0; spin < 2; spin++)
    {
        madness::Tensor<double> integrals(active_orbitals[spin].size(), active_orbitals[spin].size());

        madness::Tensor<double> pot_integrals = madness::matrix_inner(*(madness_process.world), active_orbitals[spin], V * active_orbitals[spin]);
        for (int k = 0; k < active_orbitals[spin].size(); k++) {
            for (int l = 0; l < active_orbitals[spin].size(); l++) {
                // Kinetic
                for (int axis = 0; axis < NDIM; axis++) {
                    Derivative<double, NDIM> D = free_space_derivative<double, NDIM>(*(madness_process.world), axis);
                    Function<double, NDIM> d_orb_k = D(active_orbitals[spin][k]);
                    Function<double, NDIM> d_orb_l = D(active_orbitals[spin][l]);
                    integrals(k, l) += 0.5 * inner(d_orb_k, d_orb_l);
                }
                // Pot
                integrals(k, l) += pot_integrals(k,l);
            }
        }
        one_integrals[spin] = integrals;
    }
    //Two electron integrals
    std::vector<Numpy4D> as_two_body_integrals = compute_two_body_integrals(active_alpha_orbs, active_beta_orbs);


    //*********************
    // Core
    //******************** */
    // 1e core energy 
    double core_kinetic_energy = 0;
    double core_nuclear_attraction_energy = 0;
    for (int spin = 0; spin < 2; spin++)
    {
        for (int k = 0; k < core_orbitals[spin].size(); k++) {
            // Kinetic
            for (int axis = 0; axis < NDIM; axis++) {
                Derivative<double, NDIM> D = free_space_derivative<double, NDIM>(*(madness_process.world), axis);
                Function<double, NDIM> d_orb_k = D(core_orbitals[spin][k]);
                core_kinetic_energy += 0.5 * inner(d_orb_k, d_orb_k);
            }
            // Pot
            core_nuclear_attraction_energy += inner(core_orbitals[spin][k], (V * core_orbitals[spin][k]));
        }
    }




    // 2e core energy
    double core_two_electron_energy = 0;

    // <ab|ab>
    {
        std::array<std::vector<Function<double, NDIM>>, 2> orbs_aa;
        std::array<std::vector<Function<double, NDIM>>, 2> coul_orbs_aa;
        
        for (int spin = 0; spin < 2; spin++)
        {
            if(core_orbitals[spin].size() > 0)
            {
                for (int a = 0; a < core_orbitals[spin].size(); a++) {
                    orbs_aa[spin].push_back(core_orbitals[spin][a] * core_orbitals[spin][a]);
                }
                orbs_aa[spin] = truncate(orbs_aa[spin], truncation_tol);
                coul_orbs_aa[spin] = apply(*(madness_process.world), *coul_op_parallel, orbs_aa[spin]);
                coul_orbs_aa[spin] = truncate(coul_orbs_aa[spin], truncation_tol);

                // alph-alpha and beta-beta
                for (int a = 0; a < core_orbitals[spin].size(); a++) {
                    madness::Tensor<double> Inner_prods_abab = matrix_inner(*(madness_process.world), std::vector<Function<double, NDIM>>{orbs_aa[spin][a]}, coul_orbs_aa[spin], false);
                    for (int b = 0; b < core_orbitals[spin].size(); b++) {
                        core_two_electron_energy += Inner_prods_abab(0, b);
                    }
                }
            }
        }
        // alpha-beta
        if(core_orbitals[0].size() > 0 && core_orbitals[0].size() > 1)
        {
            for (int a = 0; a < core_orbitals[0].size(); a++) {
                madness::Tensor<double> Inner_prods_abab = matrix_inner(*(madness_process.world), std::vector<Function<double, NDIM>>{orbs_aa[0][a]}, coul_orbs_aa[1], false);
                for (int b = 0; b < core_orbitals[1].size(); b++) {
                    core_two_electron_energy += 2 * Inner_prods_abab(0, b);
                }
            }
        }
    }

    // <ab|ba> Terms with a and b of different spin are 0
    for (int spin = 0; spin < 2; spin++)
    {
        if(core_orbitals[spin].size() > 0)
        {
            for (int a = 0; a < core_orbitals[spin].size(); a++)
            {
                std::vector<Function<double, NDIM>> orbs_ab = core_orbitals[spin][a] * core_orbitals[spin];
                orbs_ab = truncate(orbs_ab, truncation_tol);
                std::vector<Function<double, NDIM>> coul_orbs_ab = apply(*(madness_process.world), *coul_op_parallel, orbs_ab);
                coul_orbs_ab = truncate(coul_orbs_ab, truncation_tol);
                for (int b = 0; b < core_orbitals[spin].size(); b++) {
                    core_two_electron_energy -= inner(orbs_ab[b], coul_orbs_ab[b]);
                }
            }
        }
    }

    core_two_electron_energy = 0.5 * core_two_electron_energy;
    effective_hamiltonian_core_energy = energy_offset + core_kinetic_energy + core_nuclear_attraction_energy + core_two_electron_energy;
    print("                       Nuclear repulsion ", energy_offset);
    print("                   Core - Kinetic energy ", core_kinetic_energy);
    print("        Core - Nuclear attraction energy ", core_nuclear_attraction_energy);
    print("              Core - Two-electron energy ", core_two_electron_energy);
    print("                       Total core energy ", effective_hamiltonian_core_energy);



    //*********************
    // Core - AS
    //******************** */
    std::array<madness::Tensor<double>, 2> core_as_integrals_two_body_akal; //stored as (a,k,l); a is alpha + beta list
    {
        std::vector<Function<double, NDIM>> orbs_aa;
        for (int spin = 0; spin < 2; spin++)
        {
            for (int a = 0; a < core_orbitals[spin].size(); a++) {
                orbs_aa.push_back(core_orbitals[spin][a] * core_orbitals[spin][a]);   
            }
        }
        orbs_aa = truncate(orbs_aa, truncation_tol);

        for (int spin = 0; spin < 2; spin++)
        {
            madness::Tensor<double> integrals = madness::Tensor<double>(core_orbitals[0].size() + core_orbitals[1].size(), active_orbitals[spin].size(), active_orbitals[spin].size());
            core_as_integrals_two_body_akal[spin] = integrals;
            
            std::vector<Function<double, NDIM>> orbs_kl;
            for (int k = 0; k < active_orbitals[spin].size(); k++) {
                std::vector<Function<double, NDIM>> kl = active_orbitals[spin][k] * active_orbitals[spin];
                orbs_kl.insert(std::end(orbs_kl), std::begin(kl), std::end(kl));
            }
            orbs_kl = truncate(orbs_kl, truncation_tol);
            std::vector<Function<double, NDIM>> coul_orbs_kl = apply(*(madness_process.world), *coul_op_parallel, orbs_kl);
            coul_orbs_kl = truncate(coul_orbs_kl, truncation_tol);

            madness::Tensor<double> Inner_prods_akal = matrix_inner(*(madness_process.world), orbs_aa, coul_orbs_kl, false);
            for (int a = 0; a < orbs_aa.size(); a++) {
                for (int k = 0; k < active_orbitals[spin].size(); k++) {
                    for (int l = 0; l < active_orbitals[spin].size(); l++) {
                        core_as_integrals_two_body_akal[spin](a, k, l) = Inner_prods_akal(a, k * active_orbitals[spin].size() + l);
                        one_integrals[spin](k,l) += core_as_integrals_two_body_akal[spin](a, k, l);
                    }
                }
            }
        }
    }

    std::array<madness::Tensor<double>, 2> core_as_integrals_two_body_akla; //stored as (a,k,l); a is list of [spin]
    for (int spin = 0; spin < 2; spin++)
    {
        madness::Tensor<double> integrals = madness::Tensor<double>(core_orbitals[spin].size(), active_orbitals[spin].size(), active_orbitals[spin].size());
        core_as_integrals_two_body_akla[spin] = integrals;
        for (int a = 0; a < core_orbitals[spin].size(); a++)
        {
            std::vector<Function<double, NDIM>> orbs_ak = core_orbitals[spin][a] * active_orbitals[spin];
            orbs_ak = truncate(orbs_ak, truncation_tol);
            std::vector<Function<double, NDIM>> coul_orbs_ak = apply(*(madness_process.world), *coul_op_parallel, orbs_ak);
            coul_orbs_ak = truncate(coul_orbs_ak, truncation_tol);

            std::vector<Function<double, NDIM>> orbs_ka = active_orbitals[spin] * core_orbitals[spin][a];
            orbs_ka = truncate(orbs_ka, truncation_tol);

            // <ak|la> = <ka|al>
            madness::Tensor<double> Inner_prods_akla = matrix_inner(*(madness_process.world), orbs_ka, coul_orbs_ak, false);
            for (int k = 0; k < active_orbitals[spin].size(); k++) {
                for (int l = 0; l < active_orbitals[spin].size(); l++) {
                    core_as_integrals_two_body_akla[spin](a, k, l) = Inner_prods_akla(l, k);
                    one_integrals[spin](k,l) -= core_as_integrals_two_body_akla[spin](a, k, l); // In open shell no factor 0.5
                }
            }
        }
    }

    //*********************
    // Finalize
    //******************** */

    //One electron integrals
    Numpy2D one_e_alpha_alpha(one_integrals[0] .ptr(), {active_orbitals[0].size(), active_orbitals[0].size()});
    Numpy2D one_e_beta_beta(one_integrals[1] .ptr(), {active_orbitals[1].size(), active_orbitals[1].size()});
    nb::list h1_list;
    h1_list.append(one_e_alpha_alpha);
    h1_list.append(one_e_beta_beta);


    //Two electron integrals
    nb::list g2_list;
    g2_list.append(as_two_body_integrals[0]);
    g2_list.append(as_two_body_integrals[1]);
    g2_list.append(as_two_body_integrals[2]);

    return nb::make_tuple(effective_hamiltonian_core_energy, h1_list, g2_list);
}

template class Integrals_open_shell<2>;
template class Integrals_open_shell<3>;


