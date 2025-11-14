#include "integrals_open_shell.hpp"

using namespace madness;

template <std::size_t NDIM>
Integrals_open_shell<NDIM>::Integrals_open_shell(MadnessProcess<NDIM>& mp) : madness_process(mp) {}




//
// Utility Functions
//

template <std::size_t NDIM>
std::array<std::vector<Function<double, NDIM>>, 2> Integrals_open_shell<NDIM>::read_orbitals(std::vector<SavedFct<NDIM>> alpha_orbs, std::vector<SavedFct<NDIM>> beta_orbs)
{
    std::array<std::vector<Function<double, NDIM>>, 2> orbitals;
    for (SavedFct<NDIM> orb : alpha_orbs)
        orbitals[0].push_back(madness_process.loadfct(orb));
    for (SavedFct<NDIM> orb : beta_orbs)
        orbitals[1].push_back(madness_process.loadfct(orb));
    return orbitals;
}




//
// Nanobind bindings
//

template <std::size_t NDIM>
std::vector<Numpy2D> Integrals_open_shell<NDIM>::nb_compute_potential_integrals(std::vector<SavedFct<NDIM>> alpha_orbs, std::vector<SavedFct<NDIM>> beta_orbs, SavedFct<NDIM> potential) 
{
    std::array<std::vector<Function<double, NDIM>>, 2> orbitals = read_orbitals(alpha_orbs, beta_orbs);
    Function<double, NDIM> V = madness_process.loadfct(potential);

    std::array<madness::Tensor<double>, 2> Integrals = compute_potential_integrals(orbitals, V);

    auto alpha_alpha_owner = std::make_shared<madness::Tensor<double>>(std::move(Integrals[0]));
    auto beta_beta_owner = std::make_shared<madness::Tensor<double>>(std::move(Integrals[1]));

    nb::capsule alpha_alpha_caps(
        new std::shared_ptr<madness::Tensor<double>>(alpha_alpha_owner),
        [](void *p) noexcept {
            delete reinterpret_cast<std::shared_ptr<madness::Tensor<double>>*>(p);
        }
    );
    nb::capsule beta_beta_caps(
        new std::shared_ptr<madness::Tensor<double>>(beta_beta_owner),
        [](void *p) noexcept {
            delete reinterpret_cast<std::shared_ptr<madness::Tensor<double>>*>(p);
        }
    );

    Numpy2D alpha_alpha(alpha_alpha_owner->ptr(), {orbitals[0].size(), orbitals[0].size()}, alpha_alpha_caps);
    Numpy2D beta_beta (beta_beta_owner->ptr(), {orbitals[1].size(), orbitals[1].size()}, beta_beta_caps);

    return std::vector<Numpy2D>{alpha_alpha, beta_beta};
}

template <std::size_t NDIM>
std::vector<Numpy2D> Integrals_open_shell<NDIM>::nb_compute_kinetic_integrals(std::vector<SavedFct<NDIM>> alpha_orbs, std::vector<SavedFct<NDIM>> beta_orbs) 
{
    std::array<std::vector<Function<double, NDIM>>, 2> orbitals = read_orbitals(alpha_orbs, beta_orbs);

    std::array<madness::Tensor<double>, 2> Integrals = compute_kinetic_integrals(orbitals);

    auto alpha_alpha_owner = std::make_shared<madness::Tensor<double>>(std::move(Integrals[0]));
    auto beta_beta_owner = std::make_shared<madness::Tensor<double>>(std::move(Integrals[1]));

    nb::capsule alpha_alpha_caps(
        new std::shared_ptr<madness::Tensor<double>>(alpha_alpha_owner),
        [](void *p) noexcept {
            delete reinterpret_cast<std::shared_ptr<madness::Tensor<double>>*>(p);
        }
    );
    nb::capsule beta_beta_caps(
        new std::shared_ptr<madness::Tensor<double>>(beta_beta_owner),
        [](void *p) noexcept {
            delete reinterpret_cast<std::shared_ptr<madness::Tensor<double>>*>(p);
        }
    );

    Numpy2D alpha_alpha(alpha_alpha_owner->ptr(), {orbitals[0].size(), orbitals[0].size()}, alpha_alpha_caps);
    Numpy2D beta_beta (beta_beta_owner->ptr(), {orbitals[1].size(), orbitals[1].size()}, beta_beta_caps);

    return std::vector<Numpy2D>{alpha_alpha, beta_beta};
}

template <std::size_t NDIM>
std::vector<Numpy4D> Integrals_open_shell<NDIM>::nb_compute_two_body_integrals(std::vector<SavedFct<NDIM>> alpha_orbs, std::vector<SavedFct<NDIM>> beta_orbs) 
{
    std::array<std::vector<Function<double, NDIM>>, 2> orbitals = read_orbitals(alpha_orbs, beta_orbs);

    std::array<madness::Tensor<double>, 3> Integrals = compute_two_body_integrals(orbitals);

    auto alpha_alpha_owner = std::make_shared<madness::Tensor<double>>(std::move(Integrals[0]));
    auto beta_beta_owner = std::make_shared<madness::Tensor<double>>(std::move(Integrals[1]));
    auto alpha_beta_owner = std::make_shared<madness::Tensor<double>>(std::move(Integrals[2]));

    nb::capsule alpha_alpha_caps(
        new std::shared_ptr<madness::Tensor<double>>(alpha_alpha_owner),
        [](void *p) noexcept {
            delete reinterpret_cast<std::shared_ptr<madness::Tensor<double>>*>(p);
        }
    );
    nb::capsule beta_beta_caps(
        new std::shared_ptr<madness::Tensor<double>>(beta_beta_owner),
        [](void *p) noexcept {
            delete reinterpret_cast<std::shared_ptr<madness::Tensor<double>>*>(p);
        }
    );
    nb::capsule alpha_beta_caps(
        new std::shared_ptr<madness::Tensor<double>>(alpha_beta_owner),
        [](void *p) noexcept {
            delete reinterpret_cast<std::shared_ptr<madness::Tensor<double>>*>(p);
        }
    );

    Numpy4D alpha_alpha(alpha_alpha_owner->ptr(), {orbitals[0].size(), orbitals[0].size(), orbitals[0].size(), orbitals[0].size()}, alpha_alpha_caps);
    Numpy4D beta_beta(beta_beta_owner->ptr(), {orbitals[1].size(), orbitals[1].size(), orbitals[1].size(), orbitals[1].size()}, beta_beta_caps);
    Numpy4D alpha_beta(alpha_beta_owner->ptr(), {orbitals[0].size(), orbitals[1].size(), orbitals[0].size(), orbitals[1].size()}, alpha_beta_caps);

    return std::vector<Numpy4D>{alpha_alpha, alpha_beta, beta_beta}; //aa, ab, bb order external; aa, bb, ab order internal
}

template <std::size_t NDIM>
nb::tuple Integrals_open_shell<NDIM>::nb_compute_effective_hamiltonian(std::vector<SavedFct<NDIM>> core_alpha_orbs, std::vector<SavedFct<NDIM>> core_beta_orbs, std::vector<SavedFct<NDIM>> active_alpha_orbs, std::vector<SavedFct<NDIM>> active_beta_orbs, SavedFct<NDIM> potential, double energy_offset)
{
    std::array<std::vector<Function<double, NDIM>>, 2> active_orbitals = read_orbitals(active_alpha_orbs, active_beta_orbs);
    std::array<std::vector<Function<double, NDIM>>, 2> core_orbitals = read_orbitals(core_alpha_orbs, core_beta_orbs);
    Function<double, NDIM> V = madness_process.loadfct(potential);

    // Active space integrals
    std::array<madness::Tensor<double>, 2> one_e_integrals = compute_potential_integrals(active_orbitals, V);
    {
        std::array<madness::Tensor<double>, 2> kin_Integrals = compute_kinetic_integrals(active_orbitals);
        one_e_integrals[0] += kin_Integrals[0];
        one_e_integrals[1] += kin_Integrals[1];
    }
    std::array<madness::Tensor<double>, 3> two_e_integrals = compute_two_body_integrals(active_orbitals);

    // Core interactions
    double effective_hamiltonian_core_energy = energy_offset;
    if(core_orbitals[0].size() > 0 || core_orbitals[1].size() > 0)
    {
        // Core energy
        effective_hamiltonian_core_energy = compute_core_energy(core_orbitals, V, energy_offset);

        //Core-AS interaction
        std::vector<std::vector<madness::Tensor<double>>> core_as_integrals_two_body = compute_core_as_integrals_two_body(
            core_orbitals, active_orbitals, true, true, false, false, false);
        for (int spin = 0; spin < 2; spin++) {
            for (int a = 0; a < (core_orbitals[0].size() + core_orbitals[1].size()); a++) {
                for (int k = 0; k < active_orbitals[spin].size(); k++) {
                    for (int l = 0; l < active_orbitals[spin].size(); l++) {
                        one_e_integrals[spin](k,l) += core_as_integrals_two_body[0][spin](a, k, l);
                    }
                }
            }
        }
        for (int spin = 0; spin < 2; spin++) {
            for (int a = 0; a < core_orbitals[spin].size(); a++) {
                for (int k = 0; k < active_orbitals[spin].size(); k++) {
                    for (int l = 0; l < active_orbitals[spin].size(); l++) {
                        one_e_integrals[spin](k,l) -= core_as_integrals_two_body[1][spin](a, k, l);
                    }
                }
            }
        }
    }

    // Return integrals
    Numpy2D one_e_alpha_alpha(one_e_integrals[0].ptr(), {active_orbitals[0].size(), active_orbitals[0].size()});
    Numpy2D one_e_beta_beta(one_e_integrals[1].ptr(), {active_orbitals[1].size(), active_orbitals[1].size()});
    nb::list h1_list;
    h1_list.append(one_e_alpha_alpha);
    h1_list.append(one_e_beta_beta);

    Numpy4D alpha_alpha(two_e_integrals[0].ptr(), {active_orbitals[0].size(), active_orbitals[0].size(), active_orbitals[0].size(), active_orbitals[0].size()});
    Numpy4D beta_beta(two_e_integrals[1].ptr(), {active_orbitals[1].size(), active_orbitals[1].size(), active_orbitals[1].size(), active_orbitals[1].size()});
    Numpy4D alpha_beta(two_e_integrals[2].ptr(), {active_orbitals[0].size(), active_orbitals[1].size(), active_orbitals[0].size(), active_orbitals[1].size()});
    //aa, ab, bb order external; aa, bb, ab order internal
    nb::list g2_list;
    g2_list.append(alpha_alpha);
    g2_list.append(alpha_beta);
    g2_list.append(beta_beta);

    return nb::make_tuple(effective_hamiltonian_core_energy, h1_list, g2_list);
}



//
// Integrators
//

template <std::size_t NDIM>
std::array<madness::Tensor<double>, 2> Integrals_open_shell<NDIM>::compute_potential_integrals(std::array<std::vector<Function<double, NDIM>>, 2> orbitals, Function<double, NDIM> V) 
{
    std::array<madness::Tensor<double>, 2> Integrals;
    for(int spin = 0; spin < 2; spin++)
    {
        Integrals[spin] = madness::matrix_inner(*(madness_process.world), orbitals[spin], V * orbitals[spin]);
    }
    return Integrals;
}

template <std::size_t NDIM>
std::array<madness::Tensor<double>, 2> Integrals_open_shell<NDIM>::compute_kinetic_integrals(std::array<std::vector<Function<double, NDIM>>, 2> orbitals) 
{
    std::array<madness::Tensor<double>, 2> Integrals;
    for(int spin = 0; spin < 2; spin++)
    {
        madness::Tensor<double> ints = madness::Tensor<double>(orbitals[spin].size(), orbitals[spin].size());
        for (int k = 0; k < orbitals[spin].size(); k++) {
            for (int l = 0; l < orbitals[spin].size(); l++) {
                for (int axis = 0; axis < NDIM; axis++) {
                    Derivative<double, NDIM> D = free_space_derivative<double, NDIM>(*(madness_process.world), axis);
                    Function<double, NDIM> d_orb_k = D(orbitals[spin][k]);
                    Function<double, NDIM> d_orb_l = D(orbitals[spin][l]);
                    ints(k, l) += 0.5 * inner(d_orb_k, d_orb_l);
                }
            }
        }
        Integrals[spin] = ints;
    }
    return Integrals;
}

template <std::size_t NDIM>
std::array<madness::Tensor<double>, 3> Integrals_open_shell<NDIM>::compute_two_body_integrals(std::array<std::vector<Function<double, NDIM>>, 2> orbitals) 
{
    double truncation_tol = 1e-6;
    double coulomb_lo = 0.001;
    double coulomb_eps = 1e-6;
    auto coul_op_parallel = std::shared_ptr<SeparatedConvolution<double, NDIM>>(CoulombOperatorNDPtr<NDIM>(*(madness_process.world), coulomb_lo, coulomb_eps));

    std::array<madness::Tensor<double>, 3> Integrals;

    //alpha-alpha and beta-beta interaction
    for(int spin = 0; spin < 2; spin++)
    {
        std::vector<Function<double, NDIM>> orbs_kl;
        for (int k = 0; k < orbitals[spin].size(); k++) {
            std::vector<Function<double, NDIM>> kl = orbitals[spin][k] * orbitals[spin];
            orbs_kl.insert(std::end(orbs_kl), std::begin(kl), std::end(kl));
        }
        orbs_kl = truncate(orbs_kl, truncation_tol);

        std::vector<Function<double, NDIM>> coul_orbs_mn = apply(*(madness_process.world), *coul_op_parallel, orbs_kl);
        coul_orbs_mn = truncate(coul_orbs_mn, truncation_tol);

        madness::Tensor<double> ints = madness::Tensor<double>(orbitals[spin].size(), orbitals[spin].size(), orbitals[spin].size(), orbitals[spin].size());
        madness::Tensor<double> Inner_prods = matrix_inner(*(madness_process.world), orbs_kl, coul_orbs_mn, false);
        std::vector<double> flat;
        for (int k = 0; k < orbitals[spin].size(); k++) {
            for (int l = 0; l < orbitals[spin].size(); l++) {
                for (int m = 0; m < orbitals[spin].size(); m++) {
                    for (int n = 0; n < orbitals[spin].size(); n++) {
                        auto tmp = Inner_prods(k * orbitals[spin].size() + l, m * orbitals[spin].size() + n);
                        ints(k, m, l, n) = tmp;
                        flat.push_back(tmp);
                    }
                }
            }
        }
        Integrals[spin] = ints;
    }
    
    //alpha beta interaction
    {
        std::vector<Function<double, NDIM>> orbs_kl_alpha;
        for (int k = 0; k < orbitals[0].size(); k++) {
            std::vector<Function<double, NDIM>> kl = orbitals[0][k] * orbitals[0];
            orbs_kl_alpha.insert(std::end(orbs_kl_alpha), std::begin(kl), std::end(kl));
        }
        orbs_kl_alpha = truncate(orbs_kl_alpha, truncation_tol);


        std::vector<Function<double, NDIM>> orbs_mn_beta;
        for (int m = 0; m < orbitals[1].size(); m++) {
            std::vector<Function<double, NDIM>> mn = orbitals[1][m] * orbitals[1];
            orbs_mn_beta.insert(std::end(orbs_mn_beta), std::begin(mn), std::end(mn));
        }
        orbs_mn_beta = truncate(orbs_mn_beta, truncation_tol);
        std::vector<Function<double, NDIM>> coul_orbs_mn_beta = apply(*(madness_process.world), *coul_op_parallel, orbs_mn_beta);
        coul_orbs_mn_beta = truncate(coul_orbs_mn_beta, truncation_tol);


        madness::Tensor<double> ints = madness::Tensor<double>(orbitals[0].size(), orbitals[1].size(), orbitals[0].size(), orbitals[1].size());
        madness::Tensor<double> Inner_prods = matrix_inner(*(madness_process.world), orbs_kl_alpha, coul_orbs_mn_beta, false);
        for (int k = 0; k < orbitals[0].size(); k++) {
            for (int l = 0; l < orbitals[0].size(); l++) {
                for (int m = 0; m < orbitals[1].size(); m++) {
                    for (int n = 0; n < orbitals[1].size(); n++) {
                        auto tmp = Inner_prods(k * orbitals[0].size() + l, m * orbitals[1].size() + n);
                        ints(k, m, l, n) = tmp;
                    }
                }
            }
        }
        Integrals[2] = ints;
    }

    return Integrals;
}

template <std::size_t NDIM>
double Integrals_open_shell<NDIM>::compute_core_energy(std::array<std::vector<Function<double, NDIM>>, 2> core_orbitals, Function<double, NDIM> V, double energy_offset) 
{
    double truncation_tol = 1e-6;
    double coulomb_lo = 0.001;
    double coulomb_eps = 1e-6;
    auto coul_op_parallel = std::shared_ptr<SeparatedConvolution<double, NDIM>>(CoulombOperatorNDPtr<NDIM>(*(madness_process.world), coulomb_lo, coulomb_eps));

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
        if(core_orbitals[0].size() > 0 && core_orbitals[1].size() > 0)
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
    double core_energy = energy_offset + core_kinetic_energy + core_nuclear_attraction_energy + core_two_electron_energy;
    print("      Initial core enrgy (energy offset) ", energy_offset);
    print("                   Core - Kinetic energy ", core_kinetic_energy);
    print("        Core - Nuclear attraction energy ", core_nuclear_attraction_energy);
    print("              Core - Two-electron energy ", core_two_electron_energy);
    print("                       Total core energy ", core_energy);

    return core_energy;
}

template <std::size_t NDIM>
std::vector<std::vector<madness::Tensor<double>>> Integrals_open_shell<NDIM>::compute_core_as_integrals_two_body(
    std::array<std::vector<Function<double, NDIM>>, 2> core_orbitals, std::array<std::vector<Function<double, NDIM>>, 2> active_orbitals,
    bool calc_akal, bool calc_akla, bool calc_akln, bool calc_abak, bool calc_baak
)
{
    double truncation_tol = 1e-6;
    double coulomb_lo = 0.001;
    double coulomb_eps = 1e-6;
    auto coul_op_parallel = std::shared_ptr<SeparatedConvolution<double, NDIM>>(CoulombOperatorNDPtr<NDIM>(*(madness_process.world), coulomb_lo, coulomb_eps));

    std::vector<madness::Tensor<double>> core_as_integrals_two_body_akal; //stored as (a,k,l); a is alpha + beta list; no alpha-beta-interaction
    std::vector<madness::Tensor<double>> core_as_integrals_two_body_akla; //stored as (a,k,l); a is list of [spin]; no alpha-beta-interaction
    std::vector<madness::Tensor<double>> core_as_integrals_two_body_akln;
    std::vector<madness::Tensor<double>> core_as_integrals_two_body_abak;
    std::vector<madness::Tensor<double>> core_as_integrals_two_body_baak;

    if(calc_akal)
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
            madness::Tensor<double> ints = madness::Tensor<double>(core_orbitals[0].size() + core_orbitals[1].size(), active_orbitals[spin].size(), active_orbitals[spin].size());
            core_as_integrals_two_body_akal.push_back(ints);
            
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
                    }
                }
            }
        }
    }

    if(calc_akla)
    {
        for (int spin = 0; spin < 2; spin++)
        {
            madness::Tensor<double> ints = madness::Tensor<double>(core_orbitals[spin].size(), active_orbitals[spin].size(), active_orbitals[spin].size());
            core_as_integrals_two_body_akla.push_back(ints);
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
                    }
                }
            }
        }
    }

    return std::vector<std::vector<madness::Tensor<double>>>{core_as_integrals_two_body_akal, core_as_integrals_two_body_akla, core_as_integrals_two_body_akln, core_as_integrals_two_body_abak, core_as_integrals_two_body_baak};
}


template class Integrals_open_shell<2>;
template class Integrals_open_shell<3>;