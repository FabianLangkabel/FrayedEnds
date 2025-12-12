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


template <std::size_t NDIM>
void Integrals_open_shell<NDIM>::update_as_integral_combinations(std::array<std::vector<Function<double, NDIM>>, 2> &orbitals, std::array<std::vector<Function<double, NDIM>>, 2> &orbs_kl, std::array<std::vector<Function<double, NDIM>>, 2> &coul_orbs_mn)
{
    double truncation_tol = 1e-6;
    double coulomb_lo = 0.001;
    double coulomb_eps = 1e-6;
    auto coul_op_parallel = std::shared_ptr<SeparatedConvolution<double, NDIM>>(CoulombOperatorNDPtr<NDIM>(*(madness_process.world), coulomb_lo, coulomb_eps));

    for(int spin = 0; spin < 2; spin++)
    {
        orbs_kl[spin].clear();
        coul_orbs_mn[spin].clear();
        for (int k = 0; k < orbitals[spin].size(); k++) {
            std::vector<Function<double, NDIM>> kl = orbitals[spin][k] * orbitals[spin];
            orbs_kl[spin].insert(std::end(orbs_kl[spin]), std::begin(kl), std::end(kl));
        }
        orbs_kl[spin] = truncate(orbs_kl[spin], truncation_tol);
        coul_orbs_mn[spin] = apply(*(madness_process.world), *coul_op_parallel, orbs_kl[spin]);
        coul_orbs_mn[spin] = truncate(coul_orbs_mn[spin], truncation_tol);
    }
}

template <std::size_t NDIM>
void Integrals_open_shell<NDIM>::update_core_integral_combinations(std::array<std::vector<Function<double, NDIM>>, 2> &core_orbitals, std::array<std::vector<Function<double, NDIM>>, 2> &orbs_aa)
{
    double truncation_tol = 1e-6;
    double coulomb_lo = 0.001;
    double coulomb_eps = 1e-6;
    auto coul_op_parallel = std::shared_ptr<SeparatedConvolution<double, NDIM>>(CoulombOperatorNDPtr<NDIM>(*(madness_process.world), coulomb_lo, coulomb_eps));

    for (int spin = 0; spin < 2; spin++)
    {
        orbs_aa[spin].clear();
        for (int a = 0; a < core_orbitals[spin].size(); a++) {
            orbs_aa[spin].push_back(core_orbitals[spin][a] * core_orbitals[spin][a]);   
        }
        orbs_aa[spin] = truncate(orbs_aa[spin], truncation_tol);
    }
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

    std::array<std::vector<Function<double, NDIM>>, 2> orbs_kl;
    std::array<std::vector<Function<double, NDIM>>, 2> coul_orbs_mn;
    update_as_integral_combinations(orbitals, orbs_kl, coul_orbs_mn);
    std::array<madness::Tensor<double>, 3> Integrals = compute_two_body_integrals(orbitals, orbs_kl, coul_orbs_mn);

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

    std::array<std::vector<Function<double, NDIM>>, 2> orbs_kl;
    std::array<std::vector<Function<double, NDIM>>, 2> coul_orbs_mn;
    update_as_integral_combinations(active_orbitals, orbs_kl, coul_orbs_mn);
    std::array<madness::Tensor<double>, 3> two_e_integrals = compute_two_body_integrals(active_orbitals, orbs_kl, coul_orbs_mn);

    // Core interactions
    double effective_hamiltonian_core_energy = energy_offset;
    if(core_orbitals[0].size() > 0 || core_orbitals[1].size() > 0)
    {
        // Core energy
        effective_hamiltonian_core_energy = compute_core_energy(core_orbitals, V, energy_offset);

        //Core-AS interaction
        std::array<std::vector<Function<double, NDIM>>, 2> orbs_aa;
        update_core_integral_combinations(core_orbitals, orbs_aa);

        std::vector<std::vector<madness::Tensor<double>>> core_as_integrals_two_body = compute_core_as_integrals_two_body(
            core_orbitals, active_orbitals, orbs_kl, coul_orbs_mn, orbs_aa, true, true, false, false, false);

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
std::array<madness::Tensor<double>, 3> Integrals_open_shell<NDIM>::compute_two_body_integrals(std::array<std::vector<Function<double, NDIM>>, 2> &orbitals, std::array<std::vector<Function<double, NDIM>>, 2> &orbs_kl, std::array<std::vector<Function<double, NDIM>>, 2> &coul_orbs_mn) 
{
    std::array<madness::Tensor<double>, 3> Integrals;

    int spincombination[3][2] = {{0, 0}, {1, 1}, {0, 1}}; // in chemical notation (aa|aa), (bb|bb), (aa|bb)
    for(int c = 0; c < 3; c++)
    {
        int bra_spin = spincombination[c][0];
        int ket_spin = spincombination[c][1];

        madness::Tensor<double> ints = madness::Tensor<double>(orbitals[bra_spin].size(), orbitals[ket_spin].size(), orbitals[bra_spin].size(), orbitals[ket_spin].size());
        madness::Tensor<double> Inner_prods = matrix_inner(*(madness_process.world), orbs_kl[bra_spin], coul_orbs_mn[ket_spin], false);
        for (int k = 0; k < orbitals[bra_spin].size(); k++) {
            for (int l = 0; l < orbitals[ket_spin].size(); l++) {
                for (int m = 0; m < orbitals[bra_spin].size(); m++) {
                    for (int n = 0; n < orbitals[ket_spin].size(); n++) {
                        auto tmp = Inner_prods(k * orbitals[bra_spin].size() + l, m * orbitals[ket_spin].size() + n);
                        ints(k, m, l, n) = tmp;
                    }
                }
            }
        }
        Integrals[c] = ints;
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
std::array<madness::Tensor<double>, 2> Integrals_open_shell<NDIM>::compute_core_as_integrals_one_body(
    std::array<std::vector<Function<double, NDIM>>, 2> core_orbitals, std::array<std::vector<Function<double, NDIM>>, 2> active_orbitals, Function<double, NDIM> V
)
{
    std::array<madness::Tensor<double>, 2> Integrals;
    for(int spin = 0; spin < 2; spin++)
    {
        Integrals[spin] = madness::matrix_inner(*(madness_process.world), core_orbitals[spin], V * active_orbitals[spin]);
        for (int k = 0; k < core_orbitals[spin].size(); k++) {
            for (int l = 0; l < active_orbitals[spin].size(); l++) {
                for (int axis = 0; axis < NDIM; axis++) {
                    Derivative<double, NDIM> D = free_space_derivative<double, NDIM>(*(madness_process.world), axis);
                    Function<double, NDIM> d_orb_k = D(core_orbitals[spin][k]);
                    Function<double, NDIM> d_orb_l = D(active_orbitals[spin][l]);
                    Integrals[spin](k, l) += 0.5 * inner(d_orb_k, d_orb_l);
                }
            }
        }
    }
    return Integrals;
}

template <std::size_t NDIM>
std::vector<std::vector<madness::Tensor<double>>> Integrals_open_shell<NDIM>::compute_core_as_integrals_two_body(
    std::array<std::vector<Function<double, NDIM>>, 2> &core_orbitals, 
    std::array<std::vector<Function<double, NDIM>>, 2> &active_orbitals, 
    std::array<std::vector<Function<double, NDIM>>, 2> &orbs_kl, 
    std::array<std::vector<Function<double, NDIM>>, 2> &coul_orbs_mn, 
    std::array<std::vector<Function<double, NDIM>>, 2> &orbs_aa, 
    bool calc_akal, bool calc_akla, bool calc_akln, bool calc_abak, bool calc_baak
)
{
    double truncation_tol = 1e-6;
    double coulomb_lo = 0.001;
    double coulomb_eps = 1e-6;
    auto coul_op_parallel = std::shared_ptr<SeparatedConvolution<double, NDIM>>(CoulombOperatorNDPtr<NDIM>(*(madness_process.world), coulomb_lo, coulomb_eps));

    std::vector<madness::Tensor<double>> core_as_integrals_two_body_akal; //stored as (a,k,l); a is alpha + beta list; no alpha-beta-interaction
    std::vector<madness::Tensor<double>> core_as_integrals_two_body_akla; //stored as (a,k,l); a is list of [spin]; no alpha-beta-interaction
    std::vector<madness::Tensor<double>> core_as_integrals_two_body_akln; //stored as (a,k,l,n) 4 tensors: aaaa, bbbb, aabb, bbaa
    std::vector<madness::Tensor<double>> core_as_integrals_two_body_abak; //stored as (a,b,k); 4 tensors: aaaa, bbbb, aabb, bbaa
    std::vector<madness::Tensor<double>> core_as_integrals_two_body_baak; //stored as (a,b,k); 2 tensors: aaaa, bbbb

    auto t1 = std::chrono::high_resolution_clock::now();
    if(calc_akal)
    {
        std::vector<Function<double, NDIM>> all_orbs_aa;
        all_orbs_aa.insert(all_orbs_aa.end(), orbs_aa[0].begin(), orbs_aa[0].end());
        all_orbs_aa.insert(all_orbs_aa.end(), orbs_aa[1].begin(), orbs_aa[1].end());

        for (int spin = 0; spin < 2; spin++)
        {
            madness::Tensor<double> ints = madness::Tensor<double>(core_orbitals[0].size() + core_orbitals[1].size(), active_orbitals[spin].size(), active_orbitals[spin].size());
            madness::Tensor<double> Inner_prods_akal = matrix_inner(*(madness_process.world), all_orbs_aa, coul_orbs_mn[spin], false);
            for (int a = 0; a < all_orbs_aa.size(); a++) {
                for (int k = 0; k < active_orbitals[spin].size(); k++) {
                    for (int l = 0; l < active_orbitals[spin].size(); l++) {
                        ints(a, k, l) = Inner_prods_akal(a, k * active_orbitals[spin].size() + l);
                    }
                }
            }
            core_as_integrals_two_body_akal.push_back(ints);
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    if(calc_akla)
    {
        for (int spin = 0; spin < 2; spin++)
        {
            madness::Tensor<double> ints = madness::Tensor<double>(core_orbitals[spin].size(), active_orbitals[spin].size(), active_orbitals[spin].size());
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
                        ints(a, k, l) = Inner_prods_akla(l, k);
                    }
                }
            }
            core_as_integrals_two_body_akla.push_back(ints);
        }
    }

    auto t3 = std::chrono::high_resolution_clock::now();
    if(calc_akln)
    {

        int spincombination[4][2] = {{0, 0}, {1, 1}, {0, 1}, {1, 0}}; // in chemical notation (aa|aa), (bb|bb), (aa|bb), (bb|aa)
        for(int c = 0; c < 4; c++)
        {
            int bra_spin = spincombination[c][0];
            int ket_spin = spincombination[c][1];

            madness::Tensor<double> ints = madness::Tensor<double>(core_orbitals[bra_spin].size(), active_orbitals[ket_spin].size(), active_orbitals[bra_spin].size(), active_orbitals[ket_spin].size());
            for (int a = 0; a < core_orbitals[bra_spin].size(); a++)
            {
                std::vector<Function<double, NDIM>> orbs_ak = core_orbitals[bra_spin][a] * active_orbitals[bra_spin];
                orbs_ak = truncate(orbs_ak, truncation_tol);

                madness::Tensor<double> Inner_prods_akln = matrix_inner(*(madness_process.world), orbs_ak, coul_orbs_mn[ket_spin], false);
                for (int k = 0; k < active_orbitals[bra_spin].size(); k++) {
                    for (int l = 0; l < active_orbitals[ket_spin].size(); l++) {
                        for (int n = 0; n < active_orbitals[ket_spin].size(); n++) {
                            ints(a, l, k, n) = Inner_prods_akln(k, l * active_orbitals[ket_spin].size() + n);
                        }
                    }
                }
            }
            core_as_integrals_two_body_akln.push_back(ints);
        }
    }

    auto t4 = std::chrono::high_resolution_clock::now();
    if(calc_abak)
    {
        int spincombination[4][2] = {{0, 0}, {1, 1}, {0, 1}, {1, 0}}; // in chemical notation (aa|aa), (bb|bb), (aa|bb), (bb|aa)
        for(int c = 0; c < 4; c++)
        {
            int bra_spin = spincombination[c][0];
            int ket_spin = spincombination[c][1];

            madness::Tensor<double> ints = madness::Tensor<double>(core_orbitals[bra_spin].size(), core_orbitals[bra_spin].size(), active_orbitals[ket_spin].size());
            for (int b = 0; b < core_orbitals[ket_spin].size(); b++)
            {
                std::vector<Function<double, NDIM>> orbs_bk = core_orbitals[ket_spin][b] * active_orbitals[ket_spin];
                orbs_bk = truncate(orbs_bk, truncation_tol);
                std::vector<Function<double, NDIM>> coul_orbs_bk = apply(*(madness_process.world), *coul_op_parallel, orbs_bk);
                coul_orbs_bk = truncate(coul_orbs_bk, truncation_tol);

                madness::Tensor<double> Inner_prods_abak = matrix_inner(*(madness_process.world), orbs_aa[bra_spin], coul_orbs_bk, false);
                for (int a = 0; a < core_orbitals[bra_spin].size(); a++) {
                    for (int k = 0; k < active_orbitals[ket_spin].size(); k++) {
                        ints(a,b,k) = Inner_prods_abak(a, k);
                    }
                }
            }
            core_as_integrals_two_body_abak.push_back(ints);
        }
    }

    auto t5 = std::chrono::high_resolution_clock::now();
    if(calc_baak)
    {
        for (int spin = 0; spin < 2; spin++)
        {
            madness::Tensor<double> ints = madness::Tensor<double>(core_orbitals[spin].size(), core_orbitals[spin].size(), active_orbitals[spin].size());
            for (int a = 0; a < core_orbitals[spin].size(); a++)
            {
                std::vector<Function<double, NDIM>> orbs_ak = core_orbitals[spin][a] * active_orbitals[spin];
                orbs_ak = truncate(orbs_ak, truncation_tol);
                std::vector<Function<double, NDIM>> coul_orbs_ak = apply(*(madness_process.world), *coul_op_parallel, orbs_ak);
                coul_orbs_ak = truncate(coul_orbs_ak, truncation_tol);

                for (int b = 0; b < core_orbitals[spin].size(); b++)
                {
                    std::vector<Function<double, NDIM>> ba;
                    ba.push_back(core_orbitals[spin][b] * core_orbitals[spin][a]);
                    madness::Tensor<double> Inner_prods_baak = matrix_inner(*(madness_process.world), ba, coul_orbs_ak, false);
                    for (int k = 0; k < active_orbitals[spin].size(); k++) {
                        ints(a,b,k) = Inner_prods_baak(0, k);
                    }
                }
            }
            core_as_integrals_two_body_baak.push_back(ints);
        }
    }
    auto t6 = std::chrono::high_resolution_clock::now();

    std::cout << "akal: " << std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count() << " seconds" << std::endl;
    std::cout << "akla: " << std::chrono::duration_cast<std::chrono::seconds>(t3 - t2).count() << " seconds" << std::endl;
    std::cout << "akln: " << std::chrono::duration_cast<std::chrono::seconds>(t4 - t3).count() << " seconds" << std::endl;
    std::cout << "abak: " << std::chrono::duration_cast<std::chrono::seconds>(t5 - t4).count() << " seconds" << std::endl;
    std::cout << "baak: " << std::chrono::duration_cast<std::chrono::seconds>(t6 - t5).count() << " seconds" << std::endl;

    return std::vector<std::vector<madness::Tensor<double>>>{core_as_integrals_two_body_akal, core_as_integrals_two_body_akla, core_as_integrals_two_body_akln, core_as_integrals_two_body_abak, core_as_integrals_two_body_baak};
}


template class Integrals_open_shell<2>;
template class Integrals_open_shell<3>;