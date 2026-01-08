#include "optimization_open_shell.hpp"

using namespace madness;

template <std::size_t NDIM>
Optimization_open_shell<NDIM>::Optimization_open_shell(MadnessProcess<NDIM>& mp) : madness_process(mp) {
    std::cout.precision(6);
    Integrator = new Integrals_open_shell(mp);
}

template <std::size_t NDIM>
Optimization_open_shell<NDIM>::~Optimization_open_shell() {
    Vnuc.clear();
    for(int spin = 0; spin < 2; spin++)
    {
        frozen_occ_orbs[spin].clear();
        active_orbs[spin].clear();
        orbs_kl[spin].clear();
        coul_orbs_mn[spin].clear();
        orbs_aa[spin].clear();
    }
    delete Integrator;
}

template <std::size_t NDIM>
void Optimization_open_shell<NDIM>::give_potential_and_repulsion(SavedFct<NDIM> potential, double nuclear_repulsion) {
    Vnuc = madness_process.loadfct(potential);
    nuclear_repulsion_energy = nuclear_repulsion;
}


template <std::size_t NDIM>
void Optimization_open_shell<NDIM>::give_initial_orbitals(std::vector<SavedFct<NDIM>> core_alpha_orbitals, std::vector<SavedFct<NDIM>> core_beta_orbitals, std::vector<SavedFct<NDIM>> active_alpha_orbitals, std::vector<SavedFct<NDIM>> active_beta_orbitals) {
    auto start_time = std::chrono::high_resolution_clock::now();

    for (SavedFct<NDIM> orb : core_alpha_orbitals) {
        frozen_occ_orbs[0].push_back(madness_process.loadfct(orb));
    }

    for (SavedFct<NDIM> orb : active_alpha_orbitals) {
        active_orbs[0].push_back(madness_process.loadfct(orb));
    }

    for (SavedFct<NDIM> orb : core_beta_orbitals) {
        frozen_occ_orbs[1].push_back(madness_process.loadfct(orb));
    }

    for (SavedFct<NDIM> orb : active_beta_orbitals) {
        active_orbs[1].push_back(madness_process.loadfct(orb));
    }


    core_dims[0] = frozen_occ_orbs[0].size();
    as_dims[0] = active_orbs[0].size();
    core_dims[1] = frozen_occ_orbs[1].size();
    as_dims[1] = active_orbs[1].size();

    std::cout << as_dims[0] << " active alpha orbitals and " << as_dims[1] << " active beta orbitals" << std::endl;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "GiveInitialOrbitals took " << duration.count() << " seconds" << std::endl;
}


template <std::size_t NDIM>
void Optimization_open_shell<NDIM>::give_rdm_and_rotate_orbitals(std::vector<Numpy2D>& one_rdms, std::vector<Numpy4D>& two_rdms) {

    // Read active space RDMs
    as_one_rdm[0] = open_shell_utils::to_madness(one_rdms[0]);
    as_one_rdm[1] = open_shell_utils::to_madness(one_rdms[1]);
    as_two_rdm[0] = open_shell_utils::to_madness(two_rdms[0]);
    as_two_rdm[2] = open_shell_utils::to_madness(two_rdms[1]); // bbbb and aabb are in different order in python
    as_two_rdm[1] = open_shell_utils::to_madness(two_rdms[2]); // bbbb and aabb are in different order in python



    // Rotate active Space Orbitals
    //alpha-alpha and beta-beta
    for(int spin = 0; spin < 2; spin++)
    {
        madness::Tensor<double> ActiveSpaceRotationMatrix = madness::Tensor<double>(as_dims[spin], as_dims[spin]);
        madness::Tensor<double> evals(as_dims[spin]);
        syev(as_one_rdm[spin], ActiveSpaceRotationMatrix, evals);
        open_shell_utils::sort_eigenpairs_descending(ActiveSpaceRotationMatrix, evals);
        ActiveSpaceRotationMatrices[spin] = ActiveSpaceRotationMatrix;
        open_shell_utils::TransformMatrix(&as_one_rdm[spin], ActiveSpaceRotationMatrix);
        open_shell_utils::TransformTensor(as_two_rdm[spin], ActiveSpaceRotationMatrix);
        active_orbs[spin] = transform(*(madness_process.world), active_orbs[spin], ActiveSpaceRotationMatrix);
    }

    //mixed
    open_shell_utils::Transform_ab_mixed_Tensor(as_two_rdm[2], ActiveSpaceRotationMatrices[0], ActiveSpaceRotationMatrices[1]);
}


template <std::size_t NDIM>
void Optimization_open_shell<NDIM>::calculate_all_integrals() {

    auto start_time = std::chrono::high_resolution_clock::now();

    // Calculate and cache relevant orbital-combinations
    Integrator->update_as_integral_combinations(active_orbs, orbs_kl, coul_orbs_mn);
    Integrator->update_core_integral_combinations(frozen_occ_orbs, orbs_aa);
    auto t1 = std::chrono::high_resolution_clock::now();

    Integral_storage.update_integrals(
        *Integrator,
        frozen_occ_orbs, active_orbs, 
        orbs_kl, coul_orbs_mn, orbs_aa,Vnuc,
        true, true, false);


    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Calculate all integrals: " << std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count()
            << " seconds" << std::endl;

    std::cout << "Orbital Multiplication: " << std::chrono::duration_cast<std::chrono::seconds>(t1 - start_time).count() << " seconds" << std::endl;
}

template <std::size_t NDIM>
void Optimization_open_shell<NDIM>::calculate_energies() {
    auto start_time = std::chrono::high_resolution_clock::now();

    // AS one Particle Part
    double as_one_electron_energy = 0.0;
    for (int spin = 0; spin < 2; ++spin) {
        as_one_electron_energy += open_shell_utils::contract<2>(
            { as_dims[spin], as_dims[spin] },
            [&](const auto& i) { return as_one_rdm[spin](i[0], i[1]); },
            [&](const auto& i) { return Integral_storage.kl(i[0], i[1], spin); }
        );
    }

    // AS two Particle Part alpha_alpha and beta_beta
    double as_two_electron_energy = 0.0;
    for (int spin = 0; spin < 2; ++spin) {
        as_two_electron_energy += 0.5 * open_shell_utils::contract<4>(
            { as_dims[spin], as_dims[spin], as_dims[spin], as_dims[spin] },
            [&](const auto& i) { return as_two_rdm[spin](i[0], i[1], i[2], i[3]);},
            [&](const auto& i) { return Integral_storage.phys_klmn(i[0], i[1], i[2], i[3], spin); }
        );
    }

    // AS two Particle Part alpha_beta
    as_two_electron_energy += open_shell_utils::contract<4>(
        { as_dims[0], as_dims[0], as_dims[1], as_dims[1] },
        [&](const auto& i) { return as_two_rdm[2](i[0], i[1], i[2], i[3]); },
        [&](const auto& i) { return Integral_storage.phys_klmn(i[0], i[1], i[2], i[3], 2); }
    );

    // AS-Core interaction
    double as_core_energy = 0.0;
    for (int spin = 0; spin < 2; ++spin) {
        as_core_energy += open_shell_utils::contract<3>(
            { core_dims[0] + core_dims[1], as_dims[spin], as_dims[spin] },
            [&](const auto& i) { return as_one_rdm[spin](i[1], i[2]); },
            [&](const auto& i) { return Integral_storage.phys_akal(i[0], i[1], i[2], spin); }
        );

        as_core_energy -= open_shell_utils::contract<3>(
            { core_dims[spin], as_dims[spin], as_dims[spin] },
            [&](const auto& i) { return as_one_rdm[spin](i[1], i[2]); },
            [&](const auto& i) { return Integral_storage.phys_akla( i[0], i[1], i[2], spin); }
        );
    }


    // Print results
    double total_energy =
        as_one_electron_energy + as_two_electron_energy + as_core_energy + core_total_energy + nuclear_repulsion_energy;

    print("      Active Space - One electron energy ", as_one_electron_energy);
    print("      Active Space - Two-electron energy ", as_two_electron_energy);
    print("              AS-Core correlation energy ", as_core_energy);
    print("                       Total core energy ", core_total_energy);
    print("                Nuclear repulsion energy ", nuclear_repulsion_energy);
    print("                            Total energy ", total_energy);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "CalculateEnergies took " << duration.count() << " seconds" << std::endl;
}

template <std::size_t NDIM>
void Optimization_open_shell<NDIM>::calculate_lagrange_multiplier() {
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int spin = 0; spin < 2; spin++) {
        madness::Tensor<double> LagrangeMultiplier = madness::Tensor<double>(as_dims[spin], as_dims[spin]);
        for (int z = 0; z < as_dims[spin]; z++) {
            for (int i = 0; i < as_dims[spin]; i++) {
                LagrangeMultiplier(z, i) = calculate_lagrange_multiplier_element_as_as(z, i, spin);
            }
        }
        LagrangeMultiplier_AS_AS[spin] = LagrangeMultiplier;
    }

    if ((core_dims[0] + core_dims[1]) > 0) {
        for (int spin = 0; spin < 2; spin++) {
            madness::Tensor<double> LagrangeMultiplier = madness::Tensor<double>(core_dims[spin], as_dims[spin]);
            for (int z = 0; z < core_dims[spin]; z++) {
                for (int i = 0; i < as_dims[spin]; i++) {
                    LagrangeMultiplier(z, i) = calculate_lagrange_multiplier_element_as_core(z, i, spin);
                }
            }
            LagrangeMultiplier_AS_Core[spin] = LagrangeMultiplier;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "CalculateLagrangeMultiplier took " << duration.count() << " seconds" << std::endl;
}

template <std::size_t NDIM>
double Optimization_open_shell<NDIM>::calculate_lagrange_multiplier_element_as_as(int z, int i, int spin) {

    int op_spin = 1 - spin;
    double element = as_one_rdm[spin](i, i) * Integral_storage.kl(z, i, spin);

    // alpha-alpha / beta-beta
    element += open_shell_utils::contract<3>(
        // I[0]=k, I[1]=l, I[2]=n
        { as_dims[spin], as_dims[spin], as_dims[spin] },
        [&](const auto& I) { return as_two_rdm[spin](I[0], I[1], i, I[2]); },
        [&](const auto& I) { return Integral_storage.phys_klmn(z, I[1], I[0], I[2], spin); }
    );

    //mixed terms
    element += open_shell_utils::contract<3>(
        // I[0]=k (same spin), I[1]=l, I[2]=n (opposite spin)
        { as_dims[spin], as_dims[op_spin], as_dims[op_spin] },
        [&](const auto& I) {
            if (spin == 0) { return as_two_rdm[2](I[0], I[1], i, I[2]); } 
            else { return as_two_rdm[2](I[1], I[0], I[2], i);}
        },
        [&](const auto& I) {
            if (spin == 0) { return Integral_storage.phys_klmn(z, I[1], I[0], I[2], 2);} 
            else { return Integral_storage.phys_klmn(I[1], z, I[2], I[0], 2);}
        }
    );

    // Core - AS interaction
    if((core_dims[0] + core_dims[1]) > 0)
    {
        element += open_shell_utils::contract<2>(
            // I[0]=a (spins mixed), I[1]=k/l
            { core_dims[0] + core_dims[1], as_dims[spin] },
            [&](const auto& I) { return as_one_rdm[spin](I[1], i);},
            [&](const auto& I) { return Integral_storage.phys_akal(I[0], z, I[1], spin);}
        );

        element -= open_shell_utils::contract<2>(
            // I[0]=a (spins mixed), I[1]=k/l
            { core_dims[spin], as_dims[spin] },
            [&](const auto& I) { return as_one_rdm[spin](I[1], i);},
            [&](const auto& I) { return Integral_storage.phys_akla( I[0], I[1], z, spin);}
        );
    }
    return element;
}

template <std::size_t NDIM>
double Optimization_open_shell<NDIM>::calculate_lagrange_multiplier_element_as_core(int z, int i, int spin) {
    int op_spin = 1 - spin;
    double element = as_one_rdm[spin](i, i) * Integral_storage.ak(z, i, spin);

    //akln part
    // alpha-alpha / beta-beta
    element += open_shell_utils::contract<3>(
        { as_dims[spin], as_dims[spin], as_dims[spin] },
        // I[0]=k, I[1]=l, I[2]=n
        [&](const auto& I) { return as_two_rdm[spin](I[0], I[1], i, I[2]); },
        [&](const auto& I) { return Integral_storage.phys_akln(z, I[1], I[0], I[2], spin); }
    );

    //mixed terms
    element += open_shell_utils::contract<3>(
        { as_dims[spin], as_dims[op_spin], as_dims[op_spin] },
        // I[0]=k, I[1]=l, I[2]=n
        [&](const auto& I) {
            if (spin == 0) { return as_two_rdm[2](I[0], I[1], i, I[2]); } 
            else { return as_two_rdm[2](I[1], I[0], I[2], i); }
        },
        [&](const auto& I) {
            if (spin == 0) { return Integral_storage.phys_akln(z, I[1], I[0], I[2], 2);} 
            else { return Integral_storage.phys_akln(z, I[1], I[0], I[2], 3);}
        }
    );

    //abak part
    element += open_shell_utils::contract<2>(
        { as_dims[spin], core_dims[spin] },
        // I[0]=k, I[1]=a
        [&](const auto& I) { return as_one_rdm[spin](I[0], i);},
        [&](const auto& I) { return Integral_storage.phys_abak(I[1], z, I[0], spin);}
    );

    //mixed terms
    element += open_shell_utils::contract<2>(
        { as_dims[spin], core_dims[op_spin] },
        // I[0]=k, I[1]=a
        [&](const auto& I) { return as_one_rdm[spin](I[0], i); },
        [&](const auto& I) {
            if (spin == 0) { return Integral_storage.phys_abak(I[1], z, I[0], 3);} 
            else { return Integral_storage.phys_abak(I[1], z, I[0], 2);}
        }
    );

    //baak part
    element -= open_shell_utils::contract<2>(
        { as_dims[spin], core_dims[spin] },
        // I[0]=k, I[1]=a
        [&](const auto& I) { return as_one_rdm[spin](I[0], i);},
        [&](const auto& I) { return Integral_storage.phys_baak(I[1], z, I[0], spin);}
    );

    return element;
}

template <std::size_t NDIM>
bool Optimization_open_shell<NDIM>::optimize_orbitals(double optimization_thresh, double NO_occupation_thresh, int maxiter, std::string orthonormalization_method) {

    // Calculate initial energy
    calculate_all_integrals();
    if((core_dims[0] + core_dims[1]) > 0){
        core_total_energy = Integrator->compute_core_energy(frozen_occ_orbs, Vnuc, 0);}
    else { core_total_energy = 0;}
    calculate_energies();

    bool converged = false;
    int iterstep = 0;
    while (!converged && iterstep < maxiter) {
        std::cout << "---------------------------------------------------" << std::endl;
        std::cout << "Start iteration step: " << iterstep << std::endl;


        // Update LagrangeMultiplier
        calculate_lagrange_multiplier();

        // Update orbitals
        auto start_orb_update_time = std::chrono::high_resolution_clock::now();
        highest_error = 0;

        std::array<std::vector<int>, 2> as_orbital_indicies_for_update;

        for (int spin = 0; spin < 2; spin++)
        {
            for (int idx = 0; idx < as_dims[spin]; idx++) {
                if (abs(as_one_rdm[spin](idx, idx)) >= NO_occupation_thresh) {
                    as_orbital_indicies_for_update[spin].push_back(idx);
                } else {
                    std::cout << "Skip refinement of spin" << spin << "active space orbital " << idx
                            << ", since the occupation is less than NO_occupation_thresh (" << NO_occupation_thresh << ")"
                            << std::endl;
                }
            }
        }

        std::array<std::vector<Function<double, NDIM>>, 2> AllActiveOrbitalUpdates = get_all_active_orbital_updates(as_orbital_indicies_for_update);

        for (int spin = 0; spin < 2; spin++)
        {
            for (int idx = 0; idx < as_orbital_indicies_for_update[spin].size(); idx++) {
                int actIdx = as_orbital_indicies_for_update[spin][idx];
                active_orbs[spin][actIdx] = active_orbs[spin][actIdx] - AllActiveOrbitalUpdates[spin][idx];
            }
        }

        auto end_orb_update_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_orb_update_time - start_orb_update_time);
        std::cout << "UpdateOrbitals took " << duration.count() << " seconds" << std::endl;

        // Orthonormalize orbitals
        for(int spin = 0; spin < 2; spin++)
        {
            if(orthonormalization_method == "cd") {active_orbs[spin] = orthonormalize_cd(active_orbs[spin]);}
            else if (orthonormalization_method == "symmetric") {active_orbs[spin] = orthonormalize_symmetric(active_orbs[spin]);}
            else
            {
                std::cout << "Orthonormalization method: yyy not found. Symmetric orthonormalization is used." << std::endl;
                active_orbs[spin] = orthonormalize_symmetric(active_orbs[spin]);
            }
            active_orbs[spin] = truncate(active_orbs[spin], truncation_tol);
        }

        // Check convergence
        std::cout << "Highest error: " << highest_error << std::endl;
        if (highest_error < optimization_thresh) {
            converged = true;
        }

        std::cout << "Update Integrals" << std::endl;
        // Update integrals for new orbitals
        calculate_all_integrals();

        // Calculate new energy
        calculate_energies();

        iterstep++;
    }
    return converged;
}

template <std::size_t NDIM>
std::array<std::vector<Function<double, NDIM>>, 2> Optimization_open_shell<NDIM>::get_all_active_orbital_updates(std::array<std::vector<int>, 2> orbital_indicies_for_update) {
    
    auto start = std::chrono::high_resolution_clock::now();
    std::array<std::vector<Function<double, NDIM>>, 2> AllOrbitalUpdates;

    auto t1 = std::chrono::high_resolution_clock::now();

    // Calculate rdm_ii_inv values
    std::array<std::vector<double>, 2> rdm_ii_inv;
    for (int spin = 0; spin < 2; spin++)
    {
        for (int idx = 0; idx < orbital_indicies_for_update[spin].size(); idx++) {
            int i = orbital_indicies_for_update[spin][idx];
            rdm_ii_inv[spin].push_back(1 / as_one_rdm[spin](i, i));
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    // 1e Part
    for (int spin = 0; spin < 2; spin++)
    {
        for (int idx = 0; idx < orbital_indicies_for_update[spin].size(); idx++) {
            int i = orbital_indicies_for_update[spin][idx];

            Function<double, NDIM> rhs;
            rhs = Vnuc * active_orbs[spin][i];
            for (int k = 0; k < core_dims[spin]; k++) {
               rhs -= rdm_ii_inv[spin][idx] * LagrangeMultiplier_AS_Core[spin](k, i) * frozen_occ_orbs[spin][k];
            }
            for (int k = 0; k < as_dims[spin]; k++) {
                if (k != i) {
                    rhs -= rdm_ii_inv[spin][idx] * LagrangeMultiplier_AS_AS[spin](k, i) * active_orbs[spin][k];
                }
            }
            AllOrbitalUpdates[spin].push_back(rhs);
        }
        AllOrbitalUpdates[spin] = truncate(AllOrbitalUpdates[spin], truncation_tol);
    }
    auto t3 = std::chrono::high_resolution_clock::now();


    // AS Part
    for (int spin = 0; spin < 2; spin++)
    {
        //alpha-alpha and beta-beta
        for (int k = 0; k < as_dims[spin]; k++) {
            std::vector<Function<double, NDIM>> lnk = coul_orbs_mn[spin] * active_orbs[spin][k];
            lnk = truncate(lnk, truncation_tol);
            for (int idx = 0; idx < orbital_indicies_for_update[spin].size(); idx++) {
                int i = orbital_indicies_for_update[spin][idx];
                std::vector<Function<double, NDIM>> lnk_copy = copy(*(madness_process.world), lnk, false);
                for (int l = 0; l < as_dims[spin]; l++) {
                    for (int n = 0; n < as_dims[spin]; n++) {
                        lnk_copy[l * as_dims[spin] + n] *= as_two_rdm[spin](k, l, i, n) * rdm_ii_inv[spin][idx];
                    }
                }
                AllOrbitalUpdates[spin][idx] += sum(*(madness_process.world), lnk_copy);
            }
        }

        //mixed terms
        int op_spin; if(spin == 0){op_spin = 1;} else {op_spin = 0;}
        for (int k = 0; k < as_dims[spin]; k++) {
            std::vector<Function<double, NDIM>> lnk = coul_orbs_mn[op_spin] * active_orbs[spin][k];
            lnk = truncate(lnk, truncation_tol);
            for (int idx = 0; idx < orbital_indicies_for_update[spin].size(); idx++) {
                int i = orbital_indicies_for_update[spin][idx];
                std::vector<Function<double, NDIM>> lnk_copy = copy(*(madness_process.world), lnk, false);
                for (int l = 0; l < as_dims[op_spin]; l++) {
                    for (int n = 0; n < as_dims[op_spin]; n++) {
                        if(spin == 0)
                        {
                            lnk_copy[l * as_dims[op_spin] + n] *= as_two_rdm[2](k, l, i, n) * rdm_ii_inv[spin][idx];
                        }
                        else
                        {
                            lnk_copy[l * as_dims[op_spin] + n] *= as_two_rdm[2](l, k, n, i) * rdm_ii_inv[spin][idx];
                        }
                    }
                }
                AllOrbitalUpdates[spin][idx] += sum(*(madness_process.world), lnk_copy);
            }
        }
    }
    auto t4 = std::chrono::high_resolution_clock::now();

    // Core - AS interaction
    auto coul_op_parallel =
        std::shared_ptr<SeparatedConvolution<double, NDIM>>(CoulombOperatorNDPtr<NDIM>(*(madness_process.world), 0.001, 1e-6));

    if((core_dims[0] + core_dims[1]) > 0)
    {
        // Part 1
        std::vector<Function<double, NDIM>> all_orbs_aa;
        all_orbs_aa.insert(all_orbs_aa.end(), orbs_aa[0].begin(), orbs_aa[0].end());
        all_orbs_aa.insert(all_orbs_aa.end(), orbs_aa[1].begin(), orbs_aa[1].end());

        std::vector<Function<double, NDIM>> coul_all_orbs_aa = apply(*(madness_process.world), *coul_op_parallel, all_orbs_aa);
        coul_all_orbs_aa = truncate(coul_all_orbs_aa, truncation_tol);

        for (int spin = 0; spin < 2; spin++) {
            for (int k = 0; k < as_dims[spin]; k++) {
                std::vector<Function<double, NDIM>> aak = coul_all_orbs_aa * active_orbs[spin][k];
                aak = truncate(aak, truncation_tol);
                for (int idx = 0; idx < orbital_indicies_for_update[spin].size(); idx++) {
                    int i = orbital_indicies_for_update[spin][idx];
                    std::vector<Function<double, NDIM>> aak_copy = copy(*(madness_process.world), aak, false);
                    for (int a = 0; a < coul_all_orbs_aa.size(); a++) {
                        aak_copy[a] *= as_one_rdm[spin](k, i) * rdm_ii_inv[spin][idx];
                    }
                    AllOrbitalUpdates[spin][idx] += sum(*(madness_process.world), aak_copy);
                }
            }
        }

        //Part 2
        for (int spin = 0; spin < 2; spin++) {
            for (int a = 0; a < core_dims[spin]; a++) {
                std::vector<Function<double, NDIM>> orbs_ak = frozen_occ_orbs[spin][a] * active_orbs[spin];
                orbs_ak = truncate(orbs_ak, truncation_tol);
                std::vector<Function<double, NDIM>> coul_orbs_ak = apply(*(madness_process.world), *coul_op_parallel, orbs_ak);
                coul_orbs_ak = truncate(coul_orbs_ak, truncation_tol);

                std::vector<Function<double, NDIM>> aka = coul_orbs_ak * frozen_occ_orbs[spin][a];
                aka = truncate(aka, truncation_tol);

                for (int idx = 0; idx < orbital_indicies_for_update[spin].size(); idx++) {
                    int i = orbital_indicies_for_update[spin][idx];
                    std::vector<Function<double, NDIM>> aka_copy = copy(*(madness_process.world), aka, false);
                    for (int k = 0; k < as_dims[spin]; k++) {
                        aka_copy[k] *= -1 *  as_one_rdm[spin](k, i) * rdm_ii_inv[spin][idx];
                    }
                    AllOrbitalUpdates[spin][idx] += sum(*(madness_process.world), aka_copy);
                }
            }
        }
    }
    auto t5 = std::chrono::high_resolution_clock::now();

    // BSH part
    for (int spin = 0; spin < 2; spin++)
    {
        for (int idx = 0; idx < orbital_indicies_for_update[spin].size(); idx++) {
            int i = orbital_indicies_for_update[spin][idx];
            double en = LagrangeMultiplier_AS_AS[spin](i, i) * rdm_ii_inv[spin][idx];
            SeparatedConvolution<double, NDIM> bsh_op =
                BSHOperator<NDIM>(*(madness_process.world), sqrt(-2 * en), BSH_lo, BSH_eps);
            Function<double, NDIM> r = active_orbs[spin][i] + 2.0 * bsh_op(AllOrbitalUpdates[spin][idx]); // the residual
            double err = r.norm2();
            std::cout << "Error of spin " << spin << " active space orbital " << i << ": " << err << std::endl;
            if (err > highest_error) {
                highest_error = err;
            }
            AllOrbitalUpdates[spin][idx] = r;
        }
    }
    auto t6 = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Refinement timings:" << std::endl;
    std::cout << "rdm_ii_inv calculation: " << std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count()
              << " seconds" << std::endl;
    std::cout << "one electron part: " << std::chrono::duration_cast<std::chrono::seconds>(t3 - t2).count()
              << " seconds" << std::endl;
    std::cout << "AS two electron part: " << std::chrono::duration_cast<std::chrono::seconds>(t4 - t3).count()
              << " seconds" << std::endl;
    std::cout << "Core-AS part: " << std::chrono::duration_cast<std::chrono::seconds>(t5 - t4).count()
              << " seconds" << std::endl;          
    std::cout << "BSH part: " << std::chrono::duration_cast<std::chrono::seconds>(t6 - t5).count() 
              << " seconds" << std::endl;   
    std::cout << "Full function: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
              << " seconds" << std::endl;
    return AllOrbitalUpdates;
}

template <std::size_t NDIM>
void Optimization_open_shell<NDIM>::rotate_orbitals_back() {
    std::array<madness::Tensor<double>, 2> RotationMatricesBack;
    for(int spin = 0; spin < 2; spin++)
    {
        madness::Tensor<double> RotationMatrixBack = madness::Tensor<double>(as_dims[spin], as_dims[spin]);
        for (int i = 0; i < as_dims[spin]; i++) {
            for (int j = 0; j < as_dims[spin]; j++) {
                RotationMatrixBack(i, j) = ActiveSpaceRotationMatrices[spin](j, i);
            }
        }
        RotationMatricesBack[spin] = RotationMatrixBack;
        open_shell_utils::TransformMatrix(&as_one_rdm[spin], RotationMatrixBack);
        open_shell_utils::TransformTensor(as_two_rdm[spin], RotationMatrixBack);
        active_orbs[spin] = transform(*(madness_process.world), active_orbs[spin], RotationMatrixBack);
    }
    open_shell_utils::Transform_ab_mixed_Tensor(as_two_rdm[2], RotationMatricesBack[0], RotationMatricesBack[1]);
    calculate_all_integrals();
    calculate_energies();
}

template <std::size_t NDIM>
std::vector<std::vector<SavedFct<NDIM>>> Optimization_open_shell<NDIM>::get_orbitals() {
    std::vector<std::vector<SavedFct<NDIM>>> all_orbs;
    for(int spin = 0; spin < 2; spin++) {
        std::vector<SavedFct<NDIM>> core_orbs;
        for (int i = 0; i < core_dims[spin]; i++) {
            SavedFct<NDIM> orb(frozen_occ_orbs[spin][i]);
            orb.type = "frozen_occ";
            core_orbs.push_back(orb);
            std::cout << "Norm spin: " << spin << " core orb " << i << ":" << frozen_occ_orbs[spin][i].norm2() << std::endl;
        }
        all_orbs.push_back(core_orbs);
    }

    for(int spin = 0; spin < 2; spin++) {
        std::vector<SavedFct<NDIM>> as_orbs;
        for (int i = 0; i < as_dims[spin]; i++) {
            SavedFct<NDIM> orb(active_orbs[spin][i]);
            orb.type = "active";
            as_orbs.push_back(orb);
            std::cout << "Norm spin: " << spin << " as orb " << i << ":" << active_orbs[spin][i].norm2() << std::endl;
        }
        all_orbs.push_back(as_orbs);
    }
    return all_orbs;
}



template <std::size_t NDIM>
nb::tuple Optimization_open_shell<NDIM>::get_effective_hamiltonian() {

    // Core energy
    double effective_hamiltonian_core_energy = core_total_energy + nuclear_repulsion_energy;

    // Effective one electron integrals
    std::array<madness::Tensor<double>, 2> effective_one_e_integrals;
    effective_one_e_integrals[0] = Integral_storage.as_integrals_one_body[0];
    effective_one_e_integrals[1] = Integral_storage.as_integrals_one_body[1];

    //Core-AS interaction
    Integrator->update_as_integral_combinations(active_orbs, orbs_kl, coul_orbs_mn);
    Integrator->update_core_integral_combinations(frozen_occ_orbs, orbs_aa);
    std::vector<std::vector<madness::Tensor<double>>> core_as_integrals_two_body = Integrator->compute_core_as_integrals_two_body(frozen_occ_orbs, active_orbs, orbs_kl, coul_orbs_mn, orbs_aa, true, true, false, false, false);

    if(core_dims[0] > 0 || core_dims[1] > 0)
    {
        for (int spin = 0; spin < 2; spin++) {
            for (int a = 0; a < (core_dims[0] + core_dims[1]); a++) {
                for (int k = 0; k < as_dims[spin]; k++) {
                    for (int l = 0; l < as_dims[spin]; l++) {
                        effective_one_e_integrals[spin](k,l) += core_as_integrals_two_body[0][spin](a, k, l);
                    }
                }
            }
        }
        for (int spin = 0; spin < 2; spin++) {
            for (int a = 0; a < core_dims[spin]; a++) {
                for (int k = 0; k < as_dims[spin]; k++) {
                    for (int l = 0; l < as_dims[spin]; l++) {
                        effective_one_e_integrals[spin](k,l) -= core_as_integrals_two_body[1][spin](a, k, l);
                    }
                }
            }
        }
    }

    // Capsule integrals for python
    auto one_e_alpha_owner = std::make_shared<madness::Tensor<double>>(std::move(effective_one_e_integrals[0]));
    nb::capsule one_e_alpha_caps(
        new std::shared_ptr<madness::Tensor<double>>(one_e_alpha_owner),
        [](void *p) noexcept {
            delete reinterpret_cast<std::shared_ptr<madness::Tensor<double>>*>(p);
        }
    );

    auto one_e_beta_owner = std::make_shared<madness::Tensor<double>>(std::move(effective_one_e_integrals[1]));
    nb::capsule one_e_beta_caps(
        new std::shared_ptr<madness::Tensor<double>>(one_e_beta_owner),
        [](void *p) noexcept {
            delete reinterpret_cast<std::shared_ptr<madness::Tensor<double>>*>(p);
        }
    );


    auto two_e_aa_owner = std::make_shared<madness::Tensor<double>>(std::move(Integral_storage.as_integrals_two_body[0]));
    nb::capsule two_e_aa_caps(
        new std::shared_ptr<madness::Tensor<double>>(two_e_aa_owner),
        [](void *p) noexcept {
            delete reinterpret_cast<std::shared_ptr<madness::Tensor<double>>*>(p);
        }
    );

    auto two_e_bb_owner = std::make_shared<madness::Tensor<double>>(std::move(Integral_storage.as_integrals_two_body[1]));
    nb::capsule two_e_bb_caps(
        new std::shared_ptr<madness::Tensor<double>>(two_e_bb_owner),
        [](void *p) noexcept {
            delete reinterpret_cast<std::shared_ptr<madness::Tensor<double>>*>(p);
        }
    );

    auto two_e_ab_owner = std::make_shared<madness::Tensor<double>>(std::move(Integral_storage.as_integrals_two_body[2]));
    nb::capsule two_e_ab_caps(
        new std::shared_ptr<madness::Tensor<double>>(two_e_ab_owner),
        [](void *p) noexcept {
            delete reinterpret_cast<std::shared_ptr<madness::Tensor<double>>*>(p);
        }
    );



    // Return integrals
    Numpy2D one_e_alpha_alpha(effective_one_e_integrals[0].ptr(), {as_dims[0], as_dims[0]}, one_e_alpha_caps);
    Numpy2D one_e_beta_beta(effective_one_e_integrals[1].ptr(), {as_dims[1], as_dims[1]}, one_e_beta_caps);
    nb::list h1_list;
    h1_list.append(one_e_alpha_alpha);
    h1_list.append(one_e_beta_beta);

    Numpy4D alpha_alpha(Integral_storage.as_integrals_two_body[0].ptr(), {as_dims[0], as_dims[0], as_dims[0], as_dims[0]}, two_e_aa_caps);
    Numpy4D beta_beta(Integral_storage.as_integrals_two_body[1].ptr(), {as_dims[1], as_dims[1], as_dims[1], as_dims[1]}, two_e_bb_caps);
    Numpy4D alpha_beta(Integral_storage.as_integrals_two_body[2].ptr(), {as_dims[0], as_dims[1], as_dims[0], as_dims[1]}, two_e_ab_caps);
    //aa, ab, bb order external; aa, bb, ab order internal
    nb::list g2_list;
    g2_list.append(alpha_alpha);
    g2_list.append(alpha_beta);
    g2_list.append(beta_beta);

    return nb::make_tuple(effective_hamiltonian_core_energy, h1_list, g2_list);
}

template class Optimization_open_shell<2>;
template class Optimization_open_shell<3>;