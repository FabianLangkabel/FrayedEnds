#include "optimization.hpp"

using namespace madness;

template <std::size_t NDIM>
Optimization<NDIM>::Optimization(MadnessProcess<NDIM>& mp) : madness_process(mp), Integrator(mp) {
    std::cout.precision(6);
    Integrator.override_numerical_parameters(num_params);
}

template <std::size_t NDIM>
Optimization<NDIM>::~Optimization() {
    Vnuc.clear();
    frozen_occ_orbs.clear();
    active_orbs.clear();
    orbs_kl.clear();
    coul_orbs_mn.clear();
    orbs_aa.clear();
    coul_orbs_aa.clear();
}

template <std::size_t NDIM>
void Optimization<NDIM>::give_potential_and_repulsion(SavedFct<NDIM> potential, double nuclear_repulsion) {
    Vnuc = madness_process.loadfct(potential);
    nuclear_repulsion_energy = nuclear_repulsion;
}

template <std::size_t NDIM> void Optimization<NDIM>::give_initial_orbitals(std::vector<SavedFct<NDIM>> fr_core_orbs, std::vector<SavedFct<NDIM>> act_orbs) {
    auto start_time = std::chrono::high_resolution_clock::now();

    for (SavedFct<NDIM> orb : fr_core_orbs) {
            frozen_occ_orbs.push_back(madness_process.loadfct(orb));
    }
    for (SavedFct<NDIM> orb : act_orbs) {
        active_orbs.push_back(madness_process.loadfct(orb));
    }
    core_dim = frozen_occ_orbs.size();
    as_dim = active_orbs.size();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "GiveInitialOrbitals took " << duration.count() << " seconds" << std::endl;
}

template <std::size_t NDIM>
void Optimization<NDIM>::give_rdm_and_rotate_orbitals(Numpy2D& one_rdms, Numpy4D& two_rdms) {
    auto start_time = std::chrono::high_resolution_clock::now();

    //****************************************
    // Read active space RDMs
    //****************************************
    as_one_rdm = refinement_utils::to_madness(one_rdms);
    as_two_rdm = refinement_utils::to_madness(two_rdms);

    //****************************************
    // Rotate active Space Orbitals
    //****************************************
    ActiveSpaceRotationMatrix = madness::Tensor<double>(as_dim, as_dim);
    Tensor<double> evals(as_dim);
    syev(as_one_rdm, ActiveSpaceRotationMatrix, evals);
    refinement_utils::sort_eigenpairs_descending(ActiveSpaceRotationMatrix, evals);
    refinement_utils::TransformMatrix(&as_one_rdm, ActiveSpaceRotationMatrix);
    refinement_utils::TransformTensor(as_two_rdm, ActiveSpaceRotationMatrix);
    active_orbs = transform(*(madness_process.world), active_orbs, ActiveSpaceRotationMatrix);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "GiveRDMs took " << duration.count() << " seconds" << std::endl;
}

template <std::size_t NDIM>
void Optimization<NDIM>::calculate_all_integrals(bool update_aa) {

    auto start_time = std::chrono::high_resolution_clock::now();

    // Calculate and cache relevant orbital-combinations
    Integrator.update_as_integral_combinations(active_orbs, orbs_kl, coul_orbs_mn);
    
    auto t1 = std::chrono::high_resolution_clock::now();
    // Calculate one electron Integrals
    as_integrals_one_body = Integrator.compute_potential_integrals(active_orbs, Vnuc);
    as_integrals_one_body += Integrator.compute_kinetic_integrals(active_orbs);

    // Calculate two electron Integrals
    as_integrals_two_body = Integrator.compute_two_body_integrals(active_orbs, orbs_kl, coul_orbs_mn);

    // Calculate Core-AS interaction integrals
    if (core_dim > 0) {
        if (update_aa) {
            Integrator.update_core_integral_combinations(frozen_occ_orbs, orbs_aa, coul_orbs_aa);
        }
        core_as_integrals_one_body_ak = Integrator.compute_core_as_integrals_one_body(frozen_occ_orbs, active_orbs, Vnuc);
        
        auto core_as_integrals_two_body0 = Integrator.compute_core_as_2e_integrals_as_refinement(frozen_occ_orbs, active_orbs, orbs_kl, coul_orbs_mn, orbs_aa);
        core_as_integrals_two_body_akal = core_as_integrals_two_body0[0];
        core_as_integrals_two_body_akla = core_as_integrals_two_body0[1];
        core_as_integrals_two_body_akln = core_as_integrals_two_body0[2];
        core_as_integrals_two_body_abak = core_as_integrals_two_body0[3];
        core_as_integrals_two_body_baak = core_as_integrals_two_body0[4];

        if (refine_core) {
            core_core_integrals_one_body_ab = Integrator.compute_potential_integrals(frozen_occ_orbs, Vnuc);
            core_core_integrals_one_body_ab += Integrator.compute_kinetic_integrals(frozen_occ_orbs);
            auto core_as_integrals_two_body = Integrator.compute_core_as_2e_integrals_core_refinement(frozen_occ_orbs, active_orbs, orbs_kl, coul_orbs_mn, orbs_aa, coul_orbs_aa);
            core_as_integrals_two_body_baca = core_as_integrals_two_body[0];
            core_as_integrals_two_body_baac = core_as_integrals_two_body[1];
            core_as_integrals_two_body_akcl = core_as_integrals_two_body[2];
            core_as_integrals_two_body_aklc = core_as_integrals_two_body[3];
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Calculate all integrals: " << std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count()
            << " seconds" << std::endl;

    std::cout << "Orbital Multiplication: " << std::chrono::duration_cast<std::chrono::seconds>(t1 - start_time).count() << " seconds" << std::endl;
}


template <std::size_t NDIM>
void Optimization<NDIM>::calculate_energies() {
    auto start_time = std::chrono::high_resolution_clock::now();

    // AS one Particle Part
    double as_one_electron_energy = refinement_utils::contract<2>(
        {as_dim, as_dim}, 
        [&](const auto& i) {return as_one_rdm(i[0], i[1]);},
        [&](const auto& i) {return as_integrals_one_body(i[0], i[1]);}
    );
    

    // AS two Particle Part
    double as_two_electron_energy = 0.5 * refinement_utils::contract<4>(
        {as_dim, as_dim, as_dim, as_dim},
        [&](const auto& i) {return as_two_rdm(i[0], i[1], i[2], i[3]);},
        [&](const auto& i) {return as_integrals_two_body(i[0], i[1], i[2], i[3]);}
    );

    // AS-Core interaction
    double as_core_energy = 0.0;
    if (core_dim > 0)
    {
        as_core_energy += 2*refinement_utils::contract<3>(
            {core_dim, as_dim, as_dim},
            [&](const auto& i) {return as_one_rdm(i[1], i[2]);},
            [&](const auto& i) {return core_as_integrals_two_body_akal(i[0], i[1], i[2]);}
        );

        as_core_energy -= refinement_utils::contract<3>(
            {core_dim, as_dim, as_dim},
            [&](const auto& i) {return as_one_rdm(i[1], i[2]);},
            [&](const auto& i) {return core_as_integrals_two_body_akla(i[0], i[1], i[2]);}
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
void Optimization<NDIM>::calculate_lagrange_multiplier() {
    auto start_time = std::chrono::high_resolution_clock::now();

    LagrangeMultiplier_AS_AS = madness::Tensor<double>(as_dim, as_dim);
    for (int z = 0; z < as_dim; z++) {
        for (int i = 0; i < as_dim; i++) {
            LagrangeMultiplier_AS_AS(z, i) = calculate_lagrange_multiplier_element_as_as(z, i);
        }
    }

    if (core_dim > 0) 
    {
        LagrangeMultiplier_AS_Core = madness::Tensor<double>(core_dim, as_dim);
        for (int z = 0; z < core_dim; z++) {
            for (int i = 0; i < as_dim; i++) {
                LagrangeMultiplier_AS_Core(z, i) = calculate_lagrange_multiplier_element_as_core(z, i);
            }
        }

        //Lagrange Multiplier for Core Refinement
        if(refine_core)
        {
            LagrangeMultiplier_Core_Core = madness::Tensor<double>(core_dim, core_dim);
            for (int z = 0; z < core_dim; z++) {
                for (int c = 0; c < core_dim; c++) {
                    LagrangeMultiplier_Core_Core(z, c) = calculate_lagrange_multiplier_element_core_core(z, c);
                }
            }

            LagrangeMultiplier_Core_AS = madness::Tensor<double>(as_dim, core_dim);
            for (int z = 0; z < as_dim; z++) {
                for (int c = 0; c < core_dim; c++) {
                    LagrangeMultiplier_Core_AS(z, c) = calculate_lagrange_multiplier_element_core_as(z, c);
                }
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "CalculateLagrangeMultiplier took " << duration.count() << " seconds" << std::endl;
}

template <std::size_t NDIM>
double Optimization<NDIM>::calculate_lagrange_multiplier_element_as_as(int z, int i) {
    // lagrange multiplier for the case that z and i are active orbital indizes
    double element = as_one_rdm(i, i) * as_integrals_one_body(z, i);

    element += refinement_utils::contract<3>(
        // I[0]=k, I[1]=l, I[2]=n
        {as_dim, as_dim, as_dim},
        [&](const auto& I) {return as_two_rdm(I[0], I[1], i, I[2]);},
        [&](const auto& I) {return as_integrals_two_body(z, I[1], I[0], I[2]);}
    );

    if(core_dim > 0)
    {
        element += 2 * refinement_utils::contract<2>(
            // I[0]=k, I[1]=a
            {as_dim, core_dim},
            [&](const auto& I) {return as_one_rdm(I[0], i);},
            [&](const auto& I) {return core_as_integrals_two_body_akal(I[1], I[0], z);} // (a,k,l=z) -> <ak|az> = <za|ka>
        );
        element -= refinement_utils::contract<2>(
            // I[0]=k, I[1]=a
            {as_dim, core_dim},
            [&](const auto& I) {return as_one_rdm(I[0], i);},
            [&](const auto& I) {return core_as_integrals_two_body_akla(I[1], I[0], z);} // (a,k,l=z) -> <ak|za> = <zk|aa>
        );
    }
    return element;
}

template <std::size_t NDIM>
double Optimization<NDIM>::calculate_lagrange_multiplier_element_as_core(int z, int i) {
    // lagrange multiplier for the case that z is a core and i an active orbital index
    double element = as_one_rdm(i, i) * core_as_integrals_one_body_ak(z, i);

    //akln part
    element += refinement_utils::contract<3>(
        {as_dim, as_dim, as_dim},
        // I[0]=k, I[1]=l, I[2]=n
        [&](const auto& I) {return as_two_rdm(I[0], I[1], i, I[2]);},
        [&](const auto& I) {return core_as_integrals_two_body_akln(z, I[1], I[0], I[2]);} // (a=z,k=l,l=k,n) -> <zl|kn>
    );

    //zaka part
    element += 2 * refinement_utils::contract<2>(
        {as_dim, core_dim},
        // I[0]=k, I[1]=a
        [&](const auto& I) {return as_one_rdm(I[0], i);},
        [&](const auto& I) {return core_as_integrals_two_body_abak(I[1], z, I[0]);} // (a,b=z,k) -> <az|ak> = <za|ka>
    );
    
    //zkaa part
    element -= refinement_utils::contract<2>(
        {as_dim, core_dim},
        // I[0]=k, I[1]=a
        [&](const auto& I) { return as_one_rdm(I[0], i);},
        [&](const auto& I) { return core_as_integrals_two_body_baak(I[1], z, I[0]);} // (a,b=z,k) -> <za|ak> = <zk|aa>
    );

    return element;
}

template <std::size_t NDIM>
double Optimization<NDIM>::calculate_lagrange_multiplier_element_core_core(int z, int c) {
    // lagrange multiplier for the case that z and c are both core orbital indizes, needed for core orbital refinement
    double element = 2 * core_core_integrals_one_body_ab(z, c);
    
    // 4<za|ca> - 2<za|ac>
    for (int a = 0; a < core_dim; a++) {
        element += 4 * core_as_integrals_two_body_baca(a, z, c); // (a,b=z,c) -> <za|ca>
    }
    for (int a = 0; a < core_dim; a++) {
        element -= 2 * core_as_integrals_two_body_baac(a, z, c); // (a,b=z,c) -> <za|ac>
    }

    //eta_kl(2<zk|cl> - <zk|lc>)
    element += 2 * refinement_utils::contract<2>(
        {as_dim, as_dim},
        // I[0]=k, I[1]=l
        [&](const auto& I) {return as_one_rdm(I[0], I[1]);},
        [&](const auto& I) {return core_as_integrals_two_body_akcl(z, I[0], c, I[1]);} // (a=z,k,c,l) -> <zk|cl>
    );

    element -= refinement_utils::contract<2>(
        {as_dim, as_dim},
        // I[0]=k, I[1]=l
        [&](const auto& I) {return as_one_rdm(I[0], I[1]);},
        [&](const auto& I) {return core_as_integrals_two_body_aklc(z, I[0], I[1], c);} // (a=z,k,l,c) -> <zk|lc>
    );

    return element;
}

template <std::size_t NDIM>
double Optimization<NDIM>::calculate_lagrange_multiplier_element_core_as(int z, int c) {
    // lagrange multiplier for the case that z is active and c is a core orbital index, needed for core orbital refinement
    double element = 2 * core_as_integrals_one_body_ak(c, z);

    // 4<za|ca> - 2<za|ac>
    for (int a = 0; a < core_dim; a++) {
        element += 4 * core_as_integrals_two_body_abak(a, c, z); // (a,b=c,k=z) -> <ac|az> = <za|ca>
    }
    for (int a = 0; a < core_dim; a++) {
        element -= 2 * core_as_integrals_two_body_baak(a, c, z); // (a,b=c,k=z) -> <ca|az> = <za|ac>
    }

    //eta_kl(2<zk|cl> - <zk|lc>)
    element += 2 * refinement_utils::contract<2>(
        {as_dim, as_dim},
        // I[0]=k, I[1]=l
        [&](const auto& I) {return as_one_rdm(I[0], I[1]);},
        [&](const auto& I) {return core_as_integrals_two_body_akln(c, I[0], z, I[1]);} // (a=c,k,l=z,n=l) -> <ck|zl> = <zk|cl>
    );
    
    element -= refinement_utils::contract<2>(
        { as_dim, as_dim},
        // I[0]=k, I[1]=l
        [&](const auto& I) {return as_one_rdm(I[0], I[1]);},
        [&](const auto& I) {return core_as_integrals_two_body_akln(c, z, I[0], I[1]);} // (a=c,k=z,l=k,n=l) -> <cz|kl> = <zk|lc>
    );
    
    return element;
}

template <std::size_t NDIM>
bool Optimization<NDIM>::optimize_orbitals(double optimization_thresh, double NO_occupation_thresh, int maxiter, bool refine_c,
                                            bool use_nonlinear_solver) {

    refine_core = refine_c;
    if(refine_core && core_dim==0)
    {
        std::cout << "Warning: Core orbitals shall be refined, but none were given." << std::endl;
        refine_core = false;
    }

    // Calculate initial energy
    calculate_all_integrals();
    if(core_dim > 0){
        core_total_energy = Integrator.compute_core_energy(frozen_occ_orbs, orbs_aa, coul_orbs_aa, Vnuc, 0);}
    else { core_total_energy = 0;}
    calculate_energies();

    std::vector<int> as_orbital_indices_for_update;
    as_orbital_indices_for_update.reserve(active_orbs.size());
    for (int idx = 0; idx < active_orbs.size(); idx++) {
        if (std::abs(as_one_rdm(idx, idx)) >= NO_occupation_thresh) {
            as_orbital_indices_for_update.push_back(idx);
        } else {
            std::cout << "Skip refinement of active space orbital " << idx
                      << ", since the occupation is less than NO_occupation_thresh (" << NO_occupation_thresh << ")"
                      << std::endl;
        }
    }

    auto active_block_solver = nonlinear_vector_solver<double, NDIM>(
        *(madness_process.world), as_orbital_indices_for_update.size());

    bool converged = false;
    int iterstep = 0;
    while (!converged && iterstep < maxiter) {
        iterstep++;
        std::cout << "---------------------------------------------------" << std::endl;
        std::cout << "Start iteration step: " << iterstep << std::endl;

        // Update LagrangeMultiplier
        calculate_lagrange_multiplier();

        //************************************
        // AS Orbital Refinement
        //************************************
        auto start_orb_update_time = std::chrono::high_resolution_clock::now();
        highest_as_error = 0;

        std::vector<Function<double, NDIM>> AllActiveOrbitalUpdates =
            get_all_active_orbital_updates(as_orbital_indices_for_update);

        auto end_orb_update_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_orb_update_time - start_orb_update_time);
        std::cout << "Get as orbital updates took " << duration.count() << " seconds" << std::endl;

        //************************************
        // Core Orbital Refinement
        //************************************
        if (refine_core) {
            auto start_core_orb_update_time = std::chrono::high_resolution_clock::now();
            highest_core_error = 0;

            std::vector<Function<double, NDIM>> AllCoreOrbitalUpdates = get_all_core_orbital_updates();
            for (int c = 0; c < core_dim; c++) {
                frozen_occ_orbs[c] = frozen_occ_orbs[c] - AllCoreOrbitalUpdates[c];
            }
            auto end_core_orb_update_time = std::chrono::high_resolution_clock::now();
            auto core_duration = std::chrono::duration_cast<std::chrono::seconds>(end_core_orb_update_time - start_core_orb_update_time);
            std::cout << "Get core orbital updates took " << core_duration.count() << " seconds" << std::endl;
        }

        // update AS orbitals
        if (use_nonlinear_solver && !as_orbital_indices_for_update.empty()) {
            std::vector<Function<double, NDIM>> current_active_orbital_block;
            current_active_orbital_block.reserve(as_orbital_indices_for_update.size());
            for (const int act_idx : as_orbital_indices_for_update) {
                current_active_orbital_block.push_back(active_orbs[act_idx]);
            }

            auto updated_active_orbital_block =
                active_block_solver.update(current_active_orbital_block, AllActiveOrbitalUpdates);
            for (std::size_t idx = 0; idx < as_orbital_indices_for_update.size(); idx++) {
                active_orbs[as_orbital_indices_for_update[idx]] = updated_active_orbital_block[idx];
            }
        } else {
            for (std::size_t idx = 0; idx < as_orbital_indices_for_update.size(); idx++) {
                const int act_idx = as_orbital_indices_for_update[idx];
                active_orbs[act_idx] = active_orbs[act_idx] - AllActiveOrbitalUpdates[idx];
            }
        }

        // Orthonormalize orbitals
        if(refine_core)
        {
            // Orthonormalize core orbitals
            if (orthonormalization_method == "cd" or orthonormalization_method == "cholesky") {frozen_occ_orbs = orthonormalize_cd(frozen_occ_orbs);}
            else {frozen_occ_orbs = orthonormalize_symmetric(frozen_occ_orbs);}
            frozen_occ_orbs = truncate(frozen_occ_orbs, num_params.truncation_tol);
        }
        
        // Project out frozen orbitals
        if (core_dim > 0) {
            auto Q = madness::QProjector<double, NDIM>(frozen_occ_orbs);
            active_orbs = Q(active_orbs);
        }
        // Orthonormalize active orbitals
        if (orthonormalization_method == "symmetric") {
            std::cout << "\n=== Symmetric Orthonormalization ===" << std::endl;
            active_orbs = orthonormalize_symmetric(active_orbs);
            std::cout << "=== Symmetric Orthonormalization Complete ===" << std::endl;
        } else if (orthonormalization_method == "mixed") {
            std::vector<double> occupations;
            for (int i = 0; i < as_dim; i++) {
                occupations.push_back(as_one_rdm(i, i));
            }
            active_orbs = Integrator.orthonormalize_mixed_by_degeneracy(active_orbs, occupations, degeneracy_tolerance);
        } else if (orthonormalization_method == "cd" or orthonormalization_method == "cholesky") {
            std::cout << "\n=== Cholesky Orthonormalization ===" << std::endl;
            active_orbs = orthonormalize_cd(active_orbs);
            std::cout << "=== Cholesky Orthonormalization Complete ===" << std::endl;
        } else {
            std::string error_msg = "Unknown orthonormalization method: " + orthonormalization_method + ". Available are symmetric, cd and mixed.";
            MADNESS_EXCEPTION(error_msg.c_str(), 1);
        }

        active_orbs = truncate(active_orbs, num_params.truncation_tol);

        // Check convergence
        if(refine_core) {std::cout << "Highest core orbital error: " << highest_core_error << std::endl;}
        std::cout << "Highest as orbital error: " << highest_as_error << std::endl;

        if(refine_core) {
            double highest_error = std::max(highest_core_error, highest_as_error);
            std::cout << "Highest total error: " << highest_error << std::endl;
            if (highest_error < optimization_thresh) {converged = true;}
        }
        else {if (highest_as_error < optimization_thresh) {converged = true;}}

        std::cout << "Update Integrals" << std::endl;
        // Update integrals for new orbitals
        calculate_all_integrals(refine_core); 
        if(core_dim>0 && refine_core) {core_total_energy = Integrator.compute_core_energy(frozen_occ_orbs, orbs_aa, coul_orbs_aa, Vnuc, 0);}
        // Calculate new energy
        calculate_energies();
    }
    return converged;
}

template <std::size_t NDIM>
std::vector<Function<double, NDIM>> Optimization<NDIM>::get_all_active_orbital_updates(std::vector<int> orbital_indices_for_update) {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Function<double, NDIM>> AllOrbitalUpdates;

    auto t1 = std::chrono::high_resolution_clock::now();

    // Calculate rdm_ii_inv values
    std::vector<double> rdm_ii_inv;
    for (int idx = 0; idx < orbital_indices_for_update.size(); idx++) {
        int i = orbital_indices_for_update[idx];
        rdm_ii_inv.push_back(1 / as_one_rdm(i, i));
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    // 1e Part
    for (int idx = 0; idx < orbital_indices_for_update.size(); idx++) {
        int i = orbital_indices_for_update[idx];

        Function<double, NDIM> rhs;
        rhs = Vnuc * active_orbs[i];
        for (int k = 0; k < core_dim; k++) {
            rhs -= rdm_ii_inv[idx] * LagrangeMultiplier_AS_Core(k, i) * frozen_occ_orbs[k];
        }
        for (int k = 0; k < as_dim; k++) {
            if (k != i) {
                rhs -= rdm_ii_inv[idx] * LagrangeMultiplier_AS_AS(k, i) * active_orbs[k];
            }
        }
        AllOrbitalUpdates.push_back(rhs);
    }
    AllOrbitalUpdates = truncate(AllOrbitalUpdates, num_params.truncation_tol);
    auto t3 = std::chrono::high_resolution_clock::now();

    // AS Part
    for (int k = 0; k < as_dim; k++) {
        std::vector<Function<double, NDIM>> lnk = coul_orbs_mn * active_orbs[k];
        lnk = truncate(lnk, num_params.truncation_tol);
        for (int idx = 0; idx < orbital_indices_for_update.size(); idx++) {
            int i = orbital_indices_for_update[idx];
            std::vector<Function<double, NDIM>> lnk_copy = copy(*(madness_process.world), lnk, false);
            for (int l = 0; l < as_dim; l++) {
                for (int n = 0; n < as_dim; n++) {
                    lnk_copy[l * as_dim + n] *= as_two_rdm(k, l, i, n) * rdm_ii_inv[idx];
                }
            }
            AllOrbitalUpdates[idx] += sum(*(madness_process.world), lnk_copy);
        }
    }
    auto t4 = std::chrono::high_resolution_clock::now();

    // Core - AS interaction
    auto coul_op_parallel =
        std::shared_ptr<SeparatedConvolution<double, NDIM>>(CoulombOperatorNDPtr<NDIM>(*(madness_process.world), num_params.coulomb_lo, num_params.coulomb_eps));
    
    if(core_dim>0)
    {
        // Part 1
        for (int k = 0; k < as_dim; k++) {
            std::vector<Function<double, NDIM>> aak = coul_orbs_aa * active_orbs[k];
            aak = truncate(aak, num_params.truncation_tol);
            for (int idx = 0; idx < orbital_indices_for_update.size(); idx++) {
                int i = orbital_indices_for_update[idx];
                std::vector<Function<double, NDIM>> aak_copy = copy(*(madness_process.world), aak, false);
                for (int a = 0; a < core_dim; a++) {
                    aak_copy[a] *= as_one_rdm(k, i) * rdm_ii_inv[idx];
                }
                AllOrbitalUpdates[idx] += 2 * sum(*(madness_process.world), aak_copy);
            }
        }
        
        //Part 2
        for (int a = 0; a < core_dim; a++) {
            std::vector<Function<double, NDIM>> orbs_ak = frozen_occ_orbs[a] * active_orbs;
            orbs_ak = truncate(orbs_ak, num_params.truncation_tol);
            std::vector<Function<double, NDIM>> coul_orbs_ak = apply(*(madness_process.world), *coul_op_parallel, orbs_ak);
            coul_orbs_ak = truncate(coul_orbs_ak, num_params.truncation_tol);

            std::vector<Function<double, NDIM>> aka = coul_orbs_ak * frozen_occ_orbs[a];
            aka = truncate(aka, num_params.truncation_tol);

            for (int idx = 0; idx < orbital_indices_for_update.size(); idx++) {
                int i = orbital_indices_for_update[idx];
                std::vector<Function<double, NDIM>> aka_copy = copy(*(madness_process.world), aka, false);
                for (int k = 0; k < as_dim; k++) {
                    aka_copy[k] *= as_one_rdm(k, i) * rdm_ii_inv[idx];
                }
                AllOrbitalUpdates[idx] -= sum(*(madness_process.world), aka_copy);
            }
        }
    }
    auto t5 = std::chrono::high_resolution_clock::now();

    // BSH part
    for (int idx = 0; idx < orbital_indices_for_update.size(); idx++) {
        int i = orbital_indices_for_update[idx];
        double en = LagrangeMultiplier_AS_AS(i, i) * rdm_ii_inv[idx];
        if (en > 0) {
            std::cout << "Warning: Positive Lagrange multiplier for active orbital " << i << ": " << en << std::endl;
            en = -1e-3; // Set to small negative value to avoid issues
            std::cout << "Setting it to " << en << "." << std::endl;
        }
        SeparatedConvolution<double, NDIM> bsh_op =
            BSHOperator<NDIM>(*(madness_process.world), sqrt(-2 * en), num_params.BSH_lo, num_params.BSH_eps);

        Function<double, NDIM> r = active_orbs[i] + 2.0 * bsh_op(AllOrbitalUpdates[idx]); // the residual
        double err = r.norm2();
        if (err > highest_as_error) {
            highest_as_error = err;
        }
        AllOrbitalUpdates[idx] = r;
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
std::vector<Function<double, NDIM>> Optimization<NDIM>::get_all_core_orbital_updates() {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Function<double, NDIM>> AllOrbitalUpdates;

    auto coul_op_parallel =
        std::shared_ptr<SeparatedConvolution<double, NDIM>>(CoulombOperatorNDPtr<NDIM>(*(madness_process.world), num_params.coulomb_lo, num_params.coulomb_eps));

    auto t1 = std::chrono::high_resolution_clock::now();

    // 1e Part
    for (int c = 0; c < core_dim; c++) {
        Function<double, NDIM> rhs;
        rhs = Vnuc * frozen_occ_orbs[c];

        for (int x = 0; x < core_dim; x++) {
            if (x != c) {
                rhs -= 0.5 * LagrangeMultiplier_Core_Core(x, c) * frozen_occ_orbs[x];
            }
        }
        for (int x = 0; x < as_dim; x++) {
            rhs -= 0.5 * LagrangeMultiplier_Core_AS(x, c) * active_orbs[x];
        }

        AllOrbitalUpdates.push_back(rhs);
    }
    AllOrbitalUpdates = truncate(AllOrbitalUpdates, num_params.truncation_tol);
    
    auto t2 = std::chrono::high_resolution_clock::now();
    // \sum_a 2g_a^a \phi_c - \sum_a g_a^c \phi_a
    for (int c = 0; c < core_dim; c++) {
        //\sum_a 2g_a^a \phi_c
        std::vector<Function<double, NDIM>> aac = coul_orbs_aa * frozen_occ_orbs[c];
        aac = truncate(aac, num_params.truncation_tol);
        AllOrbitalUpdates[c] += 2 * sum(*(madness_process.world), aac);

        // - \sum_a g_a^c \phi_a
        std::vector<Function<double, NDIM>> ac = frozen_occ_orbs * frozen_occ_orbs[c];
        std::vector<Function<double, NDIM>> coul_ac = apply(*(madness_process.world), *coul_op_parallel, ac);
        coul_ac = truncate(coul_ac, num_params.truncation_tol);
        //This loop could be executed in parallel to get aca
        for (int a = 0; a < core_dim; a++) {
            coul_ac[a] = coul_ac[a] * frozen_occ_orbs[a];
        }
        std::vector<Function<double, NDIM>> aca = truncate(coul_ac, num_params.truncation_tol);
        AllOrbitalUpdates[c] -= sum(*(madness_process.world), aca);
    }
    
    
    auto t3 = std::chrono::high_resolution_clock::now();
    // 1/2 * \sum_kl \eta_k^l 2g_k^l \phi_c - 1/2 * \sum_kl \eta_k^l g_k^c \phi_l
    for (int c = 0; c < core_dim; c++) {
        // 1/2 * \sum_kl \eta_k^l 2g_k^l \phi_c
        std::vector<Function<double, NDIM>> klc = coul_orbs_mn * frozen_occ_orbs[c];
        klc = truncate(klc, num_params.truncation_tol);
        for (int k = 0; k < as_dim; k++) {
            for (int l = 0; l < as_dim; l++) {
                klc[k * as_dim + l] *= as_one_rdm(k, l);
            }
        }
        AllOrbitalUpdates[c] += sum(*(madness_process.world), klc);
        

        // - 1/2 * \sum_kl \eta_k^l g_k^c \phi_l
        std::vector<Function<double, NDIM>> kc = active_orbs * frozen_occ_orbs[c];
        std::vector<Function<double, NDIM>> coul_kc = apply(*(madness_process.world), *coul_op_parallel, kc);
        coul_kc = truncate(coul_kc, num_params.truncation_tol);
        std::vector<Function<double, NDIM>> kcl;
        for (int k = 0; k < as_dim; k++) {
            for (int l = 0; l < as_dim; l++) {
                kcl.push_back(as_one_rdm(k, l) * coul_kc[k] * active_orbs[l]);
            }
        }
        kcl = truncate(kcl, num_params.truncation_tol);
        AllOrbitalUpdates[c] -= 0.5 * sum(*(madness_process.world), kcl);
    }
    
    auto t4 = std::chrono::high_resolution_clock::now();
    // BSH part
    for (int c = 0; c < core_dim; c++) {
        double en = LagrangeMultiplier_Core_Core(c, c);
        if (en > 0) {
            std::cout << "Warning: Positive Lagrange multiplier for core orbital " << c << ": " << en << std::endl;
            en = -1e-3; // Set to small negative value to avoid issues
            std::cout << "Setting it to " << en << "." << std::endl;
        }
        SeparatedConvolution<double, NDIM> bsh_op =
            BSHOperator<NDIM>(*(madness_process.world), sqrt(-en), num_params.BSH_lo, num_params.BSH_eps);
        Function<double, NDIM> r = frozen_occ_orbs[c] + 2.0 * bsh_op(AllOrbitalUpdates[c]); // the residual
        double err = r.norm2();
        
        if (err > highest_core_error) {
            highest_core_error = err;
        }
        AllOrbitalUpdates[c] = r;
    }

    auto t5 = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Core refinement timings:" << std::endl;
    std::cout << "one electron part: " << std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count()
              << " seconds" << std::endl;
    std::cout << "2e part 1: " << std::chrono::duration_cast<std::chrono::seconds>(t3 - t2).count()
              << " seconds" << std::endl;
    std::cout << "2e part 2: " << std::chrono::duration_cast<std::chrono::seconds>(t4 - t3).count()
              << " seconds" << std::endl;   
    std::cout << "BSH part: " << std::chrono::duration_cast<std::chrono::seconds>(t5 - t4).count() 
              << " seconds" << std::endl;   
    std::cout << "Full function: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
              << " seconds" << std::endl;

    return AllOrbitalUpdates;
}

template <std::size_t NDIM>
void Optimization<NDIM>::rotate_orbitals_back() {
    madness::Tensor<double> RotationMatrixBack = madness::Tensor<double>(as_dim, as_dim);
    for (int i = 0; i < as_dim; i++) {
        for (int j = 0; j < as_dim; j++) {
            RotationMatrixBack(i, j) = ActiveSpaceRotationMatrix(j, i);
        }
    }

    refinement_utils::TransformMatrix(&as_one_rdm, RotationMatrixBack);
    refinement_utils::TransformTensor(as_two_rdm, RotationMatrixBack);
    active_orbs = transform(*(madness_process.world), active_orbs, RotationMatrixBack);
    calculate_all_integrals();
}

template <std::size_t NDIM> std::tuple<std::vector<SavedFct<NDIM>>, std::vector<SavedFct<NDIM>>> Optimization<NDIM>::get_orbitals() {
    std::vector<SavedFct<NDIM>> fr_core_orbs;
    std::vector<SavedFct<NDIM>> act_orbs;
    for (int i = 0; i < core_dim; i++) {
        SavedFct<NDIM> orb(frozen_occ_orbs[i]);
        fr_core_orbs.push_back(orb);
    }
    for (int i = 0; i < as_dim; i++) {
        SavedFct<NDIM> orb(active_orbs[i]);
        act_orbs.push_back(orb);
    }
    return std::make_tuple(fr_core_orbs, act_orbs);
}

template <std::size_t NDIM>
nb::tuple Optimization<NDIM>::get_effective_hamiltonian() {
    
    Integrator.update_as_integral_combinations(active_orbs, orbs_kl, coul_orbs_mn);

    // Active space integrals
    Tensor<double> one_e_integrals = Integrator.compute_potential_integrals(active_orbs, Vnuc);
    one_e_integrals += Integrator.compute_kinetic_integrals(active_orbs);

    Tensor<double> two_e_integrals = Integrator.compute_two_body_integrals(active_orbs, orbs_kl, coul_orbs_mn);

    // Core interactions
    double effective_hamiltonian_core_energy = nuclear_repulsion_energy;
    if(core_dim > 0) 
    {
        Integrator.update_core_integral_combinations(frozen_occ_orbs, orbs_aa, coul_orbs_aa);

        // Core energy
        effective_hamiltonian_core_energy = Integrator.compute_core_energy(frozen_occ_orbs, orbs_aa, coul_orbs_aa, Vnuc, nuclear_repulsion_energy);

        // Core-AS interaction
        auto [core_as_integrals_two_body_akal, core_as_integrals_two_body_akla] = Integrator.compute_core_as_2e_integrals_energy(frozen_occ_orbs, active_orbs, orbs_kl, coul_orbs_mn, orbs_aa);
        for (int a = 0; a < core_dim; a++) {
            one_e_integrals += 2*core_as_integrals_two_body_akal(a, _, _) - core_as_integrals_two_body_akla(a, _, _);
        }
    }

    return nb::make_tuple(effective_hamiltonian_core_energy, Numpy2D(one_e_integrals.ptr(),{static_cast<unsigned long>(as_dim), static_cast<unsigned long>(as_dim)}), Numpy4D(two_e_integrals.ptr(), {static_cast<unsigned long>(as_dim), static_cast<unsigned long>(as_dim), static_cast<unsigned long>(as_dim), static_cast<unsigned long>(as_dim)}));
}

template <std::size_t NDIM>
double Optimization<NDIM>::get_c() {
    return core_total_energy + nuclear_repulsion_energy;
}

template <std::size_t NDIM>
void Optimization<NDIM>::set_orthonormalization_method(const std::string& method, double degeneracy_tol) {
    orthonormalization_method = method;
    degeneracy_tolerance = degeneracy_tol;
}

template class Optimization<2>;
template class Optimization<3>;
