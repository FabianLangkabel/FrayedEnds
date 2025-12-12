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
void Optimization_open_shell<NDIM>::sort_eigenpairs_descending(madness::Tensor<double>& eigenvectors,
                                              madness::Tensor<double>& eigenvalues) {
    std::size_t n = eigenvalues.dim(0);

    std::vector<std::pair<double, std::size_t>> pairs;
    for (std::size_t i = 0; i < n; ++i)
        pairs.emplace_back(eigenvalues(i), i);

    std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

    madness::Tensor<double> sorted_eigenvalues(n);
    madness::Tensor<double> sorted_eigenvectors(n, n);

    for (std::size_t i = 0; i < n; ++i) {
        std::size_t orig_idx = pairs[i].second;
        sorted_eigenvalues(i) = eigenvalues(orig_idx);

        for (std::size_t j = 0; j < n; ++j)
            sorted_eigenvectors(j, i) = eigenvectors(j, orig_idx);
    }

    eigenvalues = sorted_eigenvalues;
    eigenvectors = sorted_eigenvectors;
}

template <std::size_t NDIM>
madness::Tensor<double> Optimization_open_shell<NDIM>::matmul_mxm(const madness::Tensor<double>& A, const madness::Tensor<double>& B,
                                                 std::size_t n) {
    madness::Tensor<double> C(n, n);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }
    return C;
}

// this function just takes two lists of doubles, without any information about the shape, might be better to change
// that
template <std::size_t NDIM>
void Optimization_open_shell<NDIM>::give_rdm_and_rotate_orbitals(std::vector<std::vector<double>> one_rdm_elements, std::vector<std::vector<double>> two_rdm_elements) {
    auto start_time = std::chrono::high_resolution_clock::now();

    //****************************************
    // Read active space RDMs
    //****************************************
    as_one_rdm[0] = madness::Tensor<double>(as_dims[0], as_dims[0]);
    as_one_rdm[1] = madness::Tensor<double>(as_dims[1], as_dims[1]);
    as_two_rdm[0] = madness::Tensor<double>(as_dims[0], as_dims[0], as_dims[0], as_dims[0]);
    as_two_rdm[1] = madness::Tensor<double>(as_dims[1], as_dims[1], as_dims[1], as_dims[1]);
    as_two_rdm[2] = madness::Tensor<double>(as_dims[0], as_dims[1], as_dims[0], as_dims[1]); //alpha1 beta1 | alpha2 beta2

    //one_rdm_aa
    int x = 0;
    for (int i = 0; i < as_dims[0]; i++) {
        for (int j = 0; j < as_dims[0]; j++) {
            as_one_rdm[0](i, j) = one_rdm_elements[0][x];
            x++;
        }
    }

    //one_rdm_bb
    x = 0;
    for (int i = 0; i < as_dims[1]; i++) {
        for (int j = 0; j < as_dims[1]; j++) {
            as_one_rdm[1](i, j) = one_rdm_elements[1][x];
            x++;
        }
    }

    // external: aa, ab, bb; internal: aa, bb, ab

    //two_rdm_aa
    x = 0;
    for (int i = 0; i < as_dims[0]; i++) {
        for (int j = 0; j < as_dims[0]; j++) {
            for (int k = 0; k < as_dims[0]; k++) {
                for (int l = 0; l < as_dims[0]; l++) {
                    as_two_rdm[0](i, j, k, l) = two_rdm_elements[0][x];
                    x++;
                }
            }
        }
    }

    //two_rdm_bb    
    x = 0;
    for (int i = 0; i < as_dims[1]; i++) {
        for (int j = 0; j < as_dims[1]; j++) {
            for (int k = 0; k < as_dims[1]; k++) {
                for (int l = 0; l < as_dims[1]; l++) {
                    as_two_rdm[1](i, j, k, l) = two_rdm_elements[2][x];
                    x++;
                }
            }
        }
    }

    //two_rdm_ab
    x = 0;
    for (int i = 0; i < as_dims[0]; i++) {
        for (int j = 0; j < as_dims[1]; j++) {
            for (int k = 0; k < as_dims[0]; k++) {
                for (int l = 0; l < as_dims[1]; l++) {
                    as_two_rdm[2](i, j, k, l) = two_rdm_elements[1][x];
                    x++;
                }
            }
        }
    }



    //****************************************
    // Rotate active Space Orbitals
    //****************************************

    //alpha-alpha and beta-beta
    for(int spin = 0; spin < 2; spin++)
    {
        madness::Tensor<double> ActiveSpaceRotationMatrix = madness::Tensor<double>(as_dims[spin], as_dims[spin]);
        madness::Tensor<double> evals(as_dims[spin]);
        syev(as_one_rdm[spin], ActiveSpaceRotationMatrix, evals);
        sort_eigenpairs_descending(ActiveSpaceRotationMatrix, evals);
        ActiveSpaceRotationMatrices[spin] = ActiveSpaceRotationMatrix;
        TransformMatrix(&as_one_rdm[spin], ActiveSpaceRotationMatrix);
        TransformTensor(as_two_rdm[spin], ActiveSpaceRotationMatrix);
        active_orbs[spin] = transform(*(madness_process.world), active_orbs[spin], ActiveSpaceRotationMatrix);
    }

    //mixed
    Transform_ab_mixed_Tensor(as_two_rdm[2], ActiveSpaceRotationMatrices[0], ActiveSpaceRotationMatrices[1]);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "GiveRDMFiles took " << duration.count() << " seconds" << std::endl;
}

template <std::size_t NDIM>
void Optimization_open_shell<NDIM>::TransformMatrix(madness::Tensor<double>* ObjectMatrix,
                                   madness::Tensor<double>& TransformationMatrix) {
    int n = TransformationMatrix.dim(0);
    madness::Tensor<double> temp = matmul_mxm(*ObjectMatrix, TransformationMatrix, n);
    *ObjectMatrix = matmul_mxm(transpose(TransformationMatrix), temp, n);
}

template <std::size_t NDIM>
void Optimization_open_shell<NDIM>::TransformTensor(madness::Tensor<double>& ObjectTensor,
                                   madness::Tensor<double>& TransformationMatrix) {
    int n = TransformationMatrix.dim(0);
    madness::Tensor<double> temp1(n, n, n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k2 = 0; k2 < n; k2++) {
                for (int l = 0; l < n; l++) {
                    double k_value = 0;
                    for (int k = 0; k < n; k++) {
                        k_value += TransformationMatrix(k, k2) * ObjectTensor(i, j, k, l);
                    }
                    temp1(i, j, k2, l) = k_value;
                }
            }
        }
    }

    madness::Tensor<double> temp2(n, n, n, n);
    for (int i2 = 0; i2 < n; i2++) {
        for (int j = 0; j < n; j++) {
            for (int k2 = 0; k2 < n; k2++) {
                for (int l = 0; l < n; l++) {
                    double i_value = 0;
                    for (int i = 0; i < n; i++) {
                        i_value += TransformationMatrix(i, i2) * temp1(i, j, k2, l);
                    }
                    temp2(i2, j, k2, l) = i_value;
                }
            }
        }
    }

    madness::Tensor<double> temp3(n, n, n, n);
    for (int i2 = 0; i2 < n; i2++) {
        for (int j = 0; j < n; j++) {
            for (int k2 = 0; k2 < n; k2++) {
                for (int l2 = 0; l2 < n; l2++) {
                    double l_value = 0;
                    for (int l = 0; l < n; l++) {
                        l_value += TransformationMatrix(l, l2) * temp2(i2, j, k2, l);
                    }
                    temp3(i2, j, k2, l2) = l_value;
                }
            }
        }
    }

    madness::Tensor<double> temp4(n, n, n, n);
    for (int i2 = 0; i2 < n; i2++) {
        for (int j2 = 0; j2 < n; j2++) {
            for (int k2 = 0; k2 < n; k2++) {
                for (int l2 = 0; l2 < n; l2++) {
                    double j_value = 0;
                    for (int j = 0; j < n; j++) {
                        j_value += TransformationMatrix(j, j2) * temp3(i2, j, k2, l2);
                    }
                    temp4(i2, j2, k2, l2) = j_value;
                }
            }
        }
    }
    ObjectTensor = temp4;
}

template <std::size_t NDIM>
void Optimization_open_shell<NDIM>::Transform_ab_mixed_Tensor(madness::Tensor<double>& ObjectTensor,
                                   madness::Tensor<double>& TransformationMatrix_alpha, madness::Tensor<double>& TransformationMatrix_beta) {
    int n = TransformationMatrix_alpha.dim(0);
    int m = TransformationMatrix_beta.dim(0);

    madness::Tensor<double> temp1(n, m, n, m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            for (int k2 = 0; k2 < n; k2++) {
                for (int l = 0; l < m; l++) {
                    double k_value = 0;
                    for (int k = 0; k < n; k++) {
                        k_value += TransformationMatrix_alpha(k, k2) * ObjectTensor(i, j, k, l);
                    }
                    temp1(i, j, k2, l) = k_value;
                }
            }
        }
    }

    madness::Tensor<double> temp2(n, m, n, m);
    for (int i2 = 0; i2 < n; i2++) {
        for (int j = 0; j < m; j++) {
            for (int k2 = 0; k2 < n; k2++) {
                for (int l = 0; l < m; l++) {
                    double i_value = 0;
                    for (int i = 0; i < n; i++) {
                        i_value += TransformationMatrix_alpha(i, i2) * temp1(i, j, k2, l);
                    }
                    temp2(i2, j, k2, l) = i_value;
                }
            }
        }
    }

    madness::Tensor<double> temp3(n, m, n, m);
    for (int i2 = 0; i2 < n; i2++) {
        for (int j = 0; j < m; j++) {
            for (int k2 = 0; k2 < n; k2++) {
                for (int l2 = 0; l2 < m; l2++) {
                    double l_value = 0;
                    for (int l = 0; l < m; l++) {
                        l_value += TransformationMatrix_beta(l, l2) * temp2(i2, j, k2, l);
                    }
                    temp3(i2, j, k2, l2) = l_value;
                }
            }
        }
    }

    madness::Tensor<double> temp4(n, m, n, m);
    for (int i2 = 0; i2 < n; i2++) {
        for (int j2 = 0; j2 < m; j2++) {
            for (int k2 = 0; k2 < n; k2++) {
                for (int l2 = 0; l2 < m; l2++) {
                    double j_value = 0;
                    for (int j = 0; j < m; j++) {
                        j_value += TransformationMatrix_beta(j, j2) * temp3(i2, j, k2, l2);
                    }
                    temp4(i2, j2, k2, l2) = j_value;
                }
            }
        }
    }
    ObjectTensor = temp4;
}

template <std::size_t NDIM>
void Optimization_open_shell<NDIM>::calculate_all_integrals() {

    auto start_time = std::chrono::high_resolution_clock::now();

    // Calculate and cache relevant orbital-combinations
    Integrator->update_as_integral_combinations(active_orbs, orbs_kl, coul_orbs_mn);
    Integrator->update_core_integral_combinations(frozen_occ_orbs, orbs_aa);
    auto t1 = std::chrono::high_resolution_clock::now();


    // Calculate one electron Integrals
    as_integrals_one_body = Integrator->compute_potential_integrals(active_orbs, Vnuc);
    {
        std::array<madness::Tensor<double>, 2> kin_Integrals = Integrator->compute_kinetic_integrals(active_orbs);
        as_integrals_one_body[0] += kin_Integrals[0];
        as_integrals_one_body[1] += kin_Integrals[1];
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    // Calculate two electron Integrals
    as_integrals_two_body = Integrator->compute_two_body_integrals(active_orbs, orbs_kl, coul_orbs_mn);
    auto t3 = std::chrono::high_resolution_clock::now();
    
    // Calculate Core-AS interaction integrals
    if((core_dims[0] + core_dims[1]) > 0)
    {
        core_as_integrals_one_body_ak = Integrator->compute_core_as_integrals_one_body(frozen_occ_orbs, active_orbs, Vnuc);
        std::vector<std::vector<madness::Tensor<double>>> core_as_integrals_two_body = Integrator->compute_core_as_integrals_two_body(frozen_occ_orbs, active_orbs, orbs_kl, coul_orbs_mn, orbs_aa, true, true, true, true, true);
        core_as_integrals_two_body_akal = core_as_integrals_two_body[0];
        core_as_integrals_two_body_akla = core_as_integrals_two_body[1];
        core_as_integrals_two_body_akln = core_as_integrals_two_body[2];
        core_as_integrals_two_body_abak = core_as_integrals_two_body[3];
        core_as_integrals_two_body_baak = core_as_integrals_two_body[4];
    }



    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Calculate all integrals: " << std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count()
            << " seconds" << std::endl;

    std::cout << "Orbital Multiplication: " << std::chrono::duration_cast<std::chrono::seconds>(t1 - start_time).count() << " seconds" << std::endl;
    std::cout << "One Body AS: " << std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count() << " seconds" << std::endl;
    std::cout << "Two Body AS: " << std::chrono::duration_cast<std::chrono::seconds>(t3 - t2).count() << " seconds" << std::endl;
    std::cout << "Core-AS: " << std::chrono::duration_cast<std::chrono::seconds>(end_time - t3).count() << " seconds" << std::endl;
}

template <std::size_t NDIM>
void Optimization_open_shell<NDIM>::calculate_energies() {
    auto start_time = std::chrono::high_resolution_clock::now();

    
    // Active Space Part
    double as_one_electron_energy = 0.0;
    // One Particle Part
    for (int spin = 0; spin < 2; spin++) {
        for (int k = 0; k < as_dims[spin]; k++) {
            for (int l = 0; l < as_dims[spin]; l++) {
                as_one_electron_energy += as_one_rdm[spin](k, l) * as_integrals_one_body[spin](k, l);
            }
        }
    }

    // Two Particle Part alpha_alpha and beta_beta
    double as_two_electron_energy = 0.0;
    for (int spin = 0; spin < 2; spin++) {
        for (int k = 0; k < as_dims[spin]; k++) {
            for (int l = 0; l < as_dims[spin]; l++) {
                for (int m = 0; m < as_dims[spin]; m++) {
                    for (int n = 0; n < as_dims[spin]; n++) {
                        as_two_electron_energy += as_two_rdm[spin](k, l, m, n) * 0.5 * as_integrals_two_body[spin](k, l, m, n);
                    }
                }
            }
        }
    }

    // Two Particle Part alpha_beta
    for (int k = 0; k < as_dims[0]; k++) {
        for (int l = 0; l < as_dims[0]; l++) {
            for (int m = 0; m < as_dims[1]; m++) {
                for (int n = 0; n < as_dims[1]; n++) {
                    as_two_electron_energy += 2 * as_two_rdm[2](k, l, m, n) * 0.5 * as_integrals_two_body[2](k, l, m, n);
                }
            }
        }
    }

    // AS-Core interaction
    double as_core_energy = 0.0;
    if((core_dims[0] + core_dims[1]) > 0)
    {
        for (int spin = 0; spin < 2; spin++) {
            for (int a = 0; a < (core_dims[0] + core_dims[1]); a++) {
                for (int k = 0; k < as_dims[spin]; k++) {
                    for (int l = 0; l < as_dims[spin]; l++) {
                        as_core_energy += as_one_rdm[spin](k, l) * core_as_integrals_two_body_akal[spin](a, k, l);
                    }
                }
            }
        }
        for (int spin = 0; spin < 2; spin++) {
            for (int a = 0; a < core_dims[spin]; a++) {
                for (int k = 0; k < as_dims[spin]; k++) {
                    for (int l = 0; l < as_dims[spin]; l++) {
                        as_core_energy -= as_one_rdm[spin](k, l) * core_as_integrals_two_body_akla[spin](a, k, l);
                    }
                }
            }
        }
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
    
    int op_spin; if(spin == 0){op_spin = 1;} else {op_spin = 0;}

    double element = as_one_rdm[spin](i, i) * as_integrals_one_body[spin](z, i);

    // alpha-alpha / beta-beta
    for (int l = 0; l < as_dims[spin]; l++) {
        for (int n = 0; n < as_dims[spin]; n++) {
            for (int k = 0; k < as_dims[spin]; k++) {
                element += as_two_rdm[spin](k, l, i, n) * as_integrals_two_body[spin](z, l, k, n);
            }
        }
    }

    //mixed terms
    for (int l = 0; l < as_dims[op_spin]; l++) {
        for (int n = 0; n < as_dims[op_spin]; n++) {
            for (int k = 0; k < as_dims[spin]; k++) {
                if(spin == 0)
                {
                    element += as_two_rdm[2](k, l, i, n) * as_integrals_two_body[2](z, l, k, n);
                }
                else
                {
                    element += as_two_rdm[2](l, k, n, i) * as_integrals_two_body[2](l, z, n, k);
                }
            }
        }
    }

    // Core - AS interaction
    if((core_dims[0] + core_dims[1]) > 0)
    {

        for (int a = 0; a < (core_dims[0] + core_dims[1]); a++) {
            for (int k = 0; k < as_dims[spin]; k++) {
                element += as_one_rdm[spin](k, i) * core_as_integrals_two_body_akal[spin](a, z, k);
            }
        }

        for (int a = 0; a < core_dims[spin]; a++) {
            for (int k = 0; k < as_dims[spin]; k++) {
                element -= as_one_rdm[spin](k, i) * core_as_integrals_two_body_akla[spin](a, k, z);
            }
        }
    }
    return element;
}

template <std::size_t NDIM>
double Optimization_open_shell<NDIM>::calculate_lagrange_multiplier_element_as_core(int z, int i, int spin) {
    int op_spin; if(spin == 0){op_spin = 1;} else {op_spin = 0;}

    double el_p1 = 0;
    double el_p2 = 0;
    double el_p3 = 0;
    double el_p4 = 0;

    double element = as_one_rdm[spin](i, i) * core_as_integrals_one_body_ak[spin](z, i);
    el_p1 = as_one_rdm[spin](i, i) * core_as_integrals_one_body_ak[spin](z, i);

    //akln part
    // alpha-alpha / beta-beta
    for (int l = 0; l < as_dims[spin]; l++) {
        for (int n = 0; n < as_dims[spin]; n++) {
            for (int k = 0; k < as_dims[spin]; k++) {
                element += as_two_rdm[spin](k, l, i, n) * core_as_integrals_two_body_akln[spin](z, l, k, n);
                el_p2 += as_two_rdm[spin](k, l, i, n) * core_as_integrals_two_body_akln[spin](z, l, k, n);
            }
        }
    }
    //mixed terms
    for (int l = 0; l < as_dims[op_spin]; l++) {
        for (int n = 0; n < as_dims[op_spin]; n++) {
            for (int k = 0; k < as_dims[spin]; k++) {
                if(spin == 0)
                {
                    element += as_two_rdm[2](k, l, i, n) * core_as_integrals_two_body_akln[2](z, l, k, n);
                    el_p2 += as_two_rdm[2](k, l, i, n) * core_as_integrals_two_body_akln[2](z, l, k, n);
                }
                else
                {
                    element += as_two_rdm[2](l, k, n, i) * core_as_integrals_two_body_akln[3](z, l, k, n);
                    el_p2 += as_two_rdm[2](l, k, n, i) * core_as_integrals_two_body_akln[3](z, l, k, n);
                }
            }
        }
    }

    //abak part
    // alpha-alpha / beta-beta
    for (int k = 0; k < as_dims[spin]; k++) {
        for (int a = 0; a < core_dims[spin]; a++) {
            element += as_one_rdm[spin](k, i) * core_as_integrals_two_body_abak[spin](a, z, k);
            el_p3 += as_one_rdm[spin](k, i) * core_as_integrals_two_body_abak[spin](a, z, k);
        }
    }
    //mixed terms
    for (int k = 0; k < as_dims[spin]; k++) {
        for (int a = 0; a < core_dims[op_spin]; a++) {
            if(spin == 0)
            {
                element += as_one_rdm[spin](k, i) * core_as_integrals_two_body_abak[3](a, z, k);
                el_p3 += as_one_rdm[spin](k, i) * core_as_integrals_two_body_abak[3](a, z, k);
            }
            else
            {
                element += as_one_rdm[spin](k, i) * core_as_integrals_two_body_abak[2](a, z, k);
                el_p3 += as_one_rdm[spin](k, i) * core_as_integrals_two_body_abak[2](a, z, k);
            }
        }
    }

    //baak part
    // alpha-alpha / beta-beta
    for (int k = 0; k < as_dims[spin]; k++) {
        for (int a = 0; a < core_dims[spin]; a++) {
            element -= as_one_rdm[spin](k, i) * core_as_integrals_two_body_baak[spin](a, z, k);
            el_p4 -= as_one_rdm[spin](k, i) * core_as_integrals_two_body_baak[spin](a, z, k);
        }
    }


    //std::cout << "z: " << z << ", i: " << i << ", p1: " << el_p1 << ", p2: " << el_p2 << ", p3: " << el_p3 << ", p4: " << el_p4 << std::endl;
    return element;
}

template <std::size_t NDIM>
bool Optimization_open_shell<NDIM>::optimize_orbitals(double optimization_thresh, double NO_occupation_thresh, int maxiter) {

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
            active_orbs[spin] = orthonormalize_symmetric(active_orbs[spin]);
            // orbitals = orthonormalize_cd(orbitals);
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
        TransformMatrix(&as_one_rdm[spin], RotationMatrixBack);
        TransformTensor(as_two_rdm[spin], RotationMatrixBack);
        active_orbs[spin] = transform(*(madness_process.world), active_orbs[spin], RotationMatrixBack);
    }
    Transform_ab_mixed_Tensor(as_two_rdm[2], RotationMatricesBack[0], RotationMatricesBack[1]);
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
    effective_one_e_integrals[0] = as_integrals_one_body[0];
    effective_one_e_integrals[1] = as_integrals_one_body[1];

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


    auto two_e_aa_owner = std::make_shared<madness::Tensor<double>>(std::move(as_integrals_two_body[0]));
    nb::capsule two_e_aa_caps(
        new std::shared_ptr<madness::Tensor<double>>(two_e_aa_owner),
        [](void *p) noexcept {
            delete reinterpret_cast<std::shared_ptr<madness::Tensor<double>>*>(p);
        }
    );

    auto two_e_bb_owner = std::make_shared<madness::Tensor<double>>(std::move(as_integrals_two_body[1]));
    nb::capsule two_e_bb_caps(
        new std::shared_ptr<madness::Tensor<double>>(two_e_bb_owner),
        [](void *p) noexcept {
            delete reinterpret_cast<std::shared_ptr<madness::Tensor<double>>*>(p);
        }
    );

    auto two_e_ab_owner = std::make_shared<madness::Tensor<double>>(std::move(as_integrals_two_body[2]));
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

    Numpy4D alpha_alpha(as_integrals_two_body[0].ptr(), {as_dims[0], as_dims[0], as_dims[0], as_dims[0]}, two_e_aa_caps);
    Numpy4D beta_beta(as_integrals_two_body[1].ptr(), {as_dims[1], as_dims[1], as_dims[1], as_dims[1]}, two_e_bb_caps);
    Numpy4D alpha_beta(as_integrals_two_body[2].ptr(), {as_dims[0], as_dims[1], as_dims[0], as_dims[1]}, two_e_ab_caps);
    //aa, ab, bb order external; aa, bb, ab order internal
    nb::list g2_list;
    g2_list.append(alpha_alpha);
    g2_list.append(alpha_beta);
    g2_list.append(beta_beta);

    return nb::make_tuple(effective_hamiltonian_core_energy, h1_list, g2_list);
}

template class Optimization_open_shell<2>;
template class Optimization_open_shell<3>;