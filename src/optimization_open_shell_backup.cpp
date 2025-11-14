#include "optimization_open_shell.hpp"

using namespace madness;

template <std::size_t NDIM>
Optimization_open_shell<NDIM>::Optimization_open_shell(MadnessProcess<NDIM>& mp) : madness_process(mp) {
    std::cout.precision(6);
}

template <std::size_t NDIM>
Optimization_open_shell<NDIM>::~Optimization_open_shell() {
    Vnuc.clear();
    frozen_occ_orbs[0].clear();
    active_orbs[0].clear();
    frozen_virt_orbs[0].clear();
    frozen_occ_orbs[1].clear();
    active_orbs[1].clear();
    frozen_virt_orbs[1].clear();
    //orbs_kl.clear();
    coul_orbs_mn[0].clear();
    coul_orbs_mn[1].clear();
}

template <std::size_t NDIM>
void Optimization_open_shell<NDIM>::give_potential_and_repulsion(SavedFct<NDIM> potential, double nuclear_repulsion) {
    Vnuc = madness_process.loadfct(potential);
    nuclear_repulsion_energy = nuclear_repulsion;
}


template <std::size_t NDIM>
void Optimization_open_shell<NDIM>::give_initial_orbitals(std::vector<SavedFct<NDIM>> alpha_orbitals, std::vector<SavedFct<NDIM>> beta_orbitals) {
    auto start_time = std::chrono::high_resolution_clock::now();

    for (SavedFct<NDIM> orb : alpha_orbitals) {
        if (orb.type == "frozen_occ") {
            frozen_occ_orbs[0].push_back(madness_process.loadfct(orb));
        } else if (orb.type == "active") {
            active_orbs[0].push_back(madness_process.loadfct(orb));
        } else if (orb.type == "frozen_virt") {
            frozen_virt_orbs[0].push_back(madness_process.loadfct(orb));
        } else {
            std::cerr << "Unknown orbital type: " << orb.type
                      << ". Recognized types are frozen_occ, active and frozen_virt." << std::endl;
        }
    }
    core_dims[0] = frozen_occ_orbs[0].size();
    as_dims[0] = active_orbs[0].size();
    froz_virt_dims[0] = frozen_virt_orbs[0].size();

    for (SavedFct<NDIM> orb : beta_orbitals) {
        if (orb.type == "frozen_occ") {
            frozen_occ_orbs[1].push_back(madness_process.loadfct(orb));
        } else if (orb.type == "active") {
            active_orbs[1].push_back(madness_process.loadfct(orb));
        } else if (orb.type == "frozen_virt") {
            frozen_virt_orbs[1].push_back(madness_process.loadfct(orb));
        } else {
            std::cerr << "Unknown orbital type: " << orb.type
                      << ". Recognized types are frozen_occ, active and frozen_virt." << std::endl;
        }
    }
    core_dims[1] = frozen_occ_orbs[1].size();
    as_dims[1] = active_orbs[1].size();
    froz_virt_dims[1] = frozen_virt_orbs[1].size();

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
    as_two_rdm[1] = madness::Tensor<double>(as_dims[0], as_dims[1], as_dims[0], as_dims[1]); //alpha1 beta1 | alpha2 beta2
    as_two_rdm[2] = madness::Tensor<double>(as_dims[1], as_dims[1], as_dims[1], as_dims[1]);

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

    //two_rdm_ab
    x = 0;
    for (int i = 0; i < as_dims[0]; i++) {
        for (int j = 0; j < as_dims[1]; j++) {
            for (int k = 0; k < as_dims[0]; k++) {
                for (int l = 0; l < as_dims[1]; l++) {
                    as_two_rdm[1](i, j, k, l) = two_rdm_elements[1][x];
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
                    as_two_rdm[2](i, j, k, l) = two_rdm_elements[2][x];
                    x++;
                }
            }
        }
    }


    //****************************************
    // Rotate active Space Orbitals
    //****************************************

    //alpha
    madness::Tensor<double> ActiveSpaceRotationMatrix_alpha = madness::Tensor<double>(as_dims[0], as_dims[0]);
    madness::Tensor<double> evals_alpha(as_dims[0]);
    syev(as_one_rdm[0], ActiveSpaceRotationMatrix_alpha, evals_alpha);
    sort_eigenpairs_descending(ActiveSpaceRotationMatrix_alpha, evals_alpha);
    ActiveSpaceRotationMatrices[0] = ActiveSpaceRotationMatrix_alpha;
    TransformMatrix(&as_one_rdm[0], ActiveSpaceRotationMatrix_alpha);
    TransformTensor(as_two_rdm[0], ActiveSpaceRotationMatrix_alpha);
    active_orbs[0] = transform(*(madness_process.world), active_orbs[0], ActiveSpaceRotationMatrix_alpha);

    //beta
    madness::Tensor<double> ActiveSpaceRotationMatrix_beta = madness::Tensor<double>(as_dims[1], as_dims[1]);
    madness::Tensor<double> evals_beta(as_dims[1]);
    syev(as_one_rdm[1], ActiveSpaceRotationMatrix_beta, evals_beta);
    sort_eigenpairs_descending(ActiveSpaceRotationMatrix_beta, evals_beta);
    ActiveSpaceRotationMatrices[1] = ActiveSpaceRotationMatrix_beta;
    TransformMatrix(&as_one_rdm[1], ActiveSpaceRotationMatrix_beta);
    TransformTensor(as_two_rdm[2], ActiveSpaceRotationMatrix_beta);
    active_orbs[1] = transform(*(madness_process.world), active_orbs[1], ActiveSpaceRotationMatrix_beta);

    //mixed
    Transform_ab_mixed_Tensor(as_two_rdm[1], ActiveSpaceRotationMatrix_alpha, ActiveSpaceRotationMatrix_beta);

    //std::cout << evals << std::endl;

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
    //Old Code der schon übersetzt wurde!!!
    /*
    auto start_time = std::chrono::high_resolution_clock::now();

    // Initializing the Coulomb operator
    auto coul_op_parallel =
        std::shared_ptr<SeparatedConvolution<double, NDIM>>(CoulombOperatorNDPtr<NDIM>(*(madness_process.world), coulomb_lo, coulomb_eps));

    // Multiplication of AS orbital pairs and their Coulomb element (are needed more often and are therefore stored)
    auto t1 = std::chrono::high_resolution_clock::now();
    orbs_kl.clear();
    for (int k = 0; k < as_dim; k++) {
        std::vector<Function<double, NDIM>> kl = active_orbs[k] * active_orbs;
        orbs_kl.insert(std::end(orbs_kl), std::begin(kl), std::end(kl));
    }
    orbs_kl = truncate(orbs_kl, truncation_tol);
    coul_orbs_mn = apply(*(madness_process.world), *coul_op_parallel, orbs_kl);
    coul_orbs_mn = truncate(coul_orbs_mn, truncation_tol);
    auto t2 = std::chrono::high_resolution_clock::now();

    // AS-AS one electron integrals
    as_integrals_one_body = madness::Tensor<double>(as_dim, as_dim);
    for (int k = 0; k < as_dim; k++) {
        for (int l = 0; l < as_dim; l++) {
            // Kinetic
            for (int axis = 0; axis < NDIM; axis++) {
                Derivative<double, NDIM> D = free_space_derivative<double, NDIM>(*(madness_process.world), axis);
                Function<double, NDIM> d_orb_k = D(active_orbs[k]);
                Function<double, NDIM> d_orb_l = D(active_orbs[l]);
                as_integrals_one_body(k, l) += 0.5 * inner(d_orb_k, d_orb_l);
            }
            // Nuclear
            Function<double, NDIM> Vnuc_orb_l;
            Vnuc_orb_l = Vnuc * active_orbs[l];

            as_integrals_one_body(k, l) += inner(active_orbs[k], Vnuc_orb_l);
        }
    }
    auto t3 = std::chrono::high_resolution_clock::now();

    // AS two electron integrals
    as_integrals_two_body = madness::Tensor<double>(as_dim, as_dim, as_dim, as_dim);
    madness::Tensor<double> Inner_prods = matrix_inner(*(madness_process.world), orbs_kl, coul_orbs_mn, false);
    for (int k = 0; k < as_dim; k++) {
        for (int l = 0; l < as_dim; l++) {
            for (int m = 0; m < as_dim; m++) {
                for (int n = 0; n < as_dim; n++) {
                    as_integrals_two_body(k, m, l, n) = Inner_prods(k * as_dim + l, m * as_dim + n);
                }
            }
        }
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    */





    auto start_time = std::chrono::high_resolution_clock::now();

    // Initializing the Coulomb operator
    auto coul_op_parallel =
        std::shared_ptr<SeparatedConvolution<double, NDIM>>(CoulombOperatorNDPtr<NDIM>(*(madness_process.world), coulomb_lo, coulomb_eps));


    // Alpha-Alpha AS one electron integrals
    madness::Tensor<double> as_integrals_one_body_alpha_alpha = madness::Tensor<double>(as_dims[0], as_dims[0]);
    madness::Tensor<double> potential_integrals_alpha_alpha = madness::matrix_inner(*(madness_process.world), active_orbs[0], Vnuc * active_orbs[0]);
    for (int k = 0; k < as_dims[0]; k++) {
        for (int l = 0; l < as_dims[0]; l++) {
            // Kinetic
            for (int axis = 0; axis < NDIM; axis++) {
                Derivative<double, NDIM> D = free_space_derivative<double, NDIM>(*(madness_process.world), axis);
                Function<double, NDIM> d_orb_k = D(active_orbs[0][k]);
                Function<double, NDIM> d_orb_l = D(active_orbs[0][l]);
                as_integrals_one_body_alpha_alpha(k, l) += 0.5 * inner(d_orb_k, d_orb_l);
            }
            as_integrals_one_body_alpha_alpha(k, l) += potential_integrals_alpha_alpha(k, l);
        }
    }

    // Beta-Beta AS one electron integrals
    madness::Tensor<double> as_integrals_one_body_beta_beta = madness::Tensor<double>(as_dims[1], as_dims[1]);
    madness::Tensor<double> potential_integrals_beta_beta = madness::matrix_inner(*(madness_process.world), active_orbs[1], Vnuc * active_orbs[1]);
    for (int k = 0; k < as_dims[1]; k++) {
        for (int l = 0; l < as_dims[1]; l++) {
            // Kinetic
            for (int axis = 0; axis < NDIM; axis++) {
                Derivative<double, NDIM> D = free_space_derivative<double, NDIM>(*(madness_process.world), axis);
                Function<double, NDIM> d_orb_k = D(active_orbs[1][k]);
                Function<double, NDIM> d_orb_l = D(active_orbs[1][l]);
                as_integrals_one_body_beta_beta(k, l) += 0.5 * inner(d_orb_k, d_orb_l);
            }
            as_integrals_one_body_beta_beta(k, l) += potential_integrals_beta_beta(k, l);
        }
    }
    as_integrals_one_body[0] = as_integrals_one_body_alpha_alpha;
    as_integrals_one_body[1] = as_integrals_one_body_beta_beta;

    //alpha alpha interaction
    madness::Tensor<double> two_body_integrals_alpha_alpha = madness::Tensor<double>(as_dims[0], as_dims[0], as_dims[0], as_dims[0]);
    {
        std::vector<Function<double, NDIM>> orbs_kl_alpha;
        for (int k = 0; k < as_dims[0]; k++) {
            std::vector<Function<double, NDIM>> kl = active_orbs[0][k] * active_orbs[0];
            orbs_kl_alpha.insert(std::end(orbs_kl_alpha), std::begin(kl), std::end(kl));
        }
        orbs_kl_alpha = truncate(orbs_kl_alpha, truncation_tol);

        std::vector<Function<double, NDIM>> coul_orbs_mn_alpha = apply(*(madness_process.world), *coul_op_parallel, orbs_kl_alpha);
        coul_orbs_mn_alpha = truncate(coul_orbs_mn_alpha, truncation_tol);
        coul_orbs_mn[0] = coul_orbs_mn_alpha;

        madness::Tensor<double> Inner_prods = matrix_inner(*(madness_process.world), orbs_kl_alpha, coul_orbs_mn_alpha, false);
        std::vector<double> flat;
        for (int k = 0; k < as_dims[0]; k++) {
            for (int l = 0; l < as_dims[0]; l++) {
                for (int m = 0; m < as_dims[0]; m++) {
                    for (int n = 0; n < as_dims[0]; n++) {
                        auto tmp = Inner_prods(k * as_dims[0] + l, m * as_dims[0] + n);
                        two_body_integrals_alpha_alpha(k, m, l, n) = tmp;
                        flat.push_back(tmp);
                    }
                }
            }
        }
    }

    //beta beta interaction
    madness::Tensor<double> two_body_integrals_beta_beta = madness::Tensor<double>(as_dims[1], as_dims[1], as_dims[1], as_dims[1]);
    {
        std::vector<Function<double, NDIM>> orbs_kl_beta;
        for (int k = 0; k < as_dims[1]; k++) {
            std::vector<Function<double, NDIM>> kl = active_orbs[1][k] * active_orbs[1];
            orbs_kl_beta.insert(std::end(orbs_kl_beta), std::begin(kl), std::end(kl));
        }
        orbs_kl_beta = truncate(orbs_kl_beta, truncation_tol);

        std::vector<Function<double, NDIM>> coul_orbs_mn_beta = apply(*(madness_process.world), *coul_op_parallel, orbs_kl_beta);
        coul_orbs_mn_beta = truncate(coul_orbs_mn_beta, truncation_tol);
        coul_orbs_mn[1] = coul_orbs_mn_beta;

        madness::Tensor<double> Inner_prods = matrix_inner(*(madness_process.world), orbs_kl_beta, coul_orbs_mn_beta, false);
        std::vector<double> flat;
        for (int k = 0; k < as_dims[1]; k++) {
            for (int l = 0; l < as_dims[1]; l++) {
                for (int m = 0; m < as_dims[1]; m++) {
                    for (int n = 0; n < as_dims[1]; n++) {
                        auto tmp = Inner_prods(k * as_dims[1] + l, m * as_dims[1] + n);
                        two_body_integrals_beta_beta(k, m, l, n) = tmp;
                        flat.push_back(tmp);
                    }
                }
            }
        }
    }
    
    madness::Tensor<double> two_body_integrals_alpha_beta = madness::Tensor<double>(as_dims[0], as_dims[1], as_dims[0], as_dims[1]);
    {
        std::vector<Function<double, NDIM>> orbs_kl_alpha;
        for (int k = 0; k < as_dims[0]; k++) {
            std::vector<Function<double, NDIM>> kl = active_orbs[0][k] * active_orbs[0];
            orbs_kl_alpha.insert(std::end(orbs_kl_alpha), std::begin(kl), std::end(kl));
        }
        orbs_kl_alpha = truncate(orbs_kl_alpha, truncation_tol);

        std::vector<Function<double, NDIM>> orbs_mn_beta;
        for (int m = 0; m < as_dims[1]; m++) {
            std::vector<Function<double, NDIM>> mn = active_orbs[1][m] * active_orbs[1];
            orbs_mn_beta.insert(std::end(orbs_mn_beta), std::begin(mn), std::end(mn));
        }
        orbs_mn_beta = truncate(orbs_mn_beta, truncation_tol);


        std::vector<Function<double, NDIM>> coul_orbs_mn_beta = apply(*(madness_process.world), *coul_op_parallel, orbs_mn_beta);
        coul_orbs_mn_beta = truncate(coul_orbs_mn_beta, truncation_tol);

        madness::Tensor<double> Inner_prods = matrix_inner(*(madness_process.world), orbs_kl_alpha, coul_orbs_mn_beta, false);
        for (int k = 0; k < as_dims[0]; k++) {
            for (int l = 0; l < as_dims[0]; l++) {
                for (int m = 0; m < as_dims[1]; m++) {
                    for (int n = 0; n < as_dims[1]; n++) {
                        auto tmp = Inner_prods(k * as_dims[0] + l, m * as_dims[1] + n);
                        two_body_integrals_alpha_beta(k, m, l, n) = tmp;
                    }
                }
            }
        }
    }
    as_integrals_two_body[0] = two_body_integrals_alpha_alpha;
    as_integrals_two_body[1] = two_body_integrals_alpha_beta;
    as_integrals_two_body[2] = two_body_integrals_beta_beta;


    /*
    // AS two electron integrals
    as_integrals_two_body = madness::Tensor<double>(as_dim, as_dim, as_dim, as_dim);
    madness::Tensor<double> Inner_prods = matrix_inner(*(madness_process.world), orbs_kl, coul_orbs_mn, false);
    for (int k = 0; k < as_dim; k++) {
        for (int l = 0; l < as_dim; l++) {
            for (int m = 0; m < as_dim; m++) {
                for (int n = 0; n < as_dim; n++) {
                    as_integrals_two_body(k, m, l, n) = Inner_prods(k * as_dim + l, m * as_dim + n);
                }
            }
        }
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    auto end_time = std::chrono::high_resolution_clock::now();

    std::cout << "Integral timings:" << std::endl;
    std::cout << "AS-AS one-electron integrals: " << std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count()
              << " seconds" << std::endl;
    std::cout << "AS-AS two-electron integrals: " << std::chrono::duration_cast<std::chrono::seconds>(t4 - t3).count()
              << " seconds" << std::endl;
    std::cout << "Core-AS one-electron integrals: " << std::chrono::duration_cast<std::chrono::seconds>(t5 - t4).count()
              << " seconds" << std::endl;
    std::cout << "Core-AS two-electron integrals: " << std::chrono::duration_cast<std::chrono::seconds>(t6 - t5).count()
              << " seconds" << std::endl;
    std::cout << "Full function: " << std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count()
              << " seconds" << std::endl;
    */


    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Calculate all integrals: " << std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count()
            << " seconds" << std::endl;








    //Old Code der noch übersetzt werden muss
    /*
    // Core-AS one electron integrals
    core_as_integrals_one_body_ak = madness::Tensor<double>(as_dim, as_dim);
    for (int a = 0; a < core_dim; a++) {
        for (int k = 0; k < as_dim; k++) {
            // Kinetic
            for (int axis = 0; axis < NDIM; axis++) {
                Derivative<double, NDIM> D = free_space_derivative<double, NDIM>(*(madness_process.world), axis);
                Function<double, NDIM> d_orb_a = D(frozen_occ_orbs[a]);
                Function<double, NDIM> d_orb_k = D(active_orbs[k]);
                core_as_integrals_one_body_ak(a, k) += 0.5 * inner(d_orb_a, d_orb_k);
            }
            // Nuclear
            Function<double, NDIM> Vnuc_orb_k;
            Vnuc_orb_k = Vnuc * active_orbs[k];

            core_as_integrals_one_body_ak(a, k) += inner(frozen_occ_orbs[a], Vnuc_orb_k);
        }
    }
    auto t5 = std::chrono::high_resolution_clock::now();

    // Core-AS two electron integrals <ak|al>
    if (core_dim > 0) {
        core_as_integrals_two_body_akal = madness::Tensor<double>(core_dim, as_dim, as_dim);
        std::vector<Function<double, NDIM>> orbs_aa;
        for (int a = 0; a < core_dim; a++) {
            orbs_aa.push_back(frozen_occ_orbs[a] * frozen_occ_orbs[a]);
        }
        orbs_aa = truncate(orbs_aa, truncation_tol);
        madness::Tensor<double> Inner_prods_akal = matrix_inner(*(madness_process.world), orbs_aa, coul_orbs_mn, false);
        for (int a = 0; a < core_dim; a++) {
            for (int k = 0; k < as_dim; k++) {
                for (int l = 0; l < as_dim; l++) {
                    core_as_integrals_two_body_akal(a, k, l) = Inner_prods_akal(a, k * as_dim + l);
                }
            }
        }
    }

    // Core-AS two electron integrals <ak|la>, <ak|ln>, <ab|ak> and <ba|ak>
    if (core_dim > 0) {
        core_as_integrals_two_body_akla = madness::Tensor<double>(core_dim, as_dim, as_dim);
        core_as_integrals_two_body_akln = madness::Tensor<double>(core_dim, as_dim, as_dim, as_dim);
        core_as_integrals_two_body_abak = madness::Tensor<double>(core_dim, core_dim, as_dim);
        core_as_integrals_two_body_baak = madness::Tensor<double>(core_dim, core_dim, as_dim);
        for (int a = 0; a < core_dim; a++) // One core orbital after the other -> Slightly less efficient than all a at
                                           // the same time, but reduces memory
        {
            std::vector<Function<double, NDIM>> orbs_ak = frozen_occ_orbs[a] * active_orbs;
            orbs_ak = truncate(orbs_ak, truncation_tol);
            std::vector<Function<double, NDIM>> coul_orbs_ak = apply(*(madness_process.world), *coul_op_parallel, orbs_ak);
            coul_orbs_ak = truncate(coul_orbs_ak, truncation_tol);

            std::vector<Function<double, NDIM>> orbs_ka = active_orbs * frozen_occ_orbs[a];
            orbs_ka = truncate(orbs_ka, truncation_tol);

            // <ak|la> = <ka|al>
            madness::Tensor<double> Inner_prods_akla =
                matrix_inner(*(madness_process.world), orbs_ka, coul_orbs_ak, false);
            for (int k = 0; k < as_dim; k++) {
                for (int l = 0; l < as_dim; l++) {
                    core_as_integrals_two_body_akla(a, k, l) = Inner_prods_akla(l, k);
                }
            }

            // <ak|ln>
            madness::Tensor<double> Inner_prods_akln =
                matrix_inner(*(madness_process.world), orbs_ak, coul_orbs_mn, false);
            for (int k = 0; k < as_dim; k++) {
                for (int l = 0; l < as_dim; l++) {
                    for (int n = 0; n < as_dim; n++) {
                        core_as_integrals_two_body_akln(a, l, k, n) = Inner_prods_akln(l, k * as_dim + n);
                    }
                }
            }

            // <ab|ak>
            for (int b = 0; b < core_dim; b++) {
                madness::Tensor<double> Inner_prods_abak = matrix_inner(
                    *(madness_process.world), std::vector<Function<double, NDIM>>{frozen_occ_orbs[b] * frozen_occ_orbs[b]},
                    coul_orbs_ak, false); // Change of indicies
                for (int k = 0; k < as_dim; k++) {
                    core_as_integrals_two_body_abak(b, a, k) = Inner_prods_abak(0, k);
                }
            }

            // <ba|ak>
            for (int b = 0; b < core_dim; b++) {
                madness::Tensor<double> Inner_prods_baak = matrix_inner(
                    *(madness_process.world), std::vector<Function<double, NDIM>>{frozen_occ_orbs[b] * frozen_occ_orbs[a]},
                    coul_orbs_ak, false);
                for (int k = 0; k < as_dim; k++) {
                    core_as_integrals_two_body_baak(a, b, k) = Inner_prods_baak(0, k);
                }
            }
        }
    }
    auto t6 = std::chrono::high_resolution_clock::now();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Integral timings:" << std::endl;
    std::cout << "Preparation AS pairs: " << std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count()
              << " seconds" << std::endl;
    std::cout << "AS-AS one-electron integrals: " << std::chrono::duration_cast<std::chrono::seconds>(t3 - t2).count()
              << " seconds" << std::endl;
    std::cout << "AS-AS two-electron integrals: " << std::chrono::duration_cast<std::chrono::seconds>(t4 - t3).count()
              << " seconds" << std::endl;
    std::cout << "Core-AS one-electron integrals: " << std::chrono::duration_cast<std::chrono::seconds>(t5 - t4).count()
              << " seconds" << std::endl;
    std::cout << "Core-AS two-electron integrals: " << std::chrono::duration_cast<std::chrono::seconds>(t6 - t5).count()
              << " seconds" << std::endl;
    std::cout << "Full function: " << std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count()
              << " seconds" << std::endl;
    */
}

template <std::size_t NDIM>
void Optimization_open_shell<NDIM>::calculate_core_energy() {
    /*
    auto start_time = std::chrono::high_resolution_clock::now();

    double nocc = 2; // Spatial orbitals = 2, Spin orbitals = 1
    double core_kinetic_energy = 0;
    double core_nuclear_attraction_energy = 0;
    double core_two_electron_energy = 0;

    if (frozen_occ_orbs.size() > 0) {
        // 1e Part
        for (int i = 0; i < frozen_occ_orbs.size(); i++) {
            // Kinetic
            for (int axis = 0; axis < NDIM; axis++) {
                Derivative<double, NDIM> D = free_space_derivative<double, NDIM>(*(madness_process.world), axis);
                Function<double, NDIM> d_orb = D(frozen_occ_orbs[i]);
                core_kinetic_energy += 0.5 * inner(d_orb, d_orb);
            }
            // Nuclear
            Function<double, NDIM> Vnuc_orb;
            Vnuc_orb = Vnuc * frozen_occ_orbs[i];

            core_nuclear_attraction_energy += inner(frozen_occ_orbs[i], Vnuc_orb);
        }
        core_kinetic_energy = nocc * core_kinetic_energy;
        core_nuclear_attraction_energy = nocc * core_nuclear_attraction_energy;

        // 2e Part
        auto coul_op_parallel =
            std::shared_ptr<SeparatedConvolution<double, NDIM>>(CoulombOperatorNDPtr<NDIM>(*(madness_process.world), coulomb_lo, coulomb_eps));

        // <ab|ab>
        {
            std::vector<Function<double, NDIM>> orbs_aa;
            for (int a = 0; a < core_dim; a++) {
                orbs_aa.push_back(frozen_occ_orbs[a] * frozen_occ_orbs[a]);
            }
            orbs_aa = truncate(orbs_aa, truncation_tol);
            std::vector<Function<double, NDIM>> coul_orbs_aa = apply(*(madness_process.world), *coul_op_parallel, orbs_aa);
            coul_orbs_aa = truncate(coul_orbs_aa, truncation_tol);
            for (int a = 0; a < core_dim; a++) {
                madness::Tensor<double> Inner_prods_abab = matrix_inner(
                    *(madness_process.world), std::vector<Function<double, NDIM>>{orbs_aa[a]}, coul_orbs_aa, false);
                for (int b = 0; b < core_dim; b++) {
                    core_two_electron_energy += 2 * Inner_prods_abab(0, b);
                }
            }
        }

        for (int a = 0; a < core_dim; a++) // One core orbital after the other -> Slightly less efficient than all a at
                                           // the same time, but reduces memory
        {
            std::vector<Function<double, NDIM>> orbs_ab = frozen_occ_orbs[a] * frozen_occ_orbs;
            orbs_ab = truncate(orbs_ab, truncation_tol);
            std::vector<Function<double, NDIM>> coul_orbs_ab = apply(*(madness_process.world), *coul_op_parallel, orbs_ab);
            coul_orbs_ab = truncate(coul_orbs_ab, truncation_tol);
            for (int b = 0; b < core_dim; b++) {
                core_two_electron_energy -= inner(orbs_ab[b], coul_orbs_ab[b]);
            }
        }

        core_two_electron_energy = 0.5 * nocc * core_two_electron_energy;
    }

    core_total_energy = core_kinetic_energy + core_nuclear_attraction_energy + core_two_electron_energy;
    print("                   Core - Kinetic energy ", core_kinetic_energy);
    print("        Core - Nuclear attraction energy ", core_nuclear_attraction_energy);
    print("              Core - Two-electron energy ", core_two_electron_energy);
    print("                       Total core energy ", core_total_energy);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "CalculateCoreEnergy took " << duration.count() << " seconds" << std::endl;
    */
    core_total_energy = 0;
}

template <std::size_t NDIM>
void Optimization_open_shell<NDIM>::calculate_energies() {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Active Space Part
    double as_one_electron_energy = 0.0;
    // One Particle Part
    for (int ab = 0; ab < 2; ab++) {
        for (int k = 0; k < as_dims[ab]; k++) {
            for (int l = 0; l < as_dims[ab]; l++) {
                as_one_electron_energy += as_one_rdm[ab](k, l) * as_integrals_one_body[ab](k, l);
            }
        }
    }

    // Two Particle Part alpha_alpha and beta_beta
    double as_two_electron_energy = 0.0;
    for (int k = 0; k < as_dims[0]; k++) {
        for (int l = 0; l < as_dims[0]; l++) {
            for (int m = 0; m < as_dims[0]; m++) {
                for (int n = 0; n < as_dims[0]; n++) {
                    as_two_electron_energy += as_two_rdm[0](k, l, m, n) * 0.5 * as_integrals_two_body[0](k, l, m, n);
                }
            }
        }
    }
        
    for (int k = 0; k < as_dims[1]; k++) {
        for (int l = 0; l < as_dims[1]; l++) {
            for (int m = 0; m < as_dims[1]; m++) {
                for (int n = 0; n < as_dims[1]; n++) {
                    as_two_electron_energy += as_two_rdm[2](k, l, m, n) * 0.5 * as_integrals_two_body[2](k, l, m, n);
                }
            }
        }
    }

    // Two Particle Part alpha_beta
    for (int k = 0; k < as_dims[0]; k++) {
        for (int l = 0; l < as_dims[0]; l++) {
            for (int m = 0; m < as_dims[1]; m++) {
                for (int n = 0; n < as_dims[1]; n++) {
                    as_two_electron_energy += 2 * as_two_rdm[1](k, l, m, n) * 0.5 * as_integrals_two_body[1](k, l, m, n);
                }
            }
        }
    }

    // Core Part
    double as_core_energy = 0.0; //TBD
    /*
    if (core_dim > 0) {
        for (int a = 0; a < core_dim; a++) {
            for (int k = 0; k < as_dim; k++) {
                for (int l = 0; l < as_dim; l++) {
                    as_core_energy += 2 * as_one_rdm(k, l) * core_as_integrals_two_body_akal(a, k, l);
                    as_core_energy -= as_one_rdm(k, l) * core_as_integrals_two_body_akla(a, k, l);
                }
            }
        }
        as_core_energy = 0.5 * nocc * as_core_energy;
    }
    */

    // Print results
    double total_energy =
        as_one_electron_energy + as_two_electron_energy + as_core_energy + nuclear_repulsion_energy + core_total_energy;

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

    madness::Tensor<double> LagrangeMultiplier_AS_AS_alpha = madness::Tensor<double>(as_dims[0], as_dims[0]);
    madness::Tensor<double> LagrangeMultiplier_AS_AS_beta = madness::Tensor<double>(as_dims[1], as_dims[1]);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "CalculateLagrangeMultiplier took " << duration.count() << " seconds" << std::endl;

    // alpha
    for (int z = 0; z < as_dims[0]; z++) {
        for (int i = 0; i < as_dims[0]; i++) {
            LagrangeMultiplier_AS_AS_alpha(z, i) = calculate_lagrange_multiplier_element_as_as(z, i, 0);
        }
    }

    //beta
    for (int z = 0; z < as_dims[1]; z++) {
        for (int i = 0; i < as_dims[1]; i++) {
            LagrangeMultiplier_AS_AS_beta(z, i) = calculate_lagrange_multiplier_element_as_as(z, i, 1);
        }
    }
    LagrangeMultiplier_AS_AS[0] = LagrangeMultiplier_AS_AS_alpha;
    LagrangeMultiplier_AS_AS[1] = LagrangeMultiplier_AS_AS_beta;

    /*
    //Old
    auto start_time = std::chrono::high_resolution_clock::now();

    LagrangeMultiplier_AS_AS = madness::Tensor<double>(as_dim, as_dim);

    for (int z = 0; z < as_dim; z++) {
        for (int i = 0; i < as_dim; i++) {
            LagrangeMultiplier_AS_AS(z, i) = calculate_lagrange_multiplier_element_as_as(z, i);
        }
    }

    if (core_dim > 0) {
        LagrangeMultiplier_AS_Core = madness::Tensor<double>(core_dim, as_dim);
        for (int z = 0; z < core_dim; z++) {
            for (int i = 0; i < as_dim; i++) {
                LagrangeMultiplier_AS_Core(z, i) = calculate_lagrange_multiplier_element_as_core(z, i);
            }
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "CalculateLagrangeMultiplier took " << duration.count() << " seconds" << std::endl;
    */
}

template <std::size_t NDIM>
double Optimization_open_shell<NDIM>::calculate_lagrange_multiplier_element_as_as(int z, int i, int spin) {
    
    int op_spin; if(spin == 0){op_spin = 1;} else {op_spin = 0;}
    int two_e_entry; if(spin == 0){two_e_entry = 0;} else {two_e_entry = 2;}

    double element = as_one_rdm[spin](i, i) * as_integrals_one_body[spin](z, i);

    // alpha-alpha / beta-beta
    for (int l = 0; l < as_dims[spin]; l++) {
        for (int n = 0; n < as_dims[spin]; n++) {
            for (int k = 0; k < as_dims[spin]; k++) {
                element += as_two_rdm[two_e_entry](k, l, i, n) * as_integrals_two_body[two_e_entry](z, l, k, n);
            }
        }
    }

    //mixed terms
    for (int l = 0; l < as_dims[op_spin]; l++) {
        for (int n = 0; n < as_dims[op_spin]; n++) {
            for (int k = 0; k < as_dims[spin]; k++) {
                if(spin == 0)
                {
                    element += as_two_rdm[1](k, l, i, n) * as_integrals_two_body[1](z, l, k, n);
                }
                else
                {
                    element += as_two_rdm[1](l, k, n, i) * as_integrals_two_body[1](l, z, n, k);
                }
            }
        }
    }

    return element;

    /*
    double element = as_one_rdm(i, i) * as_integrals_one_body(z, i);
    for (int l = 0; l < as_dim; l++) {
        for (int n = 0; n < as_dim; n++) {
            for (int k = 0; k < as_dim; k++) {
                element += as_two_rdm(k, l, i, n) * as_integrals_two_body(z, l, k, n);
            }
        }
    }

    for (int k = 0; k < as_dim; k++) {
        for (int a = 0; a < core_dim; a++) {
            element += 0.5 * nocc * 2 * as_one_rdm(k, i) * core_as_integrals_two_body_akal(a, z, k);
            element -= 0.5 * nocc * as_one_rdm(k, i) * core_as_integrals_two_body_akla(a, k, z);
        }
    }
    return element;
    */
}

template <std::size_t NDIM>
double Optimization_open_shell<NDIM>::calculate_lagrange_multiplier_element_as_core(int z, int i) {
    /*
    double element = as_one_rdm(i, i) * core_as_integrals_one_body_ak(z, i);

    for (int l = 0; l < as_dim; l++) {
        for (int n = 0; n < as_dim; n++) {
            for (int k = 0; k < as_dim; k++) {
                element += as_two_rdm(k, l, i, n) * core_as_integrals_two_body_akln(z, k, l, n);
            }
        }
    }

    for (int k = 0; k < as_dim; k++) {
        for (int a = 0; a < core_dim; a++) {
            element += 0.5 * nocc * 2 * as_one_rdm(k, i) * core_as_integrals_two_body_abak(a, z, k);
            element -= 0.5 * nocc * as_one_rdm(k, i) * core_as_integrals_two_body_baak(a, z, k);
        }
    }
    return element;
    */
    return 0;
}

template <std::size_t NDIM>
bool Optimization_open_shell<NDIM>::optimize_orbitals(double optimization_thresh, double NO_occupation_thresh, int maxiter) {

    bool converged = false;
    int iterstep = 0;
    while (!converged && iterstep < maxiter) {
        iterstep++;
        std::cout << "---------------------------------------------------" << std::endl;
        std::cout << "Start iteration step: " << iterstep << std::endl;

        // Update LagrangeMultiplier
        calculate_lagrange_multiplier();

        // Update orbitals
        auto start_orb_update_time = std::chrono::high_resolution_clock::now();
        highest_error = 0;

        std::array<std::vector<int>, 2> as_orbital_indicies_for_update;

        for (int ab = 0; ab < 2; ab++)
        {
            for (int idx = 0; idx < as_dims[ab]; idx++) {
                if (abs(as_one_rdm[ab](idx, idx)) >= NO_occupation_thresh) {
                    as_orbital_indicies_for_update[ab].push_back(idx);
                } else {
                    std::cout << "Skip refinement of spin" << ab << "active space orbital " << idx
                            << ", since the occupation is less than NO_occupation_thresh (" << NO_occupation_thresh << ")"
                            << std::endl;
                }
            }
        }

        std::array<std::vector<Function<double, NDIM>>, 2> AllActiveOrbitalUpdates = get_all_active_orbital_updates(as_orbital_indicies_for_update);

        for (int ab = 0; ab < 2; ab++)
        {
            for (int idx = 0; idx < as_orbital_indicies_for_update[ab].size(); idx++) {
                int actIdx = as_orbital_indicies_for_update[ab][idx];
                active_orbs[ab][actIdx] = active_orbs[ab][actIdx] - AllActiveOrbitalUpdates[ab][idx];
            }
        }

        auto end_orb_update_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_orb_update_time - start_orb_update_time);
        std::cout << "UpdateOrbitals took " << duration.count() << " seconds" << std::endl;

        // Orthonormalize orbitals
        for(int ab = 0; ab < 2; ab++)
        {
            active_orbs[ab] = orthonormalize_symmetric(active_orbs[ab]);
            // orbitals = orthonormalize_cd(orbitals);
            active_orbs[ab] = truncate(active_orbs[ab], truncation_tol);
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
    for (int ab = 0; ab < 2; ab++)
    {
        for (int idx = 0; idx < orbital_indicies_for_update[ab].size(); idx++) {
            int i = orbital_indicies_for_update[ab][idx];
            rdm_ii_inv[ab].push_back(1 / as_one_rdm[ab](i, i));
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    // 1e Part
    for (int ab = 0; ab < 2; ab++)
    {
        for (int idx = 0; idx < orbital_indicies_for_update[ab].size(); idx++) {
            int i = orbital_indicies_for_update[ab][idx];

            Function<double, NDIM> rhs;
            rhs = Vnuc * active_orbs[ab][i];
            //CORE TEIL WIEDER EINFÜGEN !!!
            //for (int k = 0; k < core_dim; k++) {
            //   rhs -= rdm_ii_inv[idx] * LagrangeMultiplier_AS_Core(k, i) * frozen_occ_orbs[k];
            //}
            for (int k = 0; k < as_dims[ab]; k++) {
                if (k != i) {
                    rhs -= rdm_ii_inv[ab][idx] * LagrangeMultiplier_AS_AS[ab](k, i) * active_orbs[ab][k];
                }
            }
            AllOrbitalUpdates[ab].push_back(rhs);
        }
    }

    AllOrbitalUpdates[0] = truncate(AllOrbitalUpdates[0], truncation_tol);
    AllOrbitalUpdates[1] = truncate(AllOrbitalUpdates[1], truncation_tol);
    auto t3 = std::chrono::high_resolution_clock::now();


    // AS Part
    for (int ab = 0; ab < 2; ab++)
    {
        //alpha-alpha and beta-beta
        int two_e_entry; if(ab == 0){two_e_entry = 0;} else {two_e_entry = 2;}
        for (int k = 0; k < as_dims[ab]; k++) {
            std::vector<Function<double, NDIM>> lnk = coul_orbs_mn[ab] * active_orbs[ab][k];
            lnk = truncate(lnk, truncation_tol);
            for (int idx = 0; idx < orbital_indicies_for_update[ab].size(); idx++) {
                int i = orbital_indicies_for_update[ab][idx];
                std::vector<Function<double, NDIM>> lnk_copy = copy(*(madness_process.world), lnk, false);
                for (int l = 0; l < as_dims[ab]; l++) {
                    for (int n = 0; n < as_dims[ab]; n++) {
                        lnk_copy[l * as_dims[ab] + n] *= as_two_rdm[two_e_entry](k, l, i, n) * rdm_ii_inv[ab][idx];
                    }
                }
                AllOrbitalUpdates[ab][idx] += sum(*(madness_process.world), lnk_copy);
            }
        }

        //mixed terms
        int op_spin; if(ab == 0){op_spin = 1;} else {op_spin = 0;}
        for (int k = 0; k < as_dims[ab]; k++) {
            std::vector<Function<double, NDIM>> lnk = coul_orbs_mn[op_spin] * active_orbs[ab][k];
            lnk = truncate(lnk, truncation_tol);
            for (int idx = 0; idx < orbital_indicies_for_update[ab].size(); idx++) {
                int i = orbital_indicies_for_update[ab][idx];
                std::vector<Function<double, NDIM>> lnk_copy = copy(*(madness_process.world), lnk, false);
                for (int l = 0; l < as_dims[op_spin]; l++) {
                    for (int n = 0; n < as_dims[op_spin]; n++) {
                        if(ab == 0)
                        {
                            lnk_copy[l * as_dims[op_spin] + n] *= as_two_rdm[1](k, l, i, n) * rdm_ii_inv[ab][idx];
                        }
                        else
                        {
                            lnk_copy[l * as_dims[op_spin] + n] *= as_two_rdm[1](l, k, n, i) * rdm_ii_inv[ab][idx];
                        }
                    }
                }
                AllOrbitalUpdates[ab][idx] += sum(*(madness_process.world), lnk_copy);
            }
        }
    }
    auto t4 = std::chrono::high_resolution_clock::now();

    // Core - AS interaction EINFÜGEN !!!!
    
    // BSH part
    for (int ab = 0; ab < 2; ab++)
    {
        for (int idx = 0; idx < orbital_indicies_for_update[ab].size(); idx++) {
            int i = orbital_indicies_for_update[ab][idx];
            double en = LagrangeMultiplier_AS_AS[ab](i, i) * rdm_ii_inv[ab][idx];
            SeparatedConvolution<double, NDIM> bsh_op =
                BSHOperator<NDIM>(*(madness_process.world), sqrt(-2 * en), BSH_lo, BSH_eps);
            Function<double, NDIM> r = active_orbs[ab][i] + 2.0 * bsh_op(AllOrbitalUpdates[ab][idx]); // the residual
            double err = r.norm2();
            std::cout << "Error of spin " << ab << " active space orbital " << i << ": " << err << std::endl;
            if (err > highest_error) {
                highest_error = err;
            }
            AllOrbitalUpdates[ab][idx] = r;
        }
    }
    auto t5 = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();

    /*
    std::cout << "Refinement timings:" << std::endl;
    std::cout << "rdm_ii_inv calculation: " << std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count()
              << " seconds" << std::endl;
    std::cout << "one electron part: " << std::chrono::duration_cast<std::chrono::seconds>(t3 - t2).count()
              << " seconds" << std::endl;
    std::cout << "AS two electron part: " << std::chrono::duration_cast<std::chrono::seconds>(t4 - t3).count()
              << " seconds" << std::endl;
    std::cout << "BSH part: " << std::chrono::duration_cast<std::chrono::seconds>(t5 - t4).count() << " seconds"
              << std::endl;
    std::cout << "Full function: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
              << " seconds" << std::endl;
    */
    return AllOrbitalUpdates;
    
    
    
    
    
    
    
    
    
    
    
    
    
    /*
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<Function<double, NDIM>> AllOrbitalUpdates;

    auto t1 = std::chrono::high_resolution_clock::now();
    // Calculate rdm_ii_inv values
    std::vector<double> rdm_ii_inv;
    for (int idx = 0; idx < orbital_indicies_for_update.size(); idx++) {
        int i = orbital_indicies_for_update[idx];
        rdm_ii_inv.push_back(1 / as_one_rdm(i, i));
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    // 1e Part
    for (int idx = 0; idx < orbital_indicies_for_update.size(); idx++) {
        int i = orbital_indicies_for_update[idx];

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
    AllOrbitalUpdates = truncate(AllOrbitalUpdates, truncation_tol);
    auto t3 = std::chrono::high_resolution_clock::now();

    // AS Part
    for (int k = 0; k < as_dim; k++) {
        std::vector<Function<double, NDIM>> lnk = coul_orbs_mn * active_orbs[k];
        lnk = truncate(lnk, truncation_tol);
        for (int idx = 0; idx < orbital_indicies_for_update.size(); idx++) {
            int i = orbital_indicies_for_update[idx];
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
        std::shared_ptr<SeparatedConvolution<double, NDIM>>(CoulombOperatorNDPtr<NDIM>(*(madness_process.world), 0.001, 1e-6));

    if (core_dim > 0) {
        // Part 1
        std::vector<Function<double, NDIM>> orbs_aa;
        for (int a = 0; a < core_dim; a++) {
            orbs_aa.push_back(frozen_occ_orbs[a] * frozen_occ_orbs[a]);
        }
        orbs_aa = truncate(orbs_aa, truncation_tol);
        std::vector<Function<double, NDIM>> coul_orbs_aa = apply(*(madness_process.world), *coul_op_parallel, orbs_aa);
        coul_orbs_aa = truncate(coul_orbs_aa, truncation_tol);

        for (int k = 0; k < as_dim; k++) {
            std::vector<Function<double, NDIM>> aak = coul_orbs_aa * active_orbs[k];
            aak = truncate(aak, truncation_tol);
            for (int idx = 0; idx < orbital_indicies_for_update.size(); idx++) {
                int i = orbital_indicies_for_update[idx];
                std::vector<Function<double, NDIM>> aak_copy = copy(*(madness_process.world), aak, false);
                for (int a = 0; a < core_dim; a++) {
                    aak_copy[a] *= 0.5 * nocc * 2 * as_one_rdm(k, i) * rdm_ii_inv[idx];
                    // AllOrbitalUpdates[idx] += 0.5 * 2 * rdm_ii_inv[idx] * as_one_rdm(k, i) * 2 * coul_orbs_aa[a] *
                    // active_orbs[k];
                }
                AllOrbitalUpdates[idx] += sum(*(madness_process.world), aak_copy);
            }
        }
    }
    auto t5 = std::chrono::high_resolution_clock::now();

    // Part 2
    if (core_dim > 0) {
        for (int a = 0; a < core_dim; a++) {
            std::vector<Function<double, NDIM>> orbs_ak = frozen_occ_orbs[a] * active_orbs;
            orbs_ak = truncate(orbs_ak, truncation_tol);
            std::vector<Function<double, NDIM>> coul_orbs_ak = apply(*(madness_process.world), *coul_op_parallel, orbs_ak);
            coul_orbs_ak = truncate(coul_orbs_ak, truncation_tol);

            std::vector<Function<double, NDIM>> aka = coul_orbs_ak * frozen_occ_orbs[a];
            aka = truncate(aka, truncation_tol);

            for (int idx = 0; idx < orbital_indicies_for_update.size(); idx++) {
                int i = orbital_indicies_for_update[idx];
                std::vector<Function<double, NDIM>> aka_copy = copy(*(madness_process.world), aka, false);
                for (int k = 0; k < as_dim; k++) {
                    aka_copy[k] *= -0.5 * nocc * as_one_rdm(k, i) * rdm_ii_inv[idx];
                    // AllOrbitalUpdates[idx] -= 0.5 * rdm_ii_inv[idx] * as_one_rdm(k, i) * 2 * coul_orbs_ak[k] *
                    // frozen_occ_orbs[a];
                }
                AllOrbitalUpdates[idx] += sum(*(madness_process.world), aka_copy);
            }
        }
    }
    auto t6 = std::chrono::high_resolution_clock::now();

    // BSH part
    for (int idx = 0; idx < orbital_indicies_for_update.size(); idx++) {
        int i = orbital_indicies_for_update[idx];
        double en = LagrangeMultiplier_AS_AS(i, i) * rdm_ii_inv[idx];
        SeparatedConvolution<double, NDIM> bsh_op =
            BSHOperator<NDIM>(*(madness_process.world), sqrt(-2 * en), BSH_lo, BSH_eps);
        Function<double, NDIM> r = active_orbs[i] + 2.0 * bsh_op(AllOrbitalUpdates[idx]); // the residual
        double err = r.norm2();
        std::cout << "Error of active space orbital " << i << ": " << err << std::endl;
        if (err > highest_error) {
            highest_error = err;
        }
        AllOrbitalUpdates[idx] = r;
    }
    auto t7 = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Refinement timings:" << std::endl;
    std::cout << "rdm_ii_inv calculation: " << std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count()
              << " seconds" << std::endl;
    std::cout << "one electron part: " << std::chrono::duration_cast<std::chrono::seconds>(t3 - t2).count()
              << " seconds" << std::endl;
    std::cout << "AS two electron part: " << std::chrono::duration_cast<std::chrono::seconds>(t4 - t3).count()
              << " seconds" << std::endl;
    std::cout << "Core-AS two electron part 1: " << std::chrono::duration_cast<std::chrono::seconds>(t5 - t4).count()
              << " seconds" << std::endl;
    std::cout << "Core-AS two electron part 2: " << std::chrono::duration_cast<std::chrono::seconds>(t6 - t5).count()
              << " seconds" << std::endl;
    std::cout << "BSH part: " << std::chrono::duration_cast<std::chrono::seconds>(t7 - t6).count() << " seconds"
              << std::endl;
    std::cout << "Full function: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
              << " seconds" << std::endl;

    return AllOrbitalUpdates;
    */
}

template <std::size_t NDIM>
void Optimization_open_shell<NDIM>::rotate_orbitals_back() {
    /*
    madness::Tensor<double> RotationMatrixBack = madness::Tensor<double>(as_dim, as_dim);
    for (int i = 0; i < as_dim; i++) {
        for (int j = 0; j < as_dim; j++) {
            RotationMatrixBack(i, j) = ActiveSpaceRotationMatrix(j, i);
        }
    }

    TransformMatrix(&as_one_rdm, RotationMatrixBack);
    TransformTensor(as_two_rdm, RotationMatrixBack);
    active_orbs = transform(*(madness_process.world), active_orbs, RotationMatrixBack);
    calculate_all_integrals();
    */
}

template <std::size_t NDIM>
void Optimization_open_shell<NDIM>::save_orbitals(std::string OutputPath) {
    /*
    for (int i = 0; i < core_dim; i++) {
        std::string base_filename = frozen_occ_orbs_files[i].substr(frozen_occ_orbs_files[i].find_last_of("/\\") + 1);
        save(frozen_occ_orbs[i], OutputPath + "/" + base_filename);
    }
    for (int i = 0; i < as_dim; i++) {
        std::string base_filename = active_orbs_files[i].substr(active_orbs_files[i].find_last_of("/\\") + 1);
        save(active_orbs[i], OutputPath + "/" + base_filename);
    }
    for (int i = 0; i < froz_virt_dim; i++) {
        std::string base_filename = frozen_virt_orb_files[i].substr(frozen_virt_orb_files[i].find_last_of("/\\") + 1);
        save(frozen_virt_orb[i], OutputPath + "/" + base_filename);
    }
    */
}

template <std::size_t NDIM>
std::vector<SavedFct<NDIM>> Optimization_open_shell<NDIM>::get_orbitals() {
    /*
    std::vector<SavedFct<NDIM>> all_orbs;
    int j = 0;
    for (int i = 0; i < core_dim; i++) {
        SavedFct<NDIM> orb(frozen_occ_orbs[i]);
        orb.type = "frozen_occ";
        all_orbs.push_back(orb);
        j++;
        std::cout << "Norm orb " << i << ":" << frozen_occ_orbs[i].norm2() << std::endl;
    }
    for (int i = 0; i < as_dim; i++) {
        SavedFct<NDIM> orb(active_orbs[i]);
        orb.type = "active";
        all_orbs.push_back(orb);
        j++;
        std::cout << "Norm orb " << i << ":" << active_orbs[i].norm2() << std::endl;
    }
    for (int i = 0; i < froz_virt_dim; i++) {
        SavedFct<NDIM> orb(frozen_virt_orb[i]);
        orb.type = "frozen_virt";
        all_orbs.push_back(orb);
        j++;
        std::cout << "Norm orb " << i << ":" << frozen_virt_orb[i].norm2() << std::endl;
    }
    return all_orbs;
    */
}

template <std::size_t NDIM>
void Optimization_open_shell<NDIM>::save_effective_hamiltonian(std::string OutputPath) {
    /*
    std::vector<double> effective_one_body_integrals_elements;
    std::vector<double> effective_two_body_integrals_elements;

    madness::Tensor<double> effective_one_body_integrals = as_integrals_one_body;
    for (int k = 0; k < as_dim; k++) {
        for (int l = 0; l < as_dim; l++) {
            for (int a = 0; a < core_dim; a++) {
                effective_one_body_integrals(k, l) +=
                    0.5 * nocc *
                    (2 * core_as_integrals_two_body_akal(a, k, l) - core_as_integrals_two_body_akla(a, k, l));
            }
            effective_one_body_integrals_elements.push_back(effective_one_body_integrals(k, l));
        }
    }

    for (int k = 0; k < as_dim; k++) {
        for (int l = 0; l < as_dim; l++) {
            for (int m = 0; m < as_dim; m++) {
                for (int n = 0; n < as_dim; n++) {
                    effective_two_body_integrals_elements.push_back(as_integrals_two_body(k, l, m, n));
                }
            }
        }
    }

    std::ofstream c_file;
    c_file.open(OutputPath + "/c.txt");
    c_file << std::setprecision(15) << (core_total_energy + nuclear_repulsion_energy);
    c_file.close();

    std::vector<unsigned long> one_e_ints_shape{(unsigned long)as_dim, (unsigned long)as_dim};
    const npy::npy_data<double> one_e_data{effective_one_body_integrals_elements, one_e_ints_shape, false};
    npy::write_npy(OutputPath + "/htensor.npy", one_e_data);

    std::vector<unsigned long> two_e_ints_shape{(unsigned long)as_dim, (unsigned long)as_dim, (unsigned long)as_dim,
                                                (unsigned long)as_dim};
    const npy::npy_data<double> two_e_data{effective_two_body_integrals_elements, two_e_ints_shape, false};
    npy::write_npy(OutputPath + "/gtensor.npy", two_e_data);
    */
}

// the following three functions correspond to SaveEffectiveHamiltonian
template <std::size_t NDIM>
double Optimization_open_shell<NDIM>::get_c() {
    //return core_total_energy + nuclear_repulsion_energy;
    return 0;
}

// information about the shape of the tensor is lost if we pass it to python like this. might be better to give a
// tuple with the shape and the elements of the tensor, or write a class which has shape and a list of elements as
// members
template <std::size_t NDIM>
std::vector<double> Optimization_open_shell<NDIM>::get_h_tensor() {
    std::vector<double> effective_one_body_integrals_elements;

    /*
    madness::Tensor<double> effective_one_body_integrals = as_integrals_one_body;
    for (int k = 0; k < as_dim; k++) {
        for (int l = 0; l < as_dim; l++) {
            for (int a = 0; a < core_dim; a++) {
                effective_one_body_integrals(k, l) +=
                    0.5 * nocc *
                    (2 * core_as_integrals_two_body_akal(a, k, l) - core_as_integrals_two_body_akla(a, k, l));
            }
            effective_one_body_integrals_elements.push_back(effective_one_body_integrals(k, l));
        }
    }*/

    return effective_one_body_integrals_elements;
}

// same issue as in the htensor case
template <std::size_t NDIM>
std::vector<double> Optimization_open_shell<NDIM>::get_g_tensor() {
    std::vector<double> effective_two_body_integrals_elements;

    /*
    for (int k = 0; k < as_dim; k++) {
        for (int l = 0; l < as_dim; l++) {
            for (int m = 0; m < as_dim; m++) {
                for (int n = 0; n < as_dim; n++) {
                    effective_two_body_integrals_elements.push_back(as_integrals_two_body(k, l, m, n));
                }
            }
        }
    }
    */

    return effective_two_body_integrals_elements;
}

template class Optimization_open_shell<2>;
template class Optimization_open_shell<3>;