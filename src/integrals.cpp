#include "integrals.hpp"

using namespace madness;

template <std::size_t NDIM> Integrals<NDIM>::Integrals(MadnessProcess<NDIM>& mp) : madness_process(mp) {}

template <std::size_t NDIM>
nb::ndarray<nb::numpy, double, nb::ndim<2>>
Integrals<NDIM>::compute_potential_integrals(std::vector<Function<double, NDIM>> orbitals, Function<double, NDIM> V) {
    int as_dim = orbitals.size();
    potential_integrals = madness::matrix_inner(*(madness_process.world), orbitals, V * orbitals);
    nb::ndarray<nb::numpy, double, nb::ndim<2>> numpy_array(potential_integrals.ptr(),
                                                            {orbitals.size(), orbitals.size()});
    return numpy_array;
}

template <std::size_t NDIM>
nb::ndarray<nb::numpy, double, nb::ndim<2>>
Integrals<NDIM>::compute_overlap_integrals(std::vector<Function<double, NDIM>> orbitals1,
                                           std::vector<Function<double, NDIM>> orbitals2) {
    overlap_integrals = madness::matrix_inner(*(madness_process.world), orbitals1, orbitals2);
    nb::ndarray<nb::numpy, double, nb::ndim<2>> numpy_array(overlap_integrals.ptr(),
                                                            {orbitals1.size(), orbitals2.size()});
    return numpy_array;
}

template <std::size_t NDIM>
nb::ndarray<nb::numpy, double, nb::ndim<2>>
Integrals<NDIM>::compute_kinetic_integrals(std::vector<Function<double, NDIM>> orbitals) {
    int as_dim = orbitals.size();

    kinetic_integrals = madness::Tensor<double>(as_dim, as_dim);
    for (int k = 0; k < as_dim; k++) {
        for (int l = 0; l < as_dim; l++) {
            // Kinetic
            for (int axis = 0; axis < NDIM; axis++) {
                Derivative<double, NDIM> D = free_space_derivative<double, NDIM>(*(madness_process.world), axis);
                Function<double, NDIM> d_orb_k = D(orbitals[k]);
                Function<double, NDIM> d_orb_l = D(orbitals[l]);
                kinetic_integrals(k, l) += 0.5 * inner(d_orb_k, d_orb_l);
            }
        }
    }

    nb::ndarray<nb::numpy, double, nb::ndim<2>> numpy_array(kinetic_integrals.ptr(),
                                                            {orbitals.size(), orbitals.size()});
    return numpy_array;
}

// todo: allow 4 sets of orbitals A,B,C,D as input for <AB|g|CD> all indices
template <std::size_t NDIM>
nb::ndarray<nb::numpy, double, nb::ndim<4>>
Integrals<NDIM>::compute_two_body_integrals(std::vector<Function<double, NDIM>> orbitals, double truncation_tol,
                                            double coulomb_lo, double coulomb_eps, int nocc) {
    int as_dim = orbitals.size();

    auto t1 = std::chrono::high_resolution_clock::now();
    std::vector<Function<double, NDIM>> orbs_kl;
    for (int k = 0; k < as_dim; k++) {
        std::vector<Function<double, NDIM>> kl = orbitals[k] * orbitals;
        orbs_kl.insert(std::end(orbs_kl), std::begin(kl), std::end(kl));
    }
    orbs_kl = truncate(orbs_kl, truncation_tol);

    auto t2 = std::chrono::high_resolution_clock::now();
    auto coul_op_parallel = std::shared_ptr<SeparatedConvolution<double, NDIM>>(
        CoulombOperatorNDPtr<NDIM>(*(madness_process.world), coulomb_lo, coulomb_eps));
    std::vector<Function<double, NDIM>> coul_orbs_mn = apply(*(madness_process.world), *coul_op_parallel, orbs_kl);
    coul_orbs_mn = truncate(coul_orbs_mn, truncation_tol);

    auto t3 = std::chrono::high_resolution_clock::now();
    two_body_integrals = madness::Tensor<double>(as_dim, as_dim, as_dim, as_dim);
    madness::Tensor<double> Inner_prods = matrix_inner(*(madness_process.world), orbs_kl, coul_orbs_mn, false);
    std::vector<double> flat;
    for (int k = 0; k < as_dim; k++) {
        for (int l = 0; l < as_dim; l++) {
            for (int m = 0; m < as_dim; m++) {
                for (int n = 0; n < as_dim; n++) {
                    auto tmp = Inner_prods(k * as_dim + l, m * as_dim + n);
                    two_body_integrals(k, m, l, n) = tmp;
                    flat.push_back(tmp);
                }
            }
        }
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    // todo: save timings from Fabian

    nb::ndarray<nb::numpy, double, nb::ndim<4>> numpy_array(
        two_body_integrals.ptr(), {orbitals.size(), orbitals.size(), orbitals.size(), orbitals.size()});
    return numpy_array;
}

template <std::size_t NDIM>
nb::ndarray<nb::numpy, double, nb::ndim<2>>
Integrals<NDIM>::compute_frozen_core_interaction(std::vector<Function<double, NDIM>> frozen_occ_orbs,
                                                 std::vector<Function<double, NDIM>> active_orbs, double truncation_tol,
                                                 double coulomb_lo, double coulomb_eps, int nocc) {
    unsigned long core_dim = frozen_occ_orbs.size();
    unsigned long as_dim = active_orbs.size();

    std::vector<Function<double, NDIM>> orbs_kl;
    for (int k = 0; k < as_dim; k++) {
        std::vector<Function<double, NDIM>> kl = active_orbs[k] * active_orbs;
        orbs_kl.insert(std::end(orbs_kl), std::begin(kl), std::end(kl));
    }
    orbs_kl = truncate(orbs_kl, truncation_tol);

    auto t2 = std::chrono::high_resolution_clock::now();
    auto coul_op_parallel = std::shared_ptr<SeparatedConvolution<double, NDIM>>(
        CoulombOperatorNDPtr<NDIM>(*(madness_process.world), coulomb_lo, coulomb_eps));
    std::vector<Function<double, NDIM>> coul_orbs_mn = apply(*(madness_process.world), *coul_op_parallel, orbs_kl);
    coul_orbs_mn = truncate(coul_orbs_mn, truncation_tol);

    // Core-AS two electron integrals <ak|al>
    auto core_as_integrals_two_body_akal = madness::Tensor<double>(core_dim, as_dim, as_dim);
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

    auto core_as_integrals_two_body_akla = madness::Tensor<double>(core_dim, as_dim, as_dim);
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
        madness::Tensor<double> Inner_prods_akla = matrix_inner(*(madness_process.world), orbs_ka, coul_orbs_ak, false);
        for (int k = 0; k < as_dim; k++) {
            for (int l = 0; l < as_dim; l++) {
                core_as_integrals_two_body_akla(a, k, l) = Inner_prods_akla(l, k);
            }
        }
    }

    frozen_core_interaction = madness::Tensor<double>(as_dim, as_dim);
    for (int k = 0; k < as_dim; k++) {
        for (int l = 0; l < as_dim; l++) {
            for (int a = 0; a < core_dim; a++) {
                frozen_core_interaction(k, l) +=
                    0.5 * nocc *
                    (2 * core_as_integrals_two_body_akal(a, k, l) - core_as_integrals_two_body_akla(a, k, l));
            }
        }
    }
    nb::ndarray<nb::numpy, double, nb::ndim<2>> numpy_array(frozen_core_interaction.ptr(), {as_dim, as_dim});
    return numpy_array;
}

template <std::size_t NDIM>
std::vector<Function<double, NDIM>> Integrals<NDIM>::normalize(std::vector<Function<double, NDIM>> basis) {

    madness::normalize(*(madness_process.world), basis);
    return basis;
}

template <std::size_t NDIM>
std::vector<Function<double, NDIM>> Integrals<NDIM>::orthonormalize(std::vector<Function<double, NDIM>> basis,
                                                                    const std::string method, double rr_thresh) {
    // compute overlap, to be passed in orthonormalization routines and potentially printed
    auto S = madness::matrix_inner(*(madness_process.world), basis, basis, true);

    auto out_basis = basis;
    if (method == "cholesky") {
        out_basis = madness::orthonormalize_cd(basis, S);
    } else if (method == "symmetric") {
        out_basis = madness::orthonormalize_symmetric(basis, S);
    } else if (method == "canonical") {
        out_basis = madness::orthonormalize_canonical(basis, S, rr_thresh);
    } else if (method == "rr_cholesky") {
        out_basis = madness::orthonormalize_rrcd(basis, S, rr_thresh);
    } else {
        MADNESS_EXCEPTION("unknown orthonormalization method", 1);
    }

    return normalize(out_basis);
}

template <std::size_t NDIM>
std::vector<Function<double, NDIM>> Integrals<NDIM>::project_out(std::vector<Function<double, NDIM>> kernel,
                                                                 std::vector<Function<double, NDIM>> target) {

    auto Q = madness::QProjector<double, NDIM>(kernel);
    auto z = Q(target);
    madness::normalize(*(madness_process.world), z);
    return z;
}

template <std::size_t NDIM>
std::vector<Function<double, NDIM>> Integrals<NDIM>::project_on(std::vector<Function<double, NDIM>> kernel,
                                                                std::vector<Function<double, NDIM>> target) {

    auto P = madness::Projector<double, NDIM>(kernel);
    auto z = P(target);
    madness::normalize(*(madness_process.world), z);
    return z;
}

template <std::size_t NDIM>
std::vector<Function<double, NDIM>> Integrals<NDIM>::transform(std::vector<Function<double, NDIM>> orbitals,
                                                               nb::ndarray<nb::numpy, double, nb::ndim<2>> matrix) {

    // @todo there are more efficient ways (flatten and rewire the pointer of the first entry)
    madness::Tensor<double> U(matrix.shape(0), matrix.shape(1));
    for (auto k = 0; k < matrix.shape(0); ++k) {
        for (auto l = 0; l < matrix.shape(1); ++l) {
            U(k, l) = matrix(k, l);
        }
    }

    auto y = madness::transform(*(madness_process.world), orbitals, U);

    return y;
}

template class Integrals<2>;
template class Integrals<3>;