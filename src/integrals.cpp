#include "integrals.hpp"

using namespace madness;

template <std::size_t NDIM>
Integrals<NDIM>::Integrals(MadnessProcess<NDIM>& mp) : madness_process(mp) {}

template <std::size_t NDIM>
nb::ndarray<nb::numpy, double, nb::ndim<2>> Integrals<NDIM>::compute_potential_integrals(std::vector<SavedFct<NDIM>> all_orbs,
                                                                                   SavedFct<NDIM> potential) {
    std::vector<Function<double, NDIM>> orbitals;
    Function<double, NDIM> V = madness_process.loadfct(potential);
    for (SavedFct<NDIM> orb : all_orbs)
        orbitals.push_back(madness_process.loadfct(orb));
    int as_dim = orbitals.size();
    potential_integrals = madness::matrix_inner(*(madness_process.world), orbitals, V * orbitals);
    nb::ndarray<nb::numpy, double, nb::ndim<2>> numpy_array(potential_integrals.ptr(),
                                                            {orbitals.size(), orbitals.size()});
    return numpy_array;
}

template <std::size_t NDIM>
nb::ndarray<nb::numpy, double, nb::ndim<2>> Integrals<NDIM>::compute_overlap_integrals(std::vector<SavedFct<NDIM>> all_orbs,
                                                                                 std::vector<SavedFct<NDIM>> other) {
    std::vector<Function<double, NDIM>> orbitals1;
    for (SavedFct<NDIM> orb : all_orbs)
        orbitals1.push_back(madness_process.loadfct(orb));
    std::vector<Function<double, NDIM>> orbitals2;
    for (SavedFct<NDIM> orb : other)
        orbitals2.push_back(madness_process.loadfct(orb));
    overlap_integrals = madness::matrix_inner(*(madness_process.world), orbitals1, orbitals2);
    nb::ndarray<nb::numpy, double, nb::ndim<2>> numpy_array(overlap_integrals.ptr(),
                                                            {orbitals1.size(), orbitals2.size()});
    return numpy_array;
}

template <std::size_t NDIM>
nb::ndarray<nb::numpy, double, nb::ndim<2>> Integrals<NDIM>::compute_kinetic_integrals(std::vector<SavedFct<NDIM>> all_orbs) {
    std::vector<Function<double, NDIM>> orbitals;
    for (SavedFct<NDIM> orb : all_orbs)
        orbitals.push_back(madness_process.loadfct(orb));
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
nb::ndarray<nb::numpy, double, nb::ndim<4>> Integrals<NDIM>::compute_two_body_integrals(std::vector<SavedFct<NDIM>> all_orbs) {
    std::vector<Function<double, NDIM>> orbitals;
    for (SavedFct<NDIM> orb : all_orbs)
        orbitals.push_back(madness_process.loadfct(orb));
    int as_dim = orbitals.size();
    double truncation_tol = 1e-6;
    double coulomb_lo = 0.001;
    double coulomb_eps = 1e-6;

    auto t1 = std::chrono::high_resolution_clock::now();
    std::vector<Function<double, NDIM>> orbs_kl;
    for (int k = 0; k < as_dim; k++) {
        std::vector<Function<double, NDIM>> kl = orbitals[k] * orbitals;
        orbs_kl.insert(std::end(orbs_kl), std::begin(kl), std::end(kl));
    }
    orbs_kl = truncate(orbs_kl, truncation_tol);

    auto t2 = std::chrono::high_resolution_clock::now();
    auto coul_op_parallel =
        std::shared_ptr<SeparatedConvolution<double, NDIM>>(CoulombOperatorNDPtr<NDIM>(*(madness_process.world), coulomb_lo, coulomb_eps));
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
std::vector<SavedFct<NDIM>>  Integrals<NDIM>::normalize(std::vector<SavedFct<NDIM>> all_orbs){

    std::vector<Function<double, NDIM>> basis;
    for (SavedFct<NDIM> orb : all_orbs)
        basis.push_back(madness_process.loadfct(orb));

    madness::normalize(*(madness_process.world), basis);

    std::vector<SavedFct<NDIM>> result;
    for (auto x : basis)
        result.push_back(SavedFct<NDIM>(x));
    for (size_t k = 0; k < result.size(); k++)
        result[k].info = all_orbs[k].info;
    for (size_t k = 0; k < result.size(); k++)
        result[k].type = all_orbs[k].type;

    return result;

}

template <std::size_t NDIM>
std::vector<SavedFct<NDIM>> Integrals<NDIM>::orthonormalize(std::vector<SavedFct<NDIM>> all_orbs, const std::string method,
                                                double rr_thresh) {
    std::vector<Function<double, NDIM>> basis;
    for (SavedFct<NDIM> orb : all_orbs)
        basis.push_back(madness_process.loadfct(orb));
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

    std::vector<SavedFct<NDIM>> result;
    for (auto x : out_basis)
        result.push_back(SavedFct<NDIM>(x));
    for (size_t k = 0; k < result.size(); k++)
        result[k].info = all_orbs[k].info;
    for (size_t k = 0; k < result.size(); k++)
        result[k].type = all_orbs[k].type;

    return result;
}

template <std::size_t NDIM>
std::vector<SavedFct<NDIM>> Integrals<NDIM>::project_out(std::vector<SavedFct<NDIM>> kernel, std::vector<SavedFct<NDIM>> target) {
    std::vector<Function<double, NDIM>> x;
    for (SavedFct<NDIM> orb : kernel)
        x.push_back(madness_process.loadfct(orb));
    std::vector<Function<double, NDIM>> y;
    for (SavedFct<NDIM> orb : target)
        y.push_back(madness_process.loadfct(orb));

    auto Q = madness::QProjector<double, NDIM>(x);
    auto z = Q(y);
    madness::normalize(*(madness_process.world), z);
    std::vector<SavedFct<NDIM>> result;
    for (size_t k = 0; k < target.size(); k++)
        result.push_back(SavedFct<NDIM>(z[k], target[k].type, target[k].info));
    return result;
}

template <std::size_t NDIM>
std::vector<SavedFct<NDIM>> Integrals<NDIM>::project_on(std::vector<SavedFct<NDIM>> kernel, std::vector<SavedFct<NDIM>> target) {
    std::vector<Function<double, NDIM>> x;
    for (SavedFct<NDIM> orb : kernel)
        x.push_back(madness_process.loadfct(orb));
    std::vector<Function<double, NDIM>> y;
    for (SavedFct<NDIM> orb : target)
        y.push_back(madness_process.loadfct(orb));

    auto P = madness::Projector<double, NDIM>(x);
    auto z = P(y);
    madness::normalize(*(madness_process.world), z);
    std::vector<SavedFct<NDIM>> result;
    for (size_t k = 0; k < target.size(); k++)
        result.push_back(SavedFct<NDIM>(z[k], target[k].type, target[k].info));
    return result;
}

template class Integrals<2>;
template class Integrals<3>;