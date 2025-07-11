#include "integrals.hpp"

using namespace madness;


Integrals::Integrals(double L, long k, double thresh, int initial_level, int truncate_mode, bool refine, int n_threads) : MadnessProcess(L, k, thresh, initial_level, truncate_mode, refine, n_threads) {}

nb::ndarray<nb::numpy, double, nb::ndim<2> > Integrals::compute_potential_integrals(std::vector<SavedFct> all_orbs, SavedFct potential){
    std::vector<real_function_3d> orbitals;
    real_function_3d V = loadfct(potential);
    for(SavedFct orb : all_orbs) orbitals.push_back(loadfct(orb));
    int as_dim = orbitals.size();
    potential_integrals= madness::matrix_inner(*world, orbitals, V*orbitals);
    nb::ndarray<nb::numpy, double, nb::ndim<2> > numpy_array(potential_integrals.ptr(), {orbitals.size(), orbitals.size()});
    return numpy_array;
}

nb::ndarray<nb::numpy, double, nb::ndim<2> > Integrals::compute_overlap_integrals(std::vector<SavedFct> all_orbs, std::vector<SavedFct> other){
    std::vector<real_function_3d> orbitals1;
    for(SavedFct orb : all_orbs) orbitals1.push_back(loadfct(orb));
    std::vector<real_function_3d> orbitals2;
    for(SavedFct orb : other) orbitals2.push_back(loadfct(orb));
    overlap_integrals= madness::matrix_inner(*world, orbitals1, orbitals2);
    nb::ndarray<nb::numpy, double, nb::ndim<2> > numpy_array(overlap_integrals.ptr(), {orbitals1.size(), orbitals2.size()});
    return numpy_array;
}
nb::ndarray<nb::numpy, double, nb::ndim<2> > Integrals::compute_kinetic_integrals(std::vector<SavedFct> all_orbs){
    std::vector<real_function_3d> orbitals;
    for(SavedFct orb : all_orbs) orbitals.push_back(loadfct(orb));
    int as_dim = orbitals.size();

    kinetic_integrals = madness::Tensor<double>(as_dim, as_dim);
    for(int k = 0; k < as_dim; k++)
    {
        for(int l = 0; l < as_dim; l++)
        {
            //Kinetic
            for (int axis=0; axis<3; axis++) {
                real_derivative_3d D = free_space_derivative<double,3>(*world, axis);
                real_function_3d d_orb_k = D(orbitals[k]);
                real_function_3d d_orb_l = D(orbitals[l]);
                kinetic_integrals(k, l) += 0.5 * inner(d_orb_k,d_orb_l);
            }
        }
    }

    nb::ndarray<nb::numpy, double, nb::ndim<2> > numpy_array(kinetic_integrals.ptr(), {orbitals.size(), orbitals.size()});
    return numpy_array;
}

// todo: allow 4 sets of orbitals A,B,C,D as input for <AB|g|CD> all indices
nb::ndarray<nb::numpy, double, nb::ndim<4> > Integrals::compute_two_body_integrals(std::vector<SavedFct> all_orbs)
{
    std::vector<real_function_3d> orbitals;
    for(SavedFct orb : all_orbs) orbitals.push_back(loadfct(orb));
    int as_dim = orbitals.size();
    double truncation_tol = 1e-6;
    double coulomb_lo = 0.001;
    double coulomb_eps = 1e-6;

    auto t1 = std::chrono::high_resolution_clock::now();
    std::vector<real_function_3d> orbs_kl;
    for(int k = 0; k < as_dim; k++)
    {
        std::vector<real_function_3d> kl = orbitals[k] * orbitals;
        orbs_kl.insert(std::end(orbs_kl), std::begin(kl), std::end(kl));
    }
    orbs_kl = truncate(orbs_kl, truncation_tol);

    auto t2 = std::chrono::high_resolution_clock::now();
    real_convolution_3d coul_op = CoulombOperator(*world, coulomb_lo, coulomb_eps);
    auto coul_op_parallel = std::shared_ptr<real_convolution_3d>(CoulombOperatorPtr(*world, coulomb_lo, coulomb_eps));
    std::vector<real_function_3d> coul_orbs_mn = apply(*world, *coul_op_parallel, orbs_kl);
    coul_orbs_mn = truncate(coul_orbs_mn, truncation_tol);

    auto t3 = std::chrono::high_resolution_clock::now();
    two_body_integrals = madness::Tensor<double>(as_dim, as_dim, as_dim, as_dim);
    madness::Tensor<double> Inner_prods = matrix_inner(*world, orbs_kl, coul_orbs_mn, false);
    std::vector<double> flat;
    for(int k = 0; k < as_dim; k++)
    {
        for(int l = 0; l < as_dim; l++)
        {
            for(int m = 0; m < as_dim; m++)
            {
                for(int n = 0; n < as_dim; n++)
                {
                    auto tmp = Inner_prods(k*as_dim + l, m*as_dim + n);
                    two_body_integrals(k, m, l, n) = tmp;
                    flat.push_back(tmp);
                }
            }
        }
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    // todo: save timings from Fabian

    nb::ndarray<nb::numpy, double, nb::ndim<4> > numpy_array(two_body_integrals.ptr(), {orbitals.size(), orbitals.size(), orbitals.size(), orbitals.size()});
    return numpy_array;
}

std::vector<SavedFct> Integrals::orthonormalize(std::vector<SavedFct> all_orbs, const std::string method, double rr_thresh){
    std::vector<real_function_3d> basis;
    for(SavedFct orb : all_orbs) basis.push_back(loadfct(orb));
	// compute overlap, to be passed in orthonormalization routines and potentially printed
	auto S = madness::matrix_inner(*world, basis, basis, true);

	auto out_basis = basis;
	if (method == "cholesky"){
		out_basis = madness::orthonormalize_cd(basis, S);
	}else if(method == "symmetric"){
		out_basis = madness::orthonormalize_symmetric(basis, S);
	}else if(method == "canonical") {
		out_basis = madness::orthonormalize_canonical(basis, S, rr_thresh);
	}else if(method == "rr_cholesky") {
		out_basis = madness::orthonormalize_rrcd(basis, S, rr_thresh);
	}else{
		MADNESS_EXCEPTION("unknown orthonormalization method", 1);
	}

    std::vector<SavedFct> result;
    for(auto x : out_basis) result.push_back(SavedFct(x));
    for(size_t k=0; k<result.size(); k++) result[k].info = all_orbs[k].info;
    for(size_t k=0; k<result.size(); k++) result[k].type = all_orbs[k].type;

	return result;
}

std::vector<SavedFct> Integrals::project_out(std::vector<SavedFct> kernel, std::vector<SavedFct> target){
    std::vector<real_function_3d> x;
    for(SavedFct orb : kernel) x.push_back(loadfct(orb));
    std::vector<real_function_3d> y;
    for(SavedFct orb : target) y.push_back(loadfct(orb));

    auto Q = madness::QProjector<double,3>(x);
    auto z = Q(y);
    madness::normalize(*world, z);
    std::vector<SavedFct> result;
    for(size_t k=0; k<target.size(); k++) result.push_back(SavedFct(z[k], target[k].type, target[k].info));
    return result;
}

std::vector<SavedFct> Integrals::project_on(std::vector<SavedFct> kernel, std::vector<SavedFct> target){
    std::vector<real_function_3d> x;
    for(SavedFct orb : kernel) x.push_back(loadfct(orb));
    std::vector<real_function_3d> y;
    for(SavedFct orb : target) y.push_back(loadfct(orb));

    auto P = madness::Projector<double,3>(x);
    auto z = P(y);
    madness::normalize(*world, z);
    std::vector<SavedFct> result;
    for(size_t k=0; k<target.size(); k++) result.push_back(SavedFct(z[k], target[k].type, target[k].info));
    return result;
}


