#include "integrals.hpp"

using namespace madness;


Integrals::Integrals(MadnessProcess& mp): madness_process(mp)
{
    std::cout << "starting\n";
}

nb::ndarray<nb::numpy, double, nb::ndim<2> > Integrals::compute_potential_integrals(std::vector<SavedFct> all_orbs, SavedFct potential){
    std::vector<real_function_3d> orbitals;
    real_function_3d V = madness_process.loadfct(potential);
    for(SavedFct orb : all_orbs) orbitals.push_back(madness_process.loadfct(orb));
    int as_dim = orbitals.size();
    potential_integrals= madness::matrix_inner(*(madness_process.world), orbitals, V*orbitals);
    nb::ndarray<nb::numpy, double, nb::ndim<2> > numpy_array(potential_integrals.ptr(), {orbitals.size(), orbitals.size()});
    return numpy_array;
}

nb::ndarray<nb::numpy, double, nb::ndim<2> > Integrals::compute_overlap_integrals(std::vector<SavedFct> all_orbs){
    std::vector<real_function_3d> orbitals;
    for(SavedFct orb : all_orbs) orbitals.push_back(madness_process.loadfct(orb));
    overlap_integrals= madness::matrix_inner(*(madness_process.world), orbitals, orbitals);
    nb::ndarray<nb::numpy, double, nb::ndim<2> > numpy_array(overlap_integrals.ptr(), {orbitals.size(), orbitals.size()});
    return numpy_array;
}
nb::ndarray<nb::numpy, double, nb::ndim<2> > Integrals::compute_kinetic_integrals(std::vector<SavedFct> all_orbs){
    std::vector<real_function_3d> orbitals;
    for(SavedFct orb : all_orbs) orbitals.push_back(madness_process.loadfct(orb));
    int as_dim = orbitals.size();

    kinetic_integrals = madness::Tensor<double>(as_dim, as_dim);
    for(int k = 0; k < as_dim; k++)
    {
        for(int l = 0; l < as_dim; l++)
        {
            //Kinetic
            for (int axis=0; axis<3; axis++) {
                real_derivative_3d D = free_space_derivative<double,3>(*(madness_process.world), axis);
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
    for(SavedFct orb : all_orbs) orbitals.push_back(madness_process.loadfct(orb));
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
    real_convolution_3d coul_op = CoulombOperator(*(madness_process.world), coulomb_lo, coulomb_eps);
    auto coul_op_parallel = std::shared_ptr<real_convolution_3d>(CoulombOperatorPtr(*(madness_process.world), coulomb_lo, coulomb_eps));
    std::vector<real_function_3d> coul_orbs_mn = apply(*(madness_process.world), *coul_op_parallel, orbs_kl);
    coul_orbs_mn = truncate(coul_orbs_mn, truncation_tol);

    auto t3 = std::chrono::high_resolution_clock::now();
    two_body_integrals = madness::Tensor<double>(as_dim, as_dim, as_dim, as_dim);
    madness::Tensor<double> Inner_prods = matrix_inner(*(madness_process.world), orbs_kl, coul_orbs_mn, false);
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


