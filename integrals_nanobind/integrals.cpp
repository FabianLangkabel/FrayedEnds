#include "integrals.hpp"

using namespace madness;


Integrals::Integrals(double L, long k, double thresh)
{
    int arg = 0;
    char **a = new char*[0]();

    world = &initialize(arg, a);
    startup(*world,arg,a);

    std::cout.precision(6);
    FunctionDefaults<3>::set_k(k);
    FunctionDefaults<3>::set_thresh(thresh);
    FunctionDefaults<3>::set_refine(true);
    FunctionDefaults<3>::set_initial_level(5);
    FunctionDefaults<3>::set_truncate_mode(1);
    FunctionDefaults<3>::set_cubic_cell(-L, L);
}

Integrals::~Integrals()
{
    std::cout << "Finalize madness env" << std::endl;
    orbitals.clear();
    world->gop.fence();
    world->gop.fence();
    finalize();
}


void Integrals::ReadOrbitals()
{
    auto start_time = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < 8; i++)
    {
        real_function_3d orb = real_factory_3d(*world);
        load(orb, "/workspaces/MRA-OrbitalOptimization/integrals_run_example/orbitals/mra_orbital_" + std::to_string(i));
        orbitals.push_back(orb);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "ReadOrbitals timing: " << duration.count() << " milliseconds" << std::endl;
}

void Integrals::CalculateIntegrals()
{
    double truncation_tol = 1e-6;
    double coulomb_lo = 0.001;
    double coulomb_eps = 1e-6;

    auto t1 = std::chrono::high_resolution_clock::now();
    std::vector<real_function_3d> orbs_kl;
    for(int k = 0; k < 8; k++)
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
    madness::Tensor<double> Inner_prods = matrix_inner(*world, orbs_kl, coul_orbs_mn, false);

    auto t4 = std::chrono::high_resolution_clock::now();
    double int_sum = 0;
    for(int k = 0; k < 8; k++)
    {
        for(int l = 0; l < 8; l++)
        {
            for(int m = 0; m < 8; m++)
            {
                for(int n = 0; n < 8; n++)
                {
                    int_sum += Inner_prods(k*8 + l, m*8 + n);
                }
            }
        }
    }

    auto t5 = std::chrono::high_resolution_clock::now();
    auto t21 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    auto t32 = std::chrono::duration_cast<std::chrono::seconds>(t3 - t2);
    auto t43 = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3);
    auto t54 = std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4);

    std::cout << "Sum over all ints: " << int_sum << std::endl;
    std::cout << "Orbital multiplication timing: " << t21.count() << " milliseconds" << std::endl;
    std::cout << "Apply coulomb timing: " << t32.count() << " seconds" << std::endl;
    std::cout << "Inner products timing: " << t43.count() << " milliseconds" << std::endl;
    std::cout << "Sum timing: " << t54.count() << " milliseconds" << std::endl;
}
