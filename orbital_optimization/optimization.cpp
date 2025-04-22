#include "optimization.hpp"

using namespace madness;

const std::string path_to_plots="/Users/timo/workspace/MRA_nanobind/MRA-OrbitalOptimization/Examples/Full_run/temp_files/"; //change this file path according to your folder structure
template<int NDIM>
void plot(const char* filename, const Function<double,NDIM>& f, double L) {
    Vector<double,NDIM> lo(0.0), hi(0.0);
    lo[NDIM-1] = -L; hi[NDIM-1] = L;
    std::string full_path=path_to_plots+filename;
    plot_line(full_path.c_str(),2001,lo,hi,f);
}

Optimization::Optimization(int argc, char** argv, double L, long k, double thresh)
{
    int arg = 0;
    char **a = new char*[0]();
    initialize(arg, a);
    world = new World(SafeMPI::COMM_WORLD);
    this->atoms = atoms;

    startup(*world,argc,argv);
    std::cout.precision(6);

    FunctionDefaults<3>::set_k(k);
    FunctionDefaults<3>::set_thresh(thresh);
    FunctionDefaults<3>::set_refine(true);
    FunctionDefaults<3>::set_initial_level(5);
    FunctionDefaults<3>::set_truncate_mode(1);
    FunctionDefaults<3>::set_cubic_cell(-L, L);
}

Optimization::~Optimization()
{

}

void Optimization::CreateNuclearPotentialAndRepulsion(std::string GeometryFile)
{
    auto molecule = madness::Molecule();
    molecule.read_file(GeometryFile);
    Vnuc = new Nuclear<double,3>(*world, molecule);
    nuclear_repulsion_energy = molecule.nuclear_repulsion_energy();
}

void Optimization::ReadInitialOrbitals(std::vector<std::string> frozen_occ_orbs_files, std::vector<std::string> active_orbs_files, std::vector<std::string> frozen_virt_orb_files)
{
    auto start_time = std::chrono::high_resolution_clock::now();

    this->frozen_occ_orbs_files = frozen_occ_orbs_files;
    this->active_orbs_files = active_orbs_files;
    this->frozen_virt_orb_files = frozen_virt_orb_files;

    for(int i = 0; i < frozen_occ_orbs_files.size(); i++)
    {
        real_function_3d orb = real_factory_3d(*world);
        load(orb, frozen_occ_orbs_files[i]);
        frozen_occ_orbs.push_back(orb);
    }

    for(int i = 0; i < active_orbs_files.size(); i++)
    {
        real_function_3d orb = real_factory_3d(*world);
        load(orb, active_orbs_files[i]);
        plot<3>(("orb_"+std::to_string(i)+".dat").c_str(),orb, 50.0);
        active_orbs.push_back(orb);
    }

    for(int i = 0; i < frozen_virt_orb_files.size(); i++)
    {
        real_function_3d orb = real_factory_3d(*world);
        load(orb, frozen_virt_orb_files[i]);
        frozen_virt_orb.push_back(orb);
    }
    core_dim = frozen_occ_orbs.size();
    as_dim = active_orbs.size();
    froz_virt_dim = frozen_virt_orb.size();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "ReadOrbitals took " << duration.count() << " seconds" << std::endl;
}

void Optimization::ReadRDMFilesAndRotateOrbitals(std::string one_rdm_file, std::string two_rdm_file)
{
    auto start_time = std::chrono::high_resolution_clock::now();

    //****************************************
    // Read active space RDMs
    //****************************************
    as_one_rdm = Eigen::MatrixXd::Zero(as_dim, as_dim);
    as_two_rdm = Eigen::Tensor<double, 4>(as_dim, as_dim, as_dim, as_dim);

    std::vector<double> one_rdm_data = npy::read_npy<double>(one_rdm_file).data;
    std::vector<double> two_rdm_data = npy::read_npy<double>(two_rdm_file).data;

    int x = 0;
    for(int i = 0; i < as_dim; i++)
    {
        for(int j = 0; j < as_dim; j++)
        {
            as_one_rdm(i,j) = one_rdm_data[x];
            x++;
        }
    }

    x = 0;
    for(int i = 0; i < as_dim; i++)
    {
        for(int j = 0; j < as_dim; j++)
        {
            for(int k = 0; k < as_dim; k++)
            {
                for(int l = 0; l < as_dim; l++)
                {
                    as_two_rdm(i,j,k,l) = two_rdm_data[x];
                    x++;
                }
            }
        }
    }

    //****************************************
    // Rotate active Space Orbitals
    //****************************************
    {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
        es.compute(as_one_rdm);
        ActiveSpaceRotationMatrix = es.eigenvectors().rowwise().reverse();
        TransformMatrix(&as_one_rdm, ActiveSpaceRotationMatrix);
        TransformTensor(&as_two_rdm, ActiveSpaceRotationMatrix);
        madness::Tensor<double> T(as_dim, as_dim);
        for (int i = 0; i < as_dim; i++) {
            for (int j = 0; j < as_dim; j++) {
                T(i,j) = ActiveSpaceRotationMatrix(i,j);
            }
        }
        active_orbs = transform(*world, active_orbs, T);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "ReadRDMFiles took " << duration.count() << " seconds" << std::endl;
}

void Optimization::TransformMatrix(Eigen::MatrixXd* ObjectMatrix, Eigen::MatrixXd TransformationMatrix)
{
    *ObjectMatrix = TransformationMatrix.transpose() * *ObjectMatrix * TransformationMatrix;
}

void Optimization::TransformTensor(Eigen::Tensor<double, 4>* ObjectTensor, Eigen::MatrixXd TransformationMatrix)
{
    int n = TransformationMatrix.rows();
    Eigen::Tensor<double, 4> temp1(n, n, n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k2 = 0; k2 < n; k2++) {
                for (int l = 0; l < n; l++) {
                    double k_value = 0;
                    for (int k = 0; k < n; k++) {
                        k_value += TransformationMatrix(k, k2) * ObjectTensor->coeff(i, j, k, l);
                    }
                    temp1(i, j, k2, l) = k_value;
                }
            }
        }
    }

    Eigen::Tensor<double, 4> temp2(n, n, n, n);
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

    Eigen::Tensor<double, 4> temp3(n, n, n, n);
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

    Eigen::Tensor<double, 4> temp4(n, n, n, n);
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
    *ObjectTensor = temp4;
}

void Optimization::CalculateAllIntegrals()
{
    auto start_time = std::chrono::high_resolution_clock::now();

    // Initializing the Coulomb operator
    real_convolution_3d coul_op = CoulombOperator(*world, coulomb_lo, coulomb_eps);
    auto coul_op_parallel = std::shared_ptr<real_convolution_3d>(CoulombOperatorPtr(*world, coulomb_lo, coulomb_eps));

    // Multiplication of AS orbital pairs and their Coulomb element (are needed more often and are therefore stored)
    auto t1 = std::chrono::high_resolution_clock::now();
    orbs_kl.clear();
    for(int k = 0; k < as_dim; k++)
    {
        std::vector<real_function_3d> kl = active_orbs[k] * active_orbs;
        orbs_kl.insert(std::end(orbs_kl), std::begin(kl), std::end(kl));
    }
    orbs_kl = truncate(orbs_kl, truncation_tol);
    coul_orbs_mn = apply(*world, *coul_op_parallel, orbs_kl);
    coul_orbs_mn = truncate(coul_orbs_mn, truncation_tol);
    auto t2 = std::chrono::high_resolution_clock::now();

    // AS-AS one electron integrals
    as_integrals_one_body = Eigen::MatrixXd::Zero(as_dim, as_dim);
    for(int k = 0; k < as_dim; k++)
    {
        for(int l = 0; l < as_dim; l++)
        {
            //Kinetic
            for (int axis=0; axis<3; axis++) {
                real_derivative_3d D = free_space_derivative<double,3>(*world, axis);
                real_function_3d d_orb_k = D(active_orbs[k]);
                real_function_3d d_orb_l = D(active_orbs[l]);
                as_integrals_one_body(k, l) += 0.5 * inner(d_orb_k,d_orb_l);
            }
            //Nuclear
            real_function_3d Vnuc_orb_l = (*Vnuc)(active_orbs[l]);
            as_integrals_one_body(k, l) += inner(active_orbs[k], Vnuc_orb_l);
        }
    }
    auto t3 = std::chrono::high_resolution_clock::now();

    // AS two electron integrals
    as_integrals_two_body = Eigen::Tensor<double, 4>(as_dim, as_dim, as_dim, as_dim);
    madness::Tensor<double> Inner_prods = matrix_inner(*world, orbs_kl, coul_orbs_mn, false);
    for(int k = 0; k < as_dim; k++)
    {
        for(int l = 0; l < as_dim; l++)
        {
            for(int m = 0; m < as_dim; m++)
            {
                for(int n = 0; n < as_dim; n++)
                {
                    as_integrals_two_body(k, m, l, n) = Inner_prods(k*as_dim + l, m*as_dim + n);
                }
            }
        }
    }
    auto t4 = std::chrono::high_resolution_clock::now();

    // Core-AS one electron integrals
    core_as_integrals_one_body_ak = Eigen::MatrixXd::Zero(core_dim, as_dim);
    for(int a = 0; a < core_dim; a++)
    {
        for(int k = 0; k < as_dim; k++)
        {
            //Kinetic
            for (int axis=0; axis<3; axis++) {
                real_derivative_3d D = free_space_derivative<double,3>(*world, axis);
                real_function_3d d_orb_a = D(frozen_occ_orbs[a]);
                real_function_3d d_orb_k = D(active_orbs[k]);
                core_as_integrals_one_body_ak(a, k) += 0.5 * inner(d_orb_a, d_orb_k);
            }
            //Nuclear
            real_function_3d Vnuc_orb_k = (*Vnuc)(active_orbs[k]);
            core_as_integrals_one_body_ak(a, k) += inner(frozen_occ_orbs[a], Vnuc_orb_k);
        }
    }
    auto t5 = std::chrono::high_resolution_clock::now();

    // Core-AS two electron integrals <ak|al>
    if(core_dim > 0)
    {
        core_as_integrals_two_body_akal = Eigen::Tensor<double, 3>(core_dim, as_dim, as_dim);
        std::vector<real_function_3d> orbs_aa;
        for (int a = 0; a < core_dim; a++)
        {
            orbs_aa.push_back(frozen_occ_orbs[a] * frozen_occ_orbs[a]);
        }
        orbs_aa = truncate(orbs_aa, truncation_tol);
        madness::Tensor<double> Inner_prods_akal = matrix_inner(*world, orbs_aa, coul_orbs_mn, false);
        for (int a = 0; a < core_dim; a++)
        {
            for(int k = 0; k < as_dim; k++)
            {
                for(int l = 0; l < as_dim; l++)
                {
                    core_as_integrals_two_body_akal(a, k, l) = Inner_prods_akal(a, k*as_dim + l);
                }
            }
        }
    }

    // Core-AS two electron integrals <ak|la>, <ak|ln>, <ab|ak> and <ba|ak>
    if(core_dim > 0)
    {
        core_as_integrals_two_body_akla = Eigen::Tensor<double, 3>(core_dim, as_dim, as_dim);
        core_as_integrals_two_body_akln = Eigen::Tensor<double, 4>(core_dim, as_dim, as_dim, as_dim);
        core_as_integrals_two_body_abak = Eigen::Tensor<double, 3>(core_dim, core_dim, as_dim);
        core_as_integrals_two_body_baak = Eigen::Tensor<double, 3>(core_dim, core_dim, as_dim);
        for (int a = 0; a < core_dim; a++) //One core orbital after the other -> Slightly less efficient than all a at the same time, but reduces memory
        {
            std::vector<real_function_3d> orbs_ak = frozen_occ_orbs[a] * active_orbs;
            orbs_ak = truncate(orbs_ak, truncation_tol);
            std::vector<real_function_3d> coul_orbs_ak = apply(*world, *coul_op_parallel, orbs_ak);
            coul_orbs_ak = truncate(coul_orbs_ak, truncation_tol);

            std::vector<real_function_3d> orbs_ka = active_orbs * frozen_occ_orbs[a];
            orbs_ka = truncate(orbs_ka, truncation_tol);
            
            // <ak|la> = <ka|al>
            madness::Tensor<double> Inner_prods_akla = matrix_inner(*world, orbs_ka, coul_orbs_ak, false);
            for(int k = 0; k < as_dim; k++)
            {
                for(int l = 0; l < as_dim; l++)
                {
                    core_as_integrals_two_body_akla(a, k, l) = Inner_prods_akla(l, k);
                }
            }

            // <ak|ln>
            madness::Tensor<double> Inner_prods_akln = matrix_inner(*world, orbs_ak, coul_orbs_mn, false);
            for(int k = 0; k < as_dim; k++)
            {
                for(int l = 0; l < as_dim; l++)
                {
                    for(int n = 0; n < as_dim; n++)
                    {
                        core_as_integrals_two_body_akln(a, l, k, n) = Inner_prods_akln(l, k * as_dim + n);
                    }
                }
            }

            // <ab|ak>
            for (int b = 0; b < core_dim; b++)
            {
                madness::Tensor<double> Inner_prods_abak = matrix_inner(*world, std::vector<real_function_3d>{ frozen_occ_orbs[b] * frozen_occ_orbs[b] }, coul_orbs_ak, false); //Change of indicies
                for(int k = 0; k < as_dim; k++)
                {
                    core_as_integrals_two_body_abak(b,a,k) = Inner_prods_abak(0, k);
                }
            }

            // <ba|ak>
            for (int b = 0; b < core_dim; b++)
            {
                madness::Tensor<double> Inner_prods_baak = matrix_inner(*world, std::vector<real_function_3d>{ frozen_occ_orbs[b] * frozen_occ_orbs[a] }, coul_orbs_ak, false);
                for(int k = 0; k < as_dim; k++)
                {
                    core_as_integrals_two_body_baak(a,b,k) = Inner_prods_baak(0, k);
                }
            }
        }
    }
    auto t6 = std::chrono::high_resolution_clock::now();


    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Integral timings:" << std::endl;
    std::cout << "Preparation AS pairs: " << std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count() << " seconds" << std::endl;
    std::cout << "AS-AS one-electron integrals: " << std::chrono::duration_cast<std::chrono::seconds>(t3 - t2).count() << " seconds" << std::endl;
    std::cout << "AS-AS two-electron integrals: " << std::chrono::duration_cast<std::chrono::seconds>(t4 - t3).count() << " seconds" << std::endl;
    std::cout << "Core-AS one-electron integrals: " << std::chrono::duration_cast<std::chrono::seconds>(t5 - t4).count() << " seconds" << std::endl;
    std::cout << "Core-AS two-electron integrals: " << std::chrono::duration_cast<std::chrono::seconds>(t6 - t5).count() << " seconds" << std::endl;
    std::cout << "Full function: " << std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count() << " seconds" << std::endl;
}

void Optimization::CalculateCoreEnergy()
{
    auto start_time = std::chrono::high_resolution_clock::now();

    double nocc = 2; //Spatial orbitals = 2, Spin orbitals = 1
    double core_kinetic_energy = 0;
    double core_nuclear_attraction_energy = 0;
    double core_two_electron_energy = 0;

    if(frozen_occ_orbs.size() > 0)
    {
        //1e Part
        for(int i = 0; i < frozen_occ_orbs.size(); i++)
        {
            //Kinetic
            for (int axis=0; axis<3; axis++) {
                real_derivative_3d D = free_space_derivative<double,3>(*world, axis);
                real_function_3d d_orb = D(frozen_occ_orbs[i]);
                core_kinetic_energy += 0.5 * inner(d_orb, d_orb);
            }
            //Nuclear
            real_function_3d Vnuc_orb = (*Vnuc)(frozen_occ_orbs[i]);
            core_nuclear_attraction_energy += inner(frozen_occ_orbs[i], Vnuc_orb);
        }
        core_kinetic_energy = nocc * core_kinetic_energy;
        core_nuclear_attraction_energy = nocc * core_nuclear_attraction_energy;

        //2e Part

        real_convolution_3d coul_op = CoulombOperator(*world, coulomb_lo, coulomb_eps);
        auto coul_op_parallel = std::shared_ptr<real_convolution_3d>(CoulombOperatorPtr(*world, coulomb_lo, coulomb_eps));

        // <ab|ab>
        {
            std::vector<real_function_3d> orbs_aa;
            for (int a = 0; a < core_dim; a++)
            {
                orbs_aa.push_back(frozen_occ_orbs[a] * frozen_occ_orbs[a]);
            }
            orbs_aa = truncate(orbs_aa, truncation_tol);
            std::vector<real_function_3d> coul_orbs_aa = apply(*world, *coul_op_parallel, orbs_aa);
            coul_orbs_aa = truncate(coul_orbs_aa, truncation_tol);
            for (int a = 0; a < core_dim; a++)
            {
                madness::Tensor<double> Inner_prods_abab = matrix_inner(*world, std::vector<real_function_3d>{ orbs_aa[a] }, coul_orbs_aa, false);
                for (int b = 0; b < core_dim; b++)
                {
                    core_two_electron_energy += 2 * Inner_prods_abab(0, b);
                }
            }
        }

        for (int a = 0; a < core_dim; a++) //One core orbital after the other -> Slightly less efficient than all a at the same time, but reduces memory
        {
            std::vector<real_function_3d> orbs_ab = frozen_occ_orbs[a] * frozen_occ_orbs;
            orbs_ab = truncate(orbs_ab, truncation_tol);
            std::vector<real_function_3d> coul_orbs_ab = apply(*world, *coul_op_parallel, orbs_ab);
            coul_orbs_ab = truncate(coul_orbs_ab, truncation_tol);
            for (int b = 0; b < core_dim; b++)
            {
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
}

void Optimization::CalculateEnergies()
{
    auto start_time = std::chrono::high_resolution_clock::now();

    //Active Space Part

    double as_one_electron_energy = 0.0;

    //One Particle Part
    for(int k = 0; k < as_dim; k++)
    {
        for(int l = 0; l < as_dim; l++)
        {
            as_one_electron_energy += as_one_rdm(k, l) * as_integrals_one_body(k, l);
        }
    }

    //Two Particle Part
    double as_two_electron_energy = 0.0;
    for(int k = 0; k < as_dim; k++)
    {
        for(int l = 0; l < as_dim; l++)
        {
            for(int m = 0; m < as_dim; m++)
            {
                for(int n = 0; n < as_dim; n++)
                {
                    as_two_electron_energy += as_two_rdm(k, l, m, n) * 0.5 * as_integrals_two_body(k, l, m, n); 
                }
            }
        }
    }

    //Core Part
    double as_core_energy = 0.0;
    if(core_dim > 0)
    {
        for(int a = 0; a < core_dim; a++)
        {
            for(int k = 0; k < as_dim; k++)
            {
                for(int l = 0; l < as_dim; l++)
                {
                    as_core_energy += 2 * as_one_rdm(k, l) * core_as_integrals_two_body_akal(a,k,l); 
                    as_core_energy -= as_one_rdm(k, l) * core_as_integrals_two_body_akla(a,k,l); 
                }
            }
        }
        as_core_energy = 0.5 * nocc * as_core_energy;
    }

    
    //Print results
    double total_energy = as_one_electron_energy + as_two_electron_energy + as_core_energy + nuclear_repulsion_energy +  core_total_energy;

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

void Optimization::CalculateLagrangeMultiplier()
{
    auto start_time = std::chrono::high_resolution_clock::now();

    LagrangeMultiplier_AS_AS = Eigen::MatrixXd::Zero(as_dim, as_dim);

    for(int z = 0; z < as_dim; z++)
    {
        for(int i = 0; i < as_dim; i++)
        {
            LagrangeMultiplier_AS_AS(z, i) = CalculateLagrangeMultiplierElement_AS_AS(z, i);
        }
    }

    if(core_dim > 0)
    {
        LagrangeMultiplier_AS_Core = Eigen::MatrixXd::Zero(core_dim, as_dim);
        for(int z = 0; z < core_dim; z++)
        {
            for(int i = 0; i < as_dim; i++)
            {
                LagrangeMultiplier_AS_Core(z, i) = CalculateLagrangeMultiplierElement_AS_Core(z, i);
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "CalculateLagrangeMultiplier took " << duration.count() << " seconds" << std::endl;
}

double Optimization::CalculateLagrangeMultiplierElement_AS_AS(int z, int i)
{    
    double element = as_one_rdm(i,i) * as_integrals_one_body(z,i);
    for(int l = 0; l < as_dim; l++)
    {
        for(int n = 0; n < as_dim; n++)
        {
            for(int k = 0; k < as_dim; k++)
            {
                element += as_two_rdm(k, l, i, n) * as_integrals_two_body(z, l, k, n);
            }
        }
    }

    for(int k = 0; k < as_dim; k++)
    {
        for(int a = 0; a < core_dim; a++)
        {
            element += 0.5 * nocc * 2 * as_one_rdm(k, i) * core_as_integrals_two_body_akal(a,z,k);
            element -= 0.5 * nocc * as_one_rdm(k, i) * core_as_integrals_two_body_akla(a,k,z);
        }
    }
    return element;


}

double Optimization::CalculateLagrangeMultiplierElement_AS_Core(int z, int i)
{
    double element = as_one_rdm(i,i) * core_as_integrals_one_body_ak(z,i);

    for(int l = 0; l < as_dim; l++)
    {
        for(int n = 0; n < as_dim; n++)
        {
            for(int k = 0; k < as_dim; k++)
            {
                element += as_two_rdm(k, l, i, n) * core_as_integrals_two_body_akln(z, k, l, n);
            }
        }
    }

    for(int k = 0; k < as_dim; k++)
    {
        for(int a = 0; a < core_dim; a++)
        {           
            element += 0.5 * nocc * 2 * as_one_rdm(k, i) * core_as_integrals_two_body_abak(a,z,k);
            element -= 0.5 * nocc * as_one_rdm(k, i) * core_as_integrals_two_body_baak(a,z,k);
        }
    }
    return element;
}

void Optimization::OptimizeOrbitals(double optimization_thresh, double NO_occupation_thresh)
{

    bool converged = false;
    int iterstep = 0;
    while(!converged)
    {
        iterstep++;
        std::cout << "---------------------------------------------------" << std::endl;
        std::cout << "Start iteration step: " << iterstep << std::endl;

        //Update LagrangeMultiplier
        CalculateLagrangeMultiplier();

        //Update orbitals
        auto start_orb_update_time = std::chrono::high_resolution_clock::now();
        highest_error = 0;

        std::vector<int> as_orbital_indicies_for_update;
        for (int idx = 0; idx < active_orbs.size(); idx++)
        {
            if(abs(as_one_rdm(idx, idx)) >= NO_occupation_thresh)
            {
                as_orbital_indicies_for_update.push_back(idx);
            }
            else
            {
                std::cout << "Skip refinement of active space orbital " << idx << ", since the occupation is less than NO_occupation_thresh (" << NO_occupation_thresh << ")" << std::endl;
            }
        }

        std::vector<real_function_3d> AllActiveOrbitalUpdates = GetAllActiveOrbitalUpdates(as_orbital_indicies_for_update);

        for (int idx = 0; idx < as_orbital_indicies_for_update.size(); idx++)
        {
            int actIdx = as_orbital_indicies_for_update[idx];
            active_orbs[actIdx] = active_orbs[actIdx] - AllActiveOrbitalUpdates[idx];
        }

        auto end_orb_update_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_orb_update_time - start_orb_update_time);
        std::cout << "UpdateOrbitals took " << duration.count() << " seconds" << std::endl;

        //Orthonormalize orbitals
        active_orbs = orthonormalize_symmetric(active_orbs);
        //orbitals = orthonormalize_cd(orbitals);
        active_orbs = truncate(active_orbs, truncation_tol);

        //Check convergence
        std::cout << "Highest error: " << highest_error << std::endl;
        if(highest_error < optimization_thresh){converged = true;}

        std::cout << "Update Integrals" << std::endl;
        //Update integrals for new orbitals
        CalculateAllIntegrals();

        //Calculate new energy
        CalculateEnergies();
    }
}

std::vector<real_function_3d> Optimization::GetAllActiveOrbitalUpdates(std::vector<int> orbital_indicies_for_update)
{
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<real_function_3d> AllOrbitalUpdates;
    
    auto t1 = std::chrono::high_resolution_clock::now();
    //Calculate rdm_ii_inv values
    std::vector<double> rdm_ii_inv;
    for (int idx = 0; idx < orbital_indicies_for_update.size(); idx++)
    {
        int i = orbital_indicies_for_update[idx];
        rdm_ii_inv.push_back(1 / as_one_rdm(i,i));
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    //1e Part
    for (int idx = 0; idx < orbital_indicies_for_update.size(); idx++)
    {
        int i = orbital_indicies_for_update[idx];

        real_convolution_3d coul_op = CoulombOperator(*world, coulomb_lo, coulomb_eps);
        real_function_3d rhs = (*Vnuc)(active_orbs[i]);
        for(int k = 0; k < core_dim; k++)
        {
            rhs -= rdm_ii_inv[idx] * LagrangeMultiplier_AS_Core(k, i) * frozen_occ_orbs[k];
        }
        for(int k = 0; k < as_dim; k++)
        {
            if(k != i)
            {
                rhs -= rdm_ii_inv[idx] * LagrangeMultiplier_AS_AS(k, i) * active_orbs[k];
            }
        }
        AllOrbitalUpdates.push_back(rhs);
    }
    AllOrbitalUpdates = truncate(AllOrbitalUpdates, truncation_tol);
    auto t3 = std::chrono::high_resolution_clock::now();

    // AS Part
    for(int k = 0; k < as_dim; k++)
    {
        std::vector<real_function_3d> lnk = coul_orbs_mn * active_orbs[k];
        lnk = truncate(lnk, truncation_tol);
        for(int idx = 0; idx < orbital_indicies_for_update.size(); idx++)
        {
            int i = orbital_indicies_for_update[idx];
            std::vector<real_function_3d> lnk_copy = copy(*world, lnk, false);
            for(int l = 0; l < as_dim; l++)
            {
                for(int n = 0; n < as_dim; n++)
                {
                    lnk_copy[l*as_dim + n] *= as_two_rdm(k, l, i, n) * rdm_ii_inv[idx];
                }
            }
            AllOrbitalUpdates[idx] += sum(*world, lnk_copy);
        }
    }
    auto t4 = std::chrono::high_resolution_clock::now();

    // Core - AS interaction
    auto coul_op_parallel = std::shared_ptr<real_convolution_3d>(CoulombOperatorPtr(*world, 0.001, 1e-6));

    if(core_dim > 0)
    {
        //Part 1
        std::vector<real_function_3d> orbs_aa;
        for (int a = 0; a < core_dim; a++)
        {
            orbs_aa.push_back(frozen_occ_orbs[a] * frozen_occ_orbs[a]);
        }
        orbs_aa = truncate(orbs_aa, truncation_tol);
        std::vector<real_function_3d> coul_orbs_aa = apply(*world, *coul_op_parallel, orbs_aa);
        coul_orbs_aa = truncate(coul_orbs_aa, truncation_tol);

        for(int k = 0; k < as_dim; k++)
        {
            std::vector<real_function_3d> aak = coul_orbs_aa * active_orbs[k];
            aak = truncate(aak, truncation_tol);
            for(int idx = 0; idx < orbital_indicies_for_update.size(); idx++)
            {
                int i = orbital_indicies_for_update[idx];
                std::vector<real_function_3d> aak_copy = copy(*world, aak, false);
                for(int a = 0; a < core_dim; a++)
                {
                    aak_copy[a] *= 0.5 * nocc * 2 * as_one_rdm(k, i) * rdm_ii_inv[idx];
                    //AllOrbitalUpdates[idx] += 0.5 * 2 * rdm_ii_inv[idx] * as_one_rdm(k, i) * 2 * coul_orbs_aa[a] * active_orbs[k];
                }
                AllOrbitalUpdates[idx] += sum(*world, aak_copy);
            }
        }
    }
    auto t5 = std::chrono::high_resolution_clock::now();

    //Part 2
    if(core_dim > 0)
    {
        for(int a = 0; a < core_dim; a++)
        {
            std::vector<real_function_3d> orbs_ak = frozen_occ_orbs[a] * active_orbs;
            orbs_ak = truncate(orbs_ak, truncation_tol);
            std::vector<real_function_3d> coul_orbs_ak = apply(*world, *coul_op_parallel, orbs_ak);
            coul_orbs_ak = truncate(coul_orbs_ak, truncation_tol);

            std::vector<real_function_3d> aka = coul_orbs_ak * frozen_occ_orbs[a];
            aka = truncate(aka, truncation_tol);

            for(int idx = 0; idx < orbital_indicies_for_update.size(); idx++)
            {
                int i = orbital_indicies_for_update[idx];
                std::vector<real_function_3d> aka_copy = copy(*world, aka, false);
                for(int k = 0; k < as_dim; k++)
                {
                    aka_copy[k] *= -0.5 * nocc * as_one_rdm(k, i) * rdm_ii_inv[idx];
                    //AllOrbitalUpdates[idx] -= 0.5 * rdm_ii_inv[idx] * as_one_rdm(k, i) * 2 * coul_orbs_ak[k] * frozen_occ_orbs[a];
                }
                AllOrbitalUpdates[idx] += sum(*world, aka_copy);
            }
        }
    }
    auto t6 = std::chrono::high_resolution_clock::now();

    //BSH part
    for (int idx = 0; idx < orbital_indicies_for_update.size(); idx++)
    {
        int i = orbital_indicies_for_update[idx];
        double en = LagrangeMultiplier_AS_AS(i, i) * rdm_ii_inv[idx];
        SeparatedConvolution<double,3> bsh_op = BSHOperator<3>(*world, sqrt(-2*en), BSH_lo, BSH_eps);
        real_function_3d r = active_orbs[i] + 2.0 * bsh_op(AllOrbitalUpdates[idx]); // the residual
        double err = r.norm2();
        std::cout << "Error of active space orbital " << i << ": " << err << std::endl; 
        if(err > highest_error){highest_error = err; }
        AllOrbitalUpdates[idx] = r;
    }
    auto t7 = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Refinement timings:" << std::endl;
    std::cout << "rdm_ii_inv calculation: " << std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count() << " seconds" << std::endl;
    std::cout << "one electron part: " << std::chrono::duration_cast<std::chrono::seconds>(t3 - t2).count() << " seconds" << std::endl;
    std::cout << "AS two electron part: " << std::chrono::duration_cast<std::chrono::seconds>(t4 - t3).count() << " seconds" << std::endl;
    std::cout << "Core-AS two electron part 1: " << std::chrono::duration_cast<std::chrono::seconds>(t5 - t4).count() << " seconds" << std::endl;
    std::cout << "Core-AS two electron part 2: " << std::chrono::duration_cast<std::chrono::seconds>(t6 - t5).count() << " seconds" << std::endl;
    std::cout << "BSH part: " << std::chrono::duration_cast<std::chrono::seconds>(t7 - t6).count() << " seconds" << std::endl;
    std::cout << "Full function: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " seconds" << std::endl;

    return AllOrbitalUpdates;
} 

void Optimization::RotateOrbitalsBackAndUpdateIntegrals()
{
    //Create Full RotationMatrix
    Eigen::MatrixXd RotationMatrixBack = ActiveSpaceRotationMatrix.transpose();

    //Transform RDMs
    TransformMatrix(&as_one_rdm, RotationMatrixBack);
    TransformTensor(&as_two_rdm, RotationMatrixBack);

    //Transform active Orbitals
    madness::Tensor<double> T(as_dim, as_dim);
    for (int i = 0; i < as_dim; i++) {
        for (int j = 0; j < as_dim; j++) {
            T(i,j) = RotationMatrixBack(i,j);
        }
    }
    active_orbs = transform(*world, active_orbs, T);
    CalculateAllIntegrals();
}

void Optimization::SaveOrbitals(std::string OutputPath)
{
    for(int i = 0; i < core_dim; i++)
    {
        std::string base_filename = frozen_occ_orbs_files[i].substr(frozen_occ_orbs_files[i].find_last_of("/\\") + 1);
        save(frozen_occ_orbs[i], OutputPath + "/" + base_filename);
    }
    for(int i = 0; i < as_dim; i++)
    {
        std::string base_filename = active_orbs_files[i].substr(active_orbs_files[i].find_last_of("/\\") + 1);
        save(active_orbs[i], OutputPath + "/" + base_filename);
    }
    for(int i = 0; i < froz_virt_dim; i++)
    {
        std::string base_filename = frozen_virt_orb_files[i].substr(frozen_virt_orb_files[i].find_last_of("/\\") + 1);
        save(frozen_virt_orb[i], OutputPath + "/" + base_filename);
    }
}

void Optimization::SaveEffectiveHamiltonian(std::string OutputPath)
{
    std::vector<double> effective_one_body_integrals_elements;
    std::vector<double> effective_two_body_integrals_elements;

    Eigen::MatrixXd effective_one_body_integrals = as_integrals_one_body;
    for(int k = 0; k < as_dim; k++)
    {
        for(int l = 0; l < as_dim; l++)
        {
            for(int a = 0; a < core_dim; a++)
            {
                effective_one_body_integrals(k, l) += 0.5 * nocc * (2 * core_as_integrals_two_body_akal(a,k,l) - core_as_integrals_two_body_akla(a,k,l));
            }
            effective_one_body_integrals_elements.push_back(effective_one_body_integrals(k,l));
        }
    }
    
    for(int k = 0; k < as_dim; k++)
    {
        for(int l = 0; l < as_dim; l++)
        {
            for(int m = 0; m < as_dim; m++)
            {
                for(int n = 0; n < as_dim; n++)
                {
                    effective_two_body_integrals_elements.push_back(as_integrals_two_body(k,l,m,n));
                }
            }
        }
    }

    std::ofstream c_file;
    c_file.open(OutputPath + "/c.txt");
    c_file << std::setprecision (15) << (core_total_energy + nuclear_repulsion_energy);
    c_file.close();

    std::vector<unsigned long> one_e_ints_shape{(unsigned long)as_dim, (unsigned long)as_dim};
    const npy::npy_data<double> one_e_data{effective_one_body_integrals_elements, one_e_ints_shape, false};
    npy::write_npy(OutputPath + "/htensor.npy", one_e_data);

    std::vector<unsigned long> two_e_ints_shape{(unsigned long)as_dim, (unsigned long)as_dim, (unsigned long)as_dim, (unsigned long)as_dim};
    const npy::npy_data<double> two_e_data{effective_two_body_integrals_elements, two_e_ints_shape, false};
    npy::write_npy(OutputPath + "/gtensor.npy", two_e_data);
}