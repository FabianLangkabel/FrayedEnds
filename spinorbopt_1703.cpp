#include "spinorb_optimizer.hpp"

using namespace madness;

// KAIN allocator for vectorfunctions
struct KAIN_allocator {
    World &world;
    const int norb;

    /// @param[in]	world	the world
    /// @param[in]	nnorb	the number of functions in a given vector
    KAIN_allocator(World &world, const int nnorb) : world(world), norb(nnorb) {}

    vector_real_function_3d operator()() {
        return zero_functions<double, 3>(world, norb);
    }

    KAIN_allocator operator=(const KAIN_allocator &other) {
        KAIN_allocator tmp(world, other.norb);
        return tmp;
    }
};

SpinorbOpt::SpinorbOpt(int argc, char** argv, double L, long k, double thresh)
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

SpinorbOpt::~SpinorbOpt()
{

}

void SpinorbOpt::CreateNuclearPotentialAndRepulsion(std::string GeometryFile)
{
    auto molecule = madness::Molecule();
    molecule.read_file(GeometryFile);  //Define Molecule
    Vnuc = new Nuclear<double,3>(*world, molecule);
    nuclear_repulsion_energy = molecule.nuclear_repulsion_energy();
    std::cout << "Enuc: " << endl << nuclear_repulsion_energy << endl;
}

void SpinorbOpt::ReadInitialOrbitals(std::vector<Orbital> alpha_orbs, std::vector<Orbital> beta_orbs, int num_alpha_orbs, int num_beta_orbs)
{
    auto start_time = std::chrono::high_resolution_clock::now();

    all_alpha_orbitals = alpha_orbs;
    active_alpha_orbital_indicies.resize(num_alpha_orbs);
    std::fill(active_alpha_orbital_indicies.begin(), active_alpha_orbital_indicies.end(), -1);
    frozen_occupied_alpha_orbital_indicies.clear();

    all_beta_orbitals = beta_orbs;
    active_beta_orbital_indicies.resize(num_beta_orbs);
    std::fill(active_beta_orbital_indicies.begin(), active_beta_orbital_indicies.end(), -1);
    frozen_occupied_beta_orbital_indicies.clear();
    
    for(int i = 0; i < all_alpha_orbitals.size(); i++)
    {
        real_function_3d orb = real_factory_3d(*world);
        load(orb, all_alpha_orbitals[i].orbital_file_name);
        all_alpha_orbitals[i].function = orb;
        if(all_alpha_orbitals[i].orbital_type == "active")
        {
            active_alpha_orbital_indicies[all_alpha_orbitals[i].active_space_index] = i;
        }
        else if(all_alpha_orbitals[i].orbital_type == "frozen_occupied")
        {
            frozen_occupied_alpha_orbital_indicies.push_back(i);
        }
        else if(all_alpha_orbitals[i].orbital_type == "inactive_virtual")
        {
            inactive_virtual_alpha_orbital_indicies.push_back(i);
        }
    }

    for(int j = 0; j < all_beta_orbitals.size(); j++)
    {
        real_function_3d orb = real_factory_3d(*world);
        load(orb, all_beta_orbitals[j].orbital_file_name);
        all_beta_orbitals[j].function = orb;
        if(all_beta_orbitals[j].orbital_type == "active")
        {
            active_beta_orbital_indicies[all_beta_orbitals[j].active_space_index] = j;
        }
        else if(all_alpha_orbitals[j].orbital_type == "frozen_occupied")
        {
            frozen_occupied_beta_orbital_indicies.push_back(j);
        }
        else if(all_beta_orbitals[j].orbital_type == "inactive_virtual")
        {
            inactive_virtual_beta_orbital_indicies.push_back(j);
        }
    }

    //Print frozen occupied indicies
    std::cout << "Frozen occupied alpha orbitals: ";
    for (auto const& FroIdxa : frozen_occupied_alpha_orbital_indicies)
    {
        std::cout << FroIdxa << " ";
    }
    std::cout << std::endl;
    std::cout << "Frozen occupied beta orbitals: ";
    for (auto const& FroIdxb : frozen_occupied_beta_orbital_indicies)
    {
        std::cout << FroIdxb << " ";
    }
    std::cout << std::endl;

    //Print active space indicies
    std::cout << "Active space alpha indicies: ";
    for (auto const& AIdxa : active_alpha_orbital_indicies)
    {
        std::cout << AIdxa << " ";
    }
    std::cout << std::endl;
    std::cout << "Active space beta indicies: ";
    for (auto const& AIdxb : active_beta_orbital_indicies)
    {
        std::cout << AIdxb << " ";
    }
    std::cout << std::endl;

    //Print inactive virtual indicies
    std::cout << "Inactive virtual alpha orbitals: ";
    for (auto const& IIdxa : inactive_virtual_alpha_orbital_indicies)
    {
        std::cout << IIdxa << " ";
    }
    std::cout << std::endl;
    std::cout << "Inactive virtual beta orbitals: ";
    for (auto const& IIdxb : inactive_virtual_beta_orbital_indicies)
    {
        std::cout << IIdxb << " ";
    }
    std::cout << std::endl;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "ReadOrbitals took " << duration.count() << " seconds" << std::endl; 
} 

//void SpinorbOpt::CreateAlphaBetaMatrix()


void SpinorbOpt::CalculateOneElectronIntegrals()
{
    auto start_time = std::chrono::high_resolution_clock::now();
    //real_convolution_3d coul_op = CoulombOperator(*world, 0.001, 1e-6);

    //construct the alpha and beta matrices
    std::vector<real_function_3d> alpha_orbitals;
    for (auto const& A_orb : all_alpha_orbitals)
    {
        alpha_orbitals.push_back(A_orb.function);
    }
    int dim_a = alpha_orbitals.size();

    std::vector<real_function_3d> beta_orbitals;
    for (auto const& B_orb : all_beta_orbitals)
    {
        beta_orbitals.push_back(B_orb.function);
    }
    int dim_b = beta_orbitals.size();

    //Set up the kinetic integrals matrices
    
    integrals_kinetic_alpha.clear();
    integrals_kinetic_alpha.push_back(Eigen::MatrixXd::Zero(dim_a, dim_a)); //x
    integrals_kinetic_alpha.push_back(Eigen::MatrixXd::Zero(dim_a, dim_a)); //y
    integrals_kinetic_alpha.push_back(Eigen::MatrixXd::Zero(dim_a, dim_a)); //z
    integrals_potential_alpha = Eigen::MatrixXd::Zero(dim_a, dim_a);
    //integrals_two_body = Eigen::Tensor<double, 4>(dim, dim, dim, dim);
    
    integrals_kinetic_beta.clear();
    integrals_kinetic_beta.push_back(Eigen::MatrixXd::Zero(dim_b, dim_b)); //x
    integrals_kinetic_beta.push_back(Eigen::MatrixXd::Zero(dim_b, dim_b)); //y
    integrals_kinetic_beta.push_back(Eigen::MatrixXd::Zero(dim_b, dim_b)); //z
    integrals_potential_beta = Eigen::MatrixXd::Zero(dim_b, dim_b);
    
    
    //One Particle Part
    for(int k = 0; k < dim_a; k++)
    {
        for(int l = 0; l < dim_a; l++)
        {
            //Kinetic
            for (int axis=0; axis<3; axis++) {
                real_derivative_3d D_a = free_space_derivative<double,3>(*world, axis);
                real_function_3d da_orb_k = D_a(alpha_orbitals[k]);
                real_function_3d da_orb_l = D_a(alpha_orbitals[l]);
                integrals_kinetic_alpha[axis](k, l) = 0.5 * inner(da_orb_k,da_orb_l);
            }
            //Nuclear
            real_function_3d Vnuc_orb_l = (*Vnuc)(alpha_orbitals[l]);
            integrals_potential_alpha(k, l) = inner(alpha_orbitals[k], Vnuc_orb_l);
        }
    }

    for(int m = 0; m < dim_b; m++)
    {
        for(int n = 0; n < dim_b; n++)
        {
            //Kinetic
            for (int axis=0; axis<3; axis++) {
                real_derivative_3d D_b = free_space_derivative<double,3>(*world, axis);
                real_function_3d db_orb_m = D_b(beta_orbitals[m]);
                real_function_3d db_orb_n = D_b(beta_orbitals[n]);
                integrals_kinetic_beta[axis](m, n) = 0.5 * inner(db_orb_m,db_orb_n);
            }
            //Nuclear
            real_function_3d Vnuc_orb_n = (*Vnuc)(beta_orbitals[n]);
            integrals_potential_beta(m, n) = inner(beta_orbitals[m], Vnuc_orb_n);
        }
    }

    integrals_one_body_alpha = integrals_potential_alpha + integrals_kinetic_alpha[0] + integrals_kinetic_alpha[1] + integrals_kinetic_alpha[2];
    integrals_one_body_beta = integrals_potential_beta + integrals_kinetic_beta[0] + integrals_kinetic_beta[1] + integrals_kinetic_beta[2];
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "CalculateOneElectronIntegrals took " << duration.count() << " seconds" << std::endl;

    std::cout << "Alpha 1-body integrals: " << integrals_one_body_alpha << std::endl;
    std::cout << "Beta 1-body integrals: " << integrals_one_body_beta << std::endl;

}

