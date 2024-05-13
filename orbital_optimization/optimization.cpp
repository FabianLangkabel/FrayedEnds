#include "optimization.hpp"

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

void Optimization::ReadInitialOrbitals(std::vector<Orbital> orbs, int number_active_orbitals)
{
    auto start_time = std::chrono::high_resolution_clock::now();

    all_orbitals = orbs;
    active_orbital_indicies.resize(number_active_orbitals);
    std::fill(active_orbital_indicies.begin(), active_orbital_indicies.end(), -1);
    frozen_occupied_orbital_indicies.clear();

    for(int i = 0; i < all_orbitals.size(); i++)
    {
        real_function_3d orb = real_factory_3d(*world);
        load(orb, all_orbitals[i].orbital_file_name);
        all_orbitals[i].function = orb;
        if(all_orbitals[i].orbital_type == "active")
        {
            active_orbital_indicies[all_orbitals[i].active_space_index] = i;
        }
        else if(all_orbitals[i].orbital_type == "frozen_occupied")
        {
            frozen_occupied_orbital_indicies.push_back(i);
        }
    }

    //Print frozen occupied indicies
    std::cout << "Frozen occupied orbitals: ";
    for (auto const& Idx : frozen_occupied_orbital_indicies)
    {
        std::cout << Idx << " ";
    }
    std::cout << std::endl;

    //Print active space indicies
    std::cout << "Active space indicies: ";
    for (auto const& actIdx : active_orbital_indicies)
    {
        std::cout << actIdx << " ";
    }
    std::cout << std::endl;

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
    std::ifstream one_rdm_imput_file, two_rdm_imput_file; 
    one_rdm_imput_file.open(one_rdm_file);
    two_rdm_imput_file.open(two_rdm_file);

    //Wird nichtmehr aus den Files benötigt muss aber trotzdem noch "gelesen" oder Zeile übersprungen werden
    int one_rdm_dim, two_rdm_dim;
    std::string read_temp;
    one_rdm_imput_file >> read_temp;
    one_rdm_dim = std::stoi(read_temp);
    two_rdm_imput_file >> read_temp;
    two_rdm_dim = std::stoi(read_temp);

    int number_active_orbitals = active_orbital_indicies.size();

    Eigen::MatrixXd active_space_one_rdm = Eigen::MatrixXd::Zero(number_active_orbitals, number_active_orbitals);
    Eigen::Tensor<double, 4> active_space_two_rdm = Eigen::Tensor<double, 4>(number_active_orbitals, number_active_orbitals, number_active_orbitals, number_active_orbitals);
    active_space_two_rdm.setZero();

    for(int i = 0; i < number_active_orbitals; i++)
    {
        for(int j = 0; j < number_active_orbitals; j++)
        {
            one_rdm_imput_file >> read_temp;
            active_space_one_rdm(i,j) = std::stod(read_temp);
        }
    }
    for(int i = 0; i < number_active_orbitals; i++)
    {
        for(int j = 0; j < number_active_orbitals; j++)
        {
            for(int k = 0; k < number_active_orbitals; k++)
            {
                for(int l = 0; l < number_active_orbitals; l++)
                {
                    two_rdm_imput_file >> read_temp;
                    active_space_two_rdm(i,j,k,l) = std::stod(read_temp);
                }
            }
        }
    }
    one_rdm_imput_file.close();
    two_rdm_imput_file.close();

    //****************************************
    // Rotate active Space Orbitals
    //****************************************
    {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
        es.compute(active_space_one_rdm);
        ActiveSpaceRotationMatrix = es.eigenvectors().rowwise().reverse();
        TransformMatrix(&active_space_one_rdm, ActiveSpaceRotationMatrix);
        TransformTensor(&active_space_two_rdm, ActiveSpaceRotationMatrix);
        madness::Tensor<double> T(number_active_orbitals, number_active_orbitals);
        for (int i = 0; i < number_active_orbitals; i++) {
            for (int j = 0; j < number_active_orbitals; j++) {
                T(i,j) = ActiveSpaceRotationMatrix(i,j);
            }
        }
        std::vector<real_function_3d> orbitals_rotate;
        for (auto const& actIdx : active_orbital_indicies)
        {
            orbitals_rotate.push_back(all_orbitals[actIdx].function);
        }
        orbitals_rotate = transform(*world, orbitals_rotate, T);
        for(int i = 0; i < number_active_orbitals; i++)
        {
            int actIdx = active_orbital_indicies[i];
            all_orbitals[actIdx].function = orbitals_rotate[i];
        }
    }

    //****************************************
    // Build full RDMs
    //****************************************
    {
        //Create full RDMs
        int number_orbitals = all_orbitals.size();
        full_one_rdm = Eigen::MatrixXd::Zero(number_orbitals, number_orbitals);
        full_two_rdm = Eigen::Tensor<double, 4>(number_orbitals, number_orbitals, number_orbitals, number_orbitals);
        full_two_rdm.setZero();

        //Fill in active Orbital
        for(int i = 0; i < number_active_orbitals; i++)
        {
            int full_i = active_orbital_indicies[i];
            for(int j = 0; j < number_active_orbitals; j++)
            {
                int full_j = active_orbital_indicies[j];
                full_one_rdm(full_i, full_j) = active_space_one_rdm(i, j);
                for(int k = 0; k < number_active_orbitals; k++)
                {
                    int full_k = active_orbital_indicies[k];
                    for(int l = 0; l < number_active_orbitals; l++)
                    {
                        int full_l = active_orbital_indicies[l];
                        full_two_rdm(full_i, full_j, full_k, full_l) = active_space_two_rdm(i, j, k, l);
                    }
                }
            }
        }

        //Fill in Frozen Core orbitals
        for (int const& frozen_occ_idx : frozen_occupied_orbital_indicies)
        {
            full_one_rdm(frozen_occ_idx, frozen_occ_idx) = 2.0;
            full_two_rdm(frozen_occ_idx, frozen_occ_idx, frozen_occ_idx, frozen_occ_idx) = 2.0;

            //Frozen Orbtial with active Orbitals
            for (int const& act_idx_1 : active_orbital_indicies)
            {
                for (int const& act_idx_2 : active_orbital_indicies)
                {
                    full_two_rdm(frozen_occ_idx, act_idx_1, frozen_occ_idx, act_idx_2) = 2.0 * full_one_rdm(act_idx_1, act_idx_2);
                    full_two_rdm(act_idx_1, frozen_occ_idx, act_idx_2, frozen_occ_idx) = 2.0 * full_one_rdm(act_idx_1, act_idx_2);
                    full_two_rdm(frozen_occ_idx, act_idx_1, act_idx_2, frozen_occ_idx) = -1.0 * full_one_rdm(act_idx_1, act_idx_2);
                    full_two_rdm(act_idx_1, frozen_occ_idx, frozen_occ_idx, act_idx_2) = -1.0 * full_one_rdm(act_idx_1, act_idx_2);
                }
            }

            //Frozen orbitals with each other
            for (int const& frozen_occ_idx_2 : frozen_occupied_orbital_indicies)
            {
                if(frozen_occ_idx != frozen_occ_idx_2)
                {
                    full_two_rdm(frozen_occ_idx, frozen_occ_idx_2, frozen_occ_idx, frozen_occ_idx_2) = 4;
                    full_two_rdm(frozen_occ_idx, frozen_occ_idx_2, frozen_occ_idx_2, frozen_occ_idx) = -2;
                }
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "ReadRDMFiles took " << duration.count() << " seconds" << std::endl;
}

void Optimization::RotateOrbitalsAndIntegralsBack()
{
    int number_all_orbitals = all_orbitals.size();
    int number_active_orbitals = active_orbital_indicies.size();

    //Create Full RotationMatrix
    Eigen::MatrixXd RotationMatrixBack = ActiveSpaceRotationMatrix.transpose();
    Eigen::MatrixXd Full_RotationMatrixBack = Eigen::MatrixXd::Identity(number_all_orbitals, number_all_orbitals);
    for(int i = 0; i < number_active_orbitals; i++)
    {
        for(int j = 0; j < number_active_orbitals; j++)
        {
            Full_RotationMatrixBack(active_orbital_indicies[i], active_orbital_indicies[j]) = RotationMatrixBack(i,j);
        }
    }

    //Transform RDMs
    TransformMatrix(&full_one_rdm, Full_RotationMatrixBack);
    TransformTensor(&full_two_rdm, Full_RotationMatrixBack);

    //Transform active Orbitals
    madness::Tensor<double> T(number_active_orbitals, number_active_orbitals);
    for (int i = 0; i < number_active_orbitals; i++) {
        for (int j = 0; j < number_active_orbitals; j++) {
            T(i,j) = RotationMatrixBack(i,j);
        }
    }
    std::vector<real_function_3d> orbitals_rotate;
    for (auto const& actIdx : active_orbital_indicies)
    {
        orbitals_rotate.push_back(all_orbitals[actIdx].function);
    }
    orbitals_rotate = transform(*world, orbitals_rotate, T);
    for(int i = 0; i < number_active_orbitals; i++)
    {
        int actIdx = active_orbital_indicies[i];
        all_orbitals[actIdx].function = orbitals_rotate[i];
    }

    //Transform Integrals
    integrals_one_body = integrals_potential + integrals_kinetic[0] + integrals_kinetic[1] + integrals_kinetic[2];
    TransformMatrix(&integrals_one_body, Full_RotationMatrixBack);
    TransformTensor(&integrals_two_body, Full_RotationMatrixBack);
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
    real_convolution_3d coul_op = CoulombOperator(*world, 0.001, 1e-6);

    std::vector<real_function_3d> orbitals;
    for (auto const& orb : all_orbitals)
    {
        orbitals.push_back(orb.function);
    }
    int dim = orbitals.size();

    integrals_kinetic.clear();
    integrals_kinetic.push_back(Eigen::MatrixXd::Zero(dim, dim)); //x
    integrals_kinetic.push_back(Eigen::MatrixXd::Zero(dim, dim)); //y
    integrals_kinetic.push_back(Eigen::MatrixXd::Zero(dim, dim)); //z
    integrals_potential = Eigen::MatrixXd::Zero(dim, dim);
    integrals_two_body = Eigen::Tensor<double, 4>(dim, dim, dim, dim);

    //One Particle Part
    for(int k = 0; k < dim; k++)
    {
        for(int l = 0; l < dim; l++)
        {
            //Kinetic
            for (int axis=0; axis<3; axis++) {
                real_derivative_3d D = free_space_derivative<double,3>(*world, axis);
                real_function_3d d_orb_k = D(orbitals[k]);
                real_function_3d d_orb_l = D(orbitals[l]);
                integrals_kinetic[axis](k, l) = 0.5 * inner(d_orb_k,d_orb_l);
            }
            //Nuclear
            real_function_3d Vnuc_orb_l = (*Vnuc)(orbitals[l]);
            integrals_potential(k, l) = inner(orbitals[k], Vnuc_orb_l);
        }
    }

    orbs_kl.clear();
    for(int k = 0; k < dim; k++)
    {
        std::vector<real_function_3d> kl = orbitals[k] * orbitals;
        orbs_kl.insert(std::end(orbs_kl), std::begin(kl), std::end(kl));
    }

    auto coul_op_parallel = std::shared_ptr<real_convolution_3d>(CoulombOperatorPtr(*world, 0.001, 1e-6));
    coul_orbs_mn = apply(*world, *coul_op_parallel, orbs_kl);

    for(int k = 0; k < dim; k++)
    {
        for(int l = 0; l < dim; l++)
        {
            //std::vector<real_function_3d> elements = inner(orbs_kl[k*dim + l], orbs_mn);
            for(int m = 0; m < dim; m++)
            {
                for(int n = 0; n < dim; n++)
                {
                    integrals_two_body(k, l, m, n) = inner(orbs_kl[k*dim + l], coul_orbs_mn[m*dim + n]); 
                }
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "CalculateAllIntegrals took " << duration.count() << " seconds" << std::endl;
}

void Optimization::UpdateIntegrals()
{
    auto start_time = std::chrono::high_resolution_clock::now();
    real_convolution_3d coul_op = CoulombOperator(*world, 0.001, 1e-6);
    int dim = all_orbitals.size();

    //****************************************
    // Update one electron integrals
    //****************************************
    for(int k = 0; k < dim; k++)
    {
        for(int l = 0; l < dim; l++)
        {
            bool is_k_active_orb = std::find(active_orbital_indicies.begin(), active_orbital_indicies.end(), k) != active_orbital_indicies.end();
            bool is_l_active_orb = std::find(active_orbital_indicies.begin(), active_orbital_indicies.end(), l) != active_orbital_indicies.end();   
            if(is_k_active_orb || is_l_active_orb)
            {
                //Kinetic
                for (int axis=0; axis<3; axis++) {
                    real_derivative_3d D = free_space_derivative<double,3>(*world, axis);
                    real_function_3d d_orb_k = D(all_orbitals[k].function);
                    real_function_3d d_orb_l = D(all_orbitals[l].function);
                    integrals_kinetic[axis](k, l) = 0.5 * inner(d_orb_k,d_orb_l);
                }
                //Nuclear
                real_function_3d Vnuc_orb_l = (*Vnuc)(all_orbitals[l].function);
                integrals_potential(k, l) = inner(all_orbitals[k].function, Vnuc_orb_l);
            }
        }
    }

    auto coul_op_parallel = std::shared_ptr<real_convolution_3d>(CoulombOperatorPtr(*world, 0.001, 1e-6));


    //****************************************
    // Update two electron integrals
    //****************************************

    //Fill Orbital Lists
    std::vector<real_function_3d> orbital_list_frozen_act, orbital_list_act_frozen, orbital_list_act_act;
    for (auto const& frozenIdx : frozen_occupied_orbital_indicies)
    {
        for (auto const& actIdx : active_orbital_indicies)
        {
            orbital_list_frozen_act.push_back(all_orbitals[frozenIdx].function * all_orbitals[actIdx].function);
        }
    }
    for (auto const& actIdx : active_orbital_indicies)
    {
        for (auto const& frozenIdx : frozen_occupied_orbital_indicies)
        {
            orbital_list_act_frozen.push_back(all_orbitals[actIdx].function * all_orbitals[frozenIdx].function);
        }
    }
    for (auto const& actIdx_1 : active_orbital_indicies)
    {
        for (auto const& actIdx_2 : active_orbital_indicies)
        {
            orbital_list_act_act.push_back(all_orbitals[actIdx_1].function * all_orbitals[actIdx_2].function);
        }
    }

    //Calculate CoulombOperator(Orbital Lists) -> EXPENSIVE PART!!!
    std::vector<real_function_3d> orbital_list_coul_frozen_act, orbital_list_coul_act_frozen, orbital_list_coul_act_act;
    orbital_list_coul_frozen_act = apply(*world, *coul_op_parallel, orbital_list_frozen_act);
    orbital_list_coul_act_frozen = apply(*world, *coul_op_parallel, orbital_list_act_frozen);
    orbital_list_coul_act_act = apply(*world, *coul_op_parallel, orbital_list_act_act);

    //Update |kl> and |1/r|mn> in full orbital space
    for(int i = 0; i < frozen_occupied_orbital_indicies.size(); i++)
    {
        for(int j = 0; j < active_orbital_indicies.size(); j++)
        {
            int full_idx1 = frozen_occupied_orbital_indicies[i];
            int full_idx2 = active_orbital_indicies[j];
            orbs_kl[full_idx1 * dim + full_idx2] = orbital_list_frozen_act[i * active_orbital_indicies.size() + j];
            coul_orbs_mn[full_idx1 * dim + full_idx2] = orbital_list_coul_frozen_act[i * active_orbital_indicies.size() + j];
        }
    }
    for(int i = 0; i < active_orbital_indicies.size(); i++)
    {
        for(int j = 0; j < frozen_occupied_orbital_indicies.size(); j++)
        {
            int full_idx1 = active_orbital_indicies[i];
            int full_idx2 = frozen_occupied_orbital_indicies[j];
            orbs_kl[full_idx1 * dim + full_idx2] = orbital_list_act_frozen[i * frozen_occupied_orbital_indicies.size() + j];
            coul_orbs_mn[full_idx1 * dim + full_idx2] = orbital_list_coul_act_frozen[i * frozen_occupied_orbital_indicies.size() + j];
        }
    }
    for(int i = 0; i < active_orbital_indicies.size(); i++)
    {
        for(int j = 0; j < active_orbital_indicies.size(); j++)
        {
            int full_idx1 = active_orbital_indicies[i];
            int full_idx2 = active_orbital_indicies[j];
            orbs_kl[full_idx1 * dim + full_idx2] = orbital_list_act_act[i * active_orbital_indicies.size() + j];
            coul_orbs_mn[full_idx1 * dim + full_idx2] = orbital_list_coul_act_act[i * active_orbital_indicies.size() + j];
        }
    }

    for(int k = 0; k < dim; k++)
    {
        for(int l = 0; l < dim; l++)
        {
            for(int m = 0; m < dim; m++)
            {
                for(int n = 0; n < dim; n++)
                {
                    bool is_k_active_orb = std::find(active_orbital_indicies.begin(), active_orbital_indicies.end(), k) != active_orbital_indicies.end();
                    bool is_l_active_orb = std::find(active_orbital_indicies.begin(), active_orbital_indicies.end(), l) != active_orbital_indicies.end();  
                    bool is_m_active_orb = std::find(active_orbital_indicies.begin(), active_orbital_indicies.end(), m) != active_orbital_indicies.end();
                    bool is_n_active_orb = std::find(active_orbital_indicies.begin(), active_orbital_indicies.end(), n) != active_orbital_indicies.end();  
                    if(is_k_active_orb || is_l_active_orb || is_m_active_orb || is_n_active_orb)
                    {
                        integrals_two_body(k, l, m, n) = inner(orbs_kl[k*dim + l], coul_orbs_mn[m*dim + n]);
                    }
                }
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "UpdateIntegrals took " << duration.count() << " seconds" << std::endl;
}

void Optimization::CalculateEnergies()
{
    auto start_time = std::chrono::high_resolution_clock::now();
    int dim = full_one_rdm.rows();

    double kinetic_energy = 0.0;
    double nuclear_attraction_energy = 0.0;

    //One Particle Part
    for(int k = 0; k < dim; k++)
    {
        for(int l = 0; l < dim; l++)
        {
            //Kinetic
            for (int axis=0; axis<3; axis++) {
                kinetic_energy += full_one_rdm(k, l) * integrals_kinetic[axis](k, l);
            }
            //Nuclear
            nuclear_attraction_energy += full_one_rdm(k, l) * integrals_potential(k, l);
        }
    }

    //Two Particle Part
    double two_electron_energy = 0.0;
    for(int k = 0; k < dim; k++)
    {
        for(int l = 0; l < dim; l++)
        {
            for(int m = 0; m < dim; m++)
            {
                for(int n = 0; n < dim; n++)
                {
                    two_electron_energy += full_two_rdm(k, l, m, n) * 0.5 * integrals_two_body(k, m, l, n); 
                }
            }
        }
    }

    double total_energy = kinetic_energy + two_electron_energy + nuclear_attraction_energy + nuclear_repulsion_energy;

    print("            Kinetic energy ", kinetic_energy);
    print(" Nuclear attraction energy ", nuclear_attraction_energy);
    print("       Two-electron energy ", two_electron_energy);
    print(" Nuclear  repulsion energy ", nuclear_repulsion_energy);
    print("              Total energy ", total_energy);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "CalculateEnergies took " << duration.count() << " seconds" << std::endl;
}

void Optimization::CalculateLagrangeMultiplier()
{
    auto start_time = std::chrono::high_resolution_clock::now();

    int dim = all_orbitals.size();
    LagrangeMultiplier = Eigen::MatrixXd::Zero(dim, dim);
    for(int a = 0; a < dim; a++)
    {
        for(int i = 0; i < dim; i++)
        {
            LagrangeMultiplier(a, i) = CalculateLagrangeMultiplierElement(dim, a, i);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "CalculateLagrangeMultiplier took " << duration.count() << " seconds" << std::endl;
}

double Optimization::CalculateLagrangeMultiplierElement(int dim, int a, int i)
{
    double element = integrals_potential(a, i);
    for (int axis=0; axis<3; axis++) {
        element += integrals_kinetic[axis](a, i);
    }
    element = full_one_rdm(i,i) * element;

    for(int l = 0; l < dim; l++)
    {
        for(int n = 0; n < dim; n++)
        {
            for(int k = 0; k < dim; k++)
            {
                element += full_two_rdm(k, l, i, n) * integrals_two_body(a, k, l, n);
            }
        }
    }
    return element;
}

void Optimization::OptimizeOrbitals(double optimization_thresh, double NO_occupation_thresh)
{
    int number_active_orbitals = active_orbital_indicies.size();
    std::vector<real_function_3d> frozen_orbs;
    for (auto const& frozenIdx : frozen_occupied_orbital_indicies)
    {
        frozen_orbs.push_back(all_orbitals[frozenIdx].function);
    }
    std::vector<real_function_3d> orbitals;
    for (auto const& orb : all_orbitals)
    {
        orbitals.push_back(orb.function);
    }

    //Initialize KAIN Solver
    int kain_subspace = 3;
    std::shared_ptr<XNonlinearSolver<vector_real_function_3d, double, KAIN_allocator>> solver  = std::make_shared<XNonlinearSolver<vector_real_function_3d, double, KAIN_allocator>>(KAIN_allocator(*world, number_active_orbitals), true);
    solver->set_maxsub(kain_subspace);

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

        //Solver update
        //std::vector<real_function_3d> rs;
        //for(int i = number_frozen_orbs; i < number_orbitals; i++) {
        //    rs.push_back(GetOrbitalUpdate(i));
        //}
        //orbitals = solver->update(orbitals, rs);

        //Fixed-Point update
        for (auto const& actIdx : active_orbital_indicies)
        {
            if(abs(full_one_rdm(actIdx, actIdx)) < NO_occupation_thresh)
            {
                std::cout << "Skip optimization of orbital " << actIdx << ", since the occupation is less than NO_occupation_thresh (" << NO_occupation_thresh << ")" << std::endl;
            }
            else
            {
                orbitals[actIdx] = orbitals[actIdx] - GetOrbitalUpdate(actIdx);
            }
        }

        //Info: Madness kann Operator auf alle Elemente in Vektor anwenden

        auto end_orb_update_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_orb_update_time - start_orb_update_time);
        std::cout << "UpdateOrbitals took " << duration.count() << " seconds" << std::endl;

        std::cout << "Orthonormalize orbitals" << std::endl;
        //Project out FrozenOrbitals
        if(frozen_orbs.size() > 0)
        {
            auto Q = QProjector(*world, frozen_orbs);
            for (auto const& actIdx : active_orbital_indicies)
            {
                orbitals[actIdx] = Q(orbitals[actIdx]);
            }
        }

        //Orthonormalize orbitals
        orbitals = orthonormalize_symmetric(orbitals);
        for(int i = 0; i < orbitals.size(); i++)
        {
            orbitals[i] = orbitals[i].truncate(1e-5);
        }

        //Update all_orbitals
        for (int i = 0; i < all_orbitals.size(); i++)
        {
            all_orbitals[i].function = orbitals[i];
        }

        std::cout << "Update Integrals" << std::endl;
        //Update integrals for new orbitals
        UpdateIntegrals();
        //CalculateAllIntegrals();

        //Calculate new energy
        CalculateEnergies();

        //Check convergence
        std::cout << "Highest error: " << highest_error << std::endl;
        if(highest_error < optimization_thresh){converged = true;}
    }
}

real_function_3d Optimization::GetOrbitalUpdate(int i)
{
    std::cout << "Orbital: " << i << std::endl;
    int dim = all_orbitals.size();
    double thresh = 1e-6; // precision
    double rdm_ii_inv = 1 / full_one_rdm(i,i);
    double en = LagrangeMultiplier(i,i) * rdm_ii_inv;

    //Calculate rhs
    real_convolution_3d coul_op = CoulombOperator(*world, 0.001, 1e-6);
    real_function_3d rhs = (*Vnuc)(all_orbitals[i].function);
    for(int k = 0; k < dim; k++)
    {
        if(k != i)
        {
            rhs -= rdm_ii_inv * LagrangeMultiplier(k,i) * all_orbitals[k].function;
        }
    }

    //Alte 1. Variante -> Nachteil: Berechnet schon bekannte Integrale neu
    /*
    std::vector<real_function_3d> coul_orbs_lnk;
    std::vector<real_function_3d> orbitals_rdm_ii_inv = orbitals * rdm_ii_inv;
    for(int i = 0; i < coul_orbs_mn.size(); i++)
    {
        std::vector<real_function_3d> lnk = coul_orbs_mn[i] * orbitals_rdm_ii_inv;
        coul_orbs_lnk.insert(std::end(coul_orbs_lnk), std::begin(lnk), std::end(lnk));
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    //Eventuell noch v.sum() nutzen!!!
    for(int l = 0; l < dim; l++)
    {
        for(int n = 0; n < dim; n++)
        {
            for(int k = 0; k < dim; k++)
            {
                rhs += full_two_rdm(k, l, i, n) * coul_orbs_lnk[l*dim*dim + n*dim + k];
            }
        }
    }
    */

   //2. Variante -> Nachteil loopt auch über 2rdm Einträge die bekannterweise = 0 sind (unnötig)
   for(int l = 0; l < dim; l++)
    {
        for(int n = 0; n < dim; n++)
        {
            for(int k = 0; k < dim; k++)
            {
                if(abs(full_two_rdm(k, l, i, n)) > 1e-12)
                {
                    rhs += full_two_rdm(k, l, i, n) * rdm_ii_inv * coul_orbs_mn[l*dim + n] * all_orbitals[k].function;
                }
            }
        }
    }
    
    //3. Variante ToDo
    /*
    //Interaction of active Orbital i with frozen Orbitals
    for(int j = 0; j < number_frozen_orbs; j++)
    {
        rhs += two_rdm(i, j, i, j) * rdm_ii_inv * coul_orbs_mn[j*dim + j] * orbitals[i];
        rhs += two_rdm(j, i, i, j) * rdm_ii_inv * coul_orbs_mn[i*dim + j] * orbitals[j];
    }

    //Interaction between active Orbitals
    for(int l = number_frozen_orbs; l < dim; l++)
    {
        for(int n = number_frozen_orbs; n < dim; n++)
        {
            for(int k = number_frozen_orbs; k < dim; k++)
            {
                if(abs(two_rdm(k, l, i, n)) > 1e-10)
                {
                    //rhs += two_rdm(k, l, i, n) * coul_orbs_lnk[l*dim*dim + n*dim + k];
                    rhs += two_rdm(k, l, i, n) * rdm_ii_inv * coul_orbs_mn[l*dim + n] * orbitals[k];
                }
            }
        }
    }
    */

    SeparatedConvolution<double,3> bsh_op = BSHOperator<3>(*world, sqrt(-2*en), 0.01, thresh);
    real_function_3d r = all_orbitals[i].function + 2.0 * bsh_op(rhs); // the residual
    double err = r.norm2();
    std::cout << "Error of Orbital " << i << ": " << err << std::endl; 
    if(err > highest_error){highest_error = err; }
    return r;
} 

void Optimization::SaveOrbitals(std::string OutputPath)
{
    for(int i = 0; i < all_orbitals.size(); i++)
    {
        std::string base_filename = all_orbitals[i].orbital_file_name.substr(all_orbitals[i].orbital_file_name.find_last_of("/\\") + 1);
        save(all_orbitals[i].function, OutputPath + "/" + base_filename); // ohne das .00000 im filename
    }
}

void Optimization::SaveIntegralsToNumpy(std::string OutputPath)
{
    int number_orbitals = all_orbitals.size();

    std::vector<double> one_e_integrals_elements;
    for(int i = 0; i < number_orbitals; i++)
    {
        for(int j = 0; j < number_orbitals; j++)
        {
            one_e_integrals_elements.push_back(integrals_one_body(i,j));
        }
    }
    std::vector<unsigned long> one_e_ints_shape{(unsigned long)number_orbitals, (unsigned long)number_orbitals};
    const npy::npy_data<double> one_e_data{one_e_integrals_elements, one_e_ints_shape, false};
    npy::write_npy(OutputPath + "/htensor.npy", one_e_data);

    std::vector<double> two_e_integrals_elements;
    for(int i = 0; i < number_orbitals; i++)
    {
        for(int j = 0; j < number_orbitals; j++)
        {
            for(int k = 0; k < number_orbitals; k++)
            {
                for(int l = 0; l < number_orbitals; l++)
                {
                    two_e_integrals_elements.push_back(integrals_two_body(i,j,k,l));
                }
            }
        }
    }
    std::vector<unsigned long> two_e_ints_shape{(unsigned long)number_orbitals, (unsigned long)number_orbitals, (unsigned long)number_orbitals, (unsigned long)number_orbitals};
    const npy::npy_data<double> two_e_data{two_e_integrals_elements, two_e_ints_shape, false};
    npy::write_npy(OutputPath + "/gtensor.npy", two_e_data);
}